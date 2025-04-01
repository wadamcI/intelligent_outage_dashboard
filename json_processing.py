# Standard library imports
import json
import logging
import multiprocessing as mp
import os
import re
import time
from datetime import datetime
from functools import partial, reduce
from pathlib import Path
from typing import List, Tuple, Dict

# Third-party library imports
import geopandas as gpd
import pandas as pd
import pytz
import requests
import requests_cache
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

# Specialized libraries
from census import Census
from retry_requests import retry
import openmeteo_requests
import warnings

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

def get_json_files(folder_path: Path, max_files: int) -> List[Path]:
    """Get list of JSON files to process.

    Args:
        folder_path: Directory path to search for JSON files
        max_files: Maximum number of files to return. If 0 or None, returns all files

    Returns:
        List[Path]: List of Path objects for JSON files found
    """
    try:
        files = list(Path(folder_path).glob('**/*.json'))
        if not files:
            logger.warning(f"No JSON files found in {folder_path}")  # Fixed variable name
            return []

        if max_files:
            files = files[:max_files]

        logger.info(f"Found {len(files)} JSON files to process")
        return files

    except Exception as e:
        logger.error(f"Error getting JSON files: {e}")
        return []


def import_single_file(filename: Path) -> pd.DataFrame | None:
    """Import a single JSON file and process its contents into a DataFrame.
    Handles multiple filename patterns for PG&E outage data.

    Args:
        filename: Path object pointing to the JSON file

    Returns:
        pd.DataFrame | None: Processed DataFrame or None if file cannot be processed
    """
    filename_str = str(filename.name)
    # Check for tree pattern first - if found, reject immediately
    tree_pattern = r"^\d+\.tree[a-f0-9]+.*pge-outages\.json$"
    if re.match(tree_pattern, filename_str):
        return None

    # Try original pattern with flexible timezone (+ or -)
    pattern1 = r"^\d+\.(\d{4}-\d{2}-\d{2})(\d{2})(\d{2})(\d{2})([+-]\d{4}).*\.json$"
    match = re.match(pattern1, filename_str)

    if not match:
        # Try Z timezone pattern
        pattern2 = r"^\d+\.(\d{4}-\d{2}-\d{2})(\d{2})(\d{2})(\d{2})Z.*\.json$"
        match = re.match(pattern2, filename_str)
        if match:
            try:
                date = match.group(1)  # YYYY-MM-DD
                hour = match.group(2)  # HH
                minute = match.group(3)  # MM
                second = match.group(4)  # SS
                datetime_str = f"{date} {hour}:{minute}:{second}"
                datetime_obj = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
                datetime_obj = datetime_obj.replace(tzinfo=pytz.UTC)  # Z means UTC
                datetime_naive = datetime_obj.replace(tzinfo=None)
            except Exception as e:
                logger.error(f"Error processing Z-format datetime in file {filename_str}: {e}")
                return None
        else:
            logger.warning(f"Filename {filename_str} doesn't match any expected pattern")
            return None
    else:
        try:
            date = match.group(1)  # YYYY-MM-DD
            hour = match.group(2)  # HH
            minute = match.group(3)  # MM
            second = match.group(4)  # SS
            timezone = match.group(5) #+-00ZZ
            datetime_str = f"{date} {hour}:{minute}:{second} {timezone}"
            datetime_obj = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S %z")
            datetime_naive = datetime_obj.astimezone(pytz.UTC).replace(tzinfo=None)
        except Exception as e:
            logger.error(f"Error processing datetime in file {filename_str}: {e}")
            return None

    try:
        # Read and process data
        with open(filename, "r") as f:
            data_content = json.load(f)
            df = pd.DataFrame(data_content)
            df = df.apply(pd.to_numeric, errors='ignore')
            df["datetime"] = datetime_naive

        # Standardize column names
        df.columns = df.columns.str.lower().str.replace(' ', '_')

        # Drop specified columns
        columns_to_drop = {
            "outage_start_text",
            "last_update_text",
            "current_etor_text",
            "spid"
        }
        df.drop(columns=list(columns_to_drop.intersection(df.columns)),
                inplace=True)
        return df

    except Exception as e:
        logger.error(f"Error processing file {filename_str}: {e}")
        return None

def multiprocess_files(folder_path: Path, max_files: int) -> gpd.GeoDataFrame | None:
    """Process multiple JSON files in parallel using multiprocessing.

    Args:
        folder_path: Directory path containing JSON files
        max_files: Maximum number of files to process

    Returns:
        gpd.GeoDataFrame | None: GeoDataFrame with Point geometry, latest datetime and maximum customers affected

    The function:
        1. Gets list of JSON files
        2. Processes files in parallel using multiprocessing
        3. Concatenates valid results into single DataFrame
        4. Groups by outage ID keeping latest datetime and max customers affected
        5. Converts to GeoDataFrame with Point geometry
    """
    try:
        # Get list of files to process
        files = get_json_files(folder_path, max_files)
        if not files:
            logger.error("No files found to process")
            return None
        total_files = len(files)
        if max_files:
            total_files = max_files

        logger.info(f"Processing {total_files} files...")

        # Process files in parallel
        with mp.Pool(processes=mp.cpu_count()) as pool:
            # Initialize counters
            processed = 0
            valid_dfs = []

            # Process files with progress bar
            for result in pool.imap(import_single_file, files):
                processed += 1
                if result is not None and not result.empty:
                    valid_dfs.append(result)

                # Update progress bar
                progress = f"[{'=' * (20 * processed // total_files):{20}s}] {processed}/{total_files} files"
                print(f"\r{progress}", end='', flush=True)

        print()  # New line after progress bar

        if not valid_dfs:
            logger.error("No valid data found in any files")
            return None

        # Combine results
        df = pd.concat(valid_dfs, ignore_index=True)
        # logger.info(f"Successfully processed {len(valid_dfs)} out of {len(files)} files")
        logger.info(f"Total raw records: {len(df)}")

        # Group by outage ID and get maximum customers affected and latest datetime
        df = df.groupby("outagenumber").agg({
            'estcustaffected': 'max',
            "outagestarttime": "min",
            'datetime': 'max',
            "cause": "first",
            'latitude': 'first',  # Keep coordinates from first occurrence
            'longitude': 'first',
            "regionname": "first"
        }).reset_index()

        # Convert outagestarttime from Unix timestamp to datetime
        if 'outagestarttime' in df.columns:
            try:
                # Convert Unix timestamp in seconds to datetime
                df['outagestarttime'] = pd.to_datetime(df['outagestarttime'], unit='s', errors='coerce')
                logger.info("Converted outagestarttime from seconds timestamp")
                if df['outagestarttime'].isna().any():

                    logger.warning("Some outagestarttime values could not be converted and were set to NaT")
            except Exception as e:
                logger.warning(f"Could not convert outagestarttime: {e}")

        # Convert to GeoDataFrame with Point geometry
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.longitude, df.latitude),
            crs="EPSG:4326"  # Standard GPS coordinate system
        )

        logger.info(f"Total unique outages: {len(gdf)}")
        logger.info(f"Maximum customers affected in any outage: {gdf['estcustaffected'].max()}")

        return gdf

    except Exception as e:
        logger.error(f"Error processing files: {e}")
        raise


def load_geographic_data(zip_file: str, county_file: str) -> tuple:
    """Load ZIP code and county boundary data from GeoJSON files."""
    try:
        logger.info("Loading ZIP code data...")
        zip_gdf = gpd.read_file(zip_file)
        zip_gdf.columns = zip_gdf.columns.str.lower().str.replace(' ', '_')

        logger.info("Loading county boundary data...")
        county_gdf = gpd.read_file(county_file)
        county_gdf.columns = county_gdf.columns.str.lower().str.replace(' ', '_')

        county_name_col = 'county_name'
        if county_name_col not in county_gdf.columns:
            raise ValueError(f"Could not find county name column '{county_name_col}' in county GeoJSON")

        county_gdf['centroid'] = county_gdf.geometry.centroid
        county_gdf['centroid_lat'] = county_gdf.centroid.y
        county_gdf['centroid_lon'] = county_gdf.centroid.x

        return zip_gdf, county_gdf
    except Exception as e:
        logger.error(f"Error loading geographic data: {e}")
        raise

def add_geographic_data(gdf: gpd.GeoDataFrame, zip_gdf: gpd.GeoDataFrame,
                        county_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add ZIP and county information through spatial join.

    Args:
        gdf: GeoDataFrame with outage points
        zip_gdf: GeoDataFrame with ZIP code polygons
        county_gdf: GeoDataFrame with county polygons

    Returns:
        gpd.GeoDataFrame: Original GeoDataFrame enriched with ZIP and county data
    """
    try:
        # Ensure all GeoDataFrames have the same CRS
        if gdf.crs != zip_gdf.crs:
            zip_gdf = zip_gdf.to_crs(gdf.crs)
        if gdf.crs != county_gdf.crs:
            county_gdf = county_gdf.to_crs(gdf.crs)

        # Spatial join with ZIP codes
        logger.info("Adding ZIP code information...")
        zip_join = gpd.sjoin(gdf, zip_gdf, how='left', predicate='within')

        # Spatial join with counties
        logger.info("Adding county information...")
        county_join = gpd.sjoin(gdf, county_gdf, how='left', predicate='within')

        # Add selected columns to original GeoDataFrame
        zip_cols = ['zip_code', 'po_name', 'population', 'pop_sqmi']
        for col in zip_cols:
            if col in zip_join.columns:
                gdf[col] = zip_join[col]

        # Add county information
        county_name_col = 'county_name'  # Assuming this matches the earlier code
        gdf['county_name'] = county_join[county_name_col]
        gdf['centroid_lat'] = county_join['centroid_lat']
        gdf['centroid_lon'] = county_join['centroid_lon']

        # Log some statistics about the joins
        logger.info(f"ZIP code match rate: {(gdf['zip_code'].notna().sum() / len(gdf)) * 100:.1f}%")
        logger.info(f"County match rate: {(gdf['county_name'].notna().sum() / len(gdf)) * 100:.1f}%")

        return gdf

    except Exception as e:
        logger.error(f"Error adding geographic data: {e}")
        raise

def process_chunk(chunk: gpd.GeoDataFrame, zip_gdf: gpd.GeoDataFrame,
                  county_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Process a chunk of the data with spatial joins."""
    try:
        # Ensure CRS alignment
        if chunk.crs != zip_gdf.crs:
            chunk = chunk.to_crs(zip_gdf.crs)

        # Perform spatial joins
        zip_join = gpd.sjoin(chunk, zip_gdf, how='left', predicate='within')
        county_join = gpd.sjoin(chunk, county_gdf, how='left', predicate='within')

        # Add columns to chunk
        zip_cols = ['zip_code', 'po_name', 'population', 'pop_sqmi']
        for col in zip_cols:
            if col in zip_join.columns:
                chunk[col] = zip_join[col]

        chunk['county_name'] = county_join['county_name']
        chunk['centroid_lat'] = county_join['centroid_lat']
        chunk['centroid_lon'] = county_join['centroid_lon']

        return chunk
    except Exception as e:
        logger.error(f"Error processing chunk: {e}")
        raise


def add_geographic_data_parallel(gdf: gpd.GeoDataFrame, zip_gdf: gpd.GeoDataFrame,
                                 county_gdf: gpd.GeoDataFrame, n_processes: int = None) -> gpd.GeoDataFrame:
    """Add ZIP and county information through parallel spatial joins.

    Args:
        gdf: GeoDataFrame with outage points
        zip_gdf: GeoDataFrame with ZIP code polygons
        county_gdf: GeoDataFrame with county polygons
        n_processes: Number of processes to use (defaults to CPU count - 1)

    Returns:
        gpd.GeoDataFrame: Original GeoDataFrame enriched with ZIP and county data
    """
    try:
        if n_processes is None:
            n_processes = max(1, mp.cpu_count() - 1)

        # Calculate chunk size based on data size and number of processes
        chunk_size = max(1, len(gdf) // (n_processes * 4))  # 4 chunks per process for better load balancing

        # Split the GeoDataFrame into chunks
        chunks = [gdf.iloc[i:i + chunk_size] for i in range(0, len(gdf), chunk_size)]

        logger.info(f"Processing {len(chunks)} chunks using {n_processes} processes...")

        # Create a pool of workers
        with mp.Pool(processes=n_processes) as pool:
            # Create a partial function with fixed zip_gdf and county_gdf arguments
            process_chunk_partial = partial(process_chunk, zip_gdf=zip_gdf, county_gdf=county_gdf)

            # Process chunks in parallel
            processed_chunks = pool.map(process_chunk_partial, chunks)

        # Combine processed chunks
        result_gdf = pd.concat(processed_chunks, axis=0)

        # Log statistics
        logger.info(f"ZIP code match rate: {(result_gdf['zip_code'].notna().sum() / len(result_gdf)) * 100:.1f}%")
        logger.info(f"County match rate: {(result_gdf['county_name'].notna().sum() / len(result_gdf)) * 100:.1f}%")

        return result_gdf

    except Exception as e:
        logger.error(f"Error in parallel processing: {e}")
        raise


def get_weather_data(county_gdf: gpd.GeoDataFrame, county_name: str,
                     start_date: pd.Timestamp, end_date: pd.Timestamp,
                     max_retries: int = 3, retry_delay: int = 60) -> pd.DataFrame:
    """
    Retrieve weather data for a specific county's centroid location.
    """
    # Get centroid coordinates for the county
    try:
        county_data = county_gdf[county_gdf['county_name'] == county_name].iloc[0]
        latitude = county_data['centroid_lat']
        longitude = county_data['centroid_lon']
    except (IndexError, KeyError) as e:
        raise ValueError(f"Missing centroid data for {county_name}")

    # Set up caching and retry sessions
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.date().strftime('%Y-%m-%d'),
        "end_date": end_date.date().strftime('%Y-%m-%d'),
        "hourly": [
            "temperature_2m", "relative_humidity_2m", "precipitation",
            "rain", "snowfall", "snow_depth", "weather_code",
            "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"
        ],
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch"
    }

    # Get weather data from API with retries
    for attempt in range(max_retries):
        try:
            responses = openmeteo.weather_api("https://archive-api.open-meteo.com/v1/archive",
                                              params=params)
            response = responses[0]
            break
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            else:
                raise

    try:
        hourly = response.Hourly()
        hourly_data = {
            "datetime": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            )
        }

        # Add all weather variables
        variable_names = params["hourly"]
        for i, var_name in enumerate(variable_names):
            hourly_data[var_name] = hourly.Variables(i).ValuesAsNumpy()

    except Exception as e:
        raise

    # Create DataFrame and add county information
    hourly_df = pd.DataFrame(data=hourly_data)
    hourly_df['county_name'] = county_name
    hourly_df['latitude'] = latitude
    hourly_df['longitude'] = longitude
    hourly_df.set_index('datetime', inplace=True)

    return hourly_df

def get_unique_county_dates(data_gdf: gpd.GeoDataFrame) -> dict:
    """
    Get unique date ranges for each county to avoid redundant API calls.
    """
    county_dates = {}
    # Group by county and get min/max dates
    date_ranges = data_gdf.groupby('county_name')['datetime'].agg(['min', 'max'])

    for county, (start, end) in date_ranges.iterrows():
        county_dates[county] = {
            'start_date': start - pd.Timedelta(days=1),
            'end_date': end + pd.Timedelta(days=1)
        }

    return county_dates


def process_counties_weather(data_gdf: gpd.GeoDataFrame,
                             county_gdf: gpd.GeoDataFrame,
                             chunk_size: int = 1_000_000) -> pd.DataFrame:
    """
    Process weather data for all counties in the GeoDataFrame with optimized memory usage.
    Handles datetime timezone conversion and precision matching for proper merging.
    """
    # Get unique date ranges for each county
    county_dates = get_unique_county_dates(data_gdf)
    total_counties = len(county_dates)

    logger.info("Starting weather data retrieval...")
    weather_dfs = []
    success_count = 0

    # Process counties with progress bar
    for i, (county, dates) in enumerate(county_dates.items(), 1):
        progress = f"[{'=' * (20 * i // total_counties):{20}s}] {i}/{total_counties} counties"
        print(f"\r{progress}", end='', flush=True)

        try:
            county_weather = get_weather_data(
                county_gdf,
                county,
                dates['start_date'],
                dates['end_date']
            )
            # Convert UTC index to timezone-naive and ensure nanosecond precision
            county_weather.index = county_weather.index.tz_localize(None).astype('datetime64[ns]')
            weather_dfs.append(county_weather)
            success_count += 1
        except Exception as e:
            logger.warning(f"Failed to get weather data for {county}: {str(e)}")
            continue

    print()  # New line after progress bar

    if weather_dfs:
        combined_weather = pd.concat(weather_dfs)
        logger.info(f"Successfully processed {success_count}/{total_counties} counties")

        # Process data in chunks
        logger.info("Merging weather data with original dataset...")
        final_dfs = []

        # Calculate number of chunks
        n_chunks = len(data_gdf) // chunk_size + (1 if len(data_gdf) % chunk_size else 0)

        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(data_gdf))

            chunk = data_gdf.iloc[start_idx:end_idx].copy()

            # Handle timezone and precision
            if isinstance(chunk['datetime'].dtype, pd.DatetimeTZDtype):
                chunk['datetime'] = chunk['datetime'].dt.tz_localize(None)

            # Ensure nanosecond precision
            chunk['datetime'] = chunk['datetime'].astype('datetime64[ns]')

            # Merge weather data for this chunk
            merged_chunk = pd.merge_asof(
                chunk.sort_values('datetime'),
                combined_weather.sort_index(),
                left_on='datetime',
                right_index=True,
                by='county_name',
                direction='nearest',
                tolerance=pd.Timedelta('1H')  # Add tolerance for matching
            )

            final_dfs.append(merged_chunk)

            # Update progress
            progress = f"[{'=' * (20 * (i + 1) // n_chunks):{20}s}] {i + 1}/{n_chunks} chunks"
            print(f"\r{progress}", end='', flush=True)

        print()  # New line after progress bar

        # Combine all chunks
        logger.info("Combining processed chunks...")
        final_data = pd.concat(final_dfs, ignore_index=True)

        return final_data
    else:
        raise ValueError("No weather data could be retrieved for any county")

#socioeconomical data from census
def get_census_tables() -> Dict[str, Dict[str, str]]:
    """
    Define census variables focused on vulnerability to power outages.
    Returns dictionary of relevant demographic and socioeconomic variables.
    """
    return {
        # Demographics from 2020 Census
        'pl': {
            'P1_001N': 'total_population',
            'P1_003N': 'white_alone',
            'P1_004N': 'black_alone',
            'P1_005N': 'american_indian_alone',
            'P1_006N': 'asian_alone',
            'P1_007N': 'pacific_islander_alone',
            'P1_008N': 'other_race_alone',
            'P1_009N': 'two_or_more_races',
            'P2_002N': 'hispanic_latino'
        },

        # ACS5 Variables Related to Power Outage Vulnerability
        'acs5': {
            # Age and Elderly
            'B01002_001E': 'median_age',
            'B01001_020E': 'female_65_to_74',
            'B01001_021E': 'female_75_to_84',
            'B01001_022E': 'female_85_plus',
            'B01001_044E': 'male_65_to_74',
            'B01001_045E': 'male_75_to_84',
            'B01001_046E': 'male_85_plus',

            # Medical Vulnerability
            'B18105_001E': 'total_population_disability_status',
            'B18105_004E': 'under_18_with_disability',
            'B18105_007E': 'age_18_64_with_disability',
            'B18105_010E': 'over_65_with_disability',
            'B27010_001E': 'civilian_noninst_population',
            'B27010_017E': 'no_health_insurance',

            # Economic Vulnerability
            'B19013_001E': 'median_household_income',
            'B19131_001E': 'family_income_past_12months',
            'B17020_002E': 'income_below_poverty',
            'B25077_001E': 'median_home_value',

            # Housing and Infrastructure
            'B25001_001E': 'total_housing_units',
            'B25002_003E': 'vacant_housing_units',
            'B25040_001E': 'house_heating_fuel',
            'B25040_002E': 'utility_gas',
            'B25040_003E': 'electricity',
            'B25040_005E': 'fuel_oil',
            'B25040_010E': 'solar_energy',

            # Transportation and Mobility
            'B08014_001E': 'workers_16_and_over',
            'B08014_002E': 'no_vehicle',
            'B08301_001E': 'means_of_transportation_to_work',
            'B08301_003E': 'public_transportation',

            # Communication and Technology
            'B28002_001E': 'presence_of_internet',
            'B28002_004E': 'no_internet_access',
            'B28002_013E': 'cellular_data_with_no_other_internet',

            # Employment and Work from Home
            'B23025_005E': 'civilian_unemployed',
            'B08301_021E': 'worked_from_home'
        }
    }


def get_census_data(api_key: str) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Fetch census data and tract boundaries for California.
    """
    c = Census(api_key)
    years = [2018, 2019, 2020, 2022]
    table_content = get_census_tables()

    def fetch_acs_data(year: int) -> pd.DataFrame:
        try:
            logger.info(f"Fetching ACS data for year {year}")
            acs_year_df = pd.DataFrame(c.acs5.get(
                ('NAME', *list(table_content['acs5'].keys())),
                {'for': 'tract:*', 'in': 'state:06'},
                year=year
            ))
            acs_year_df['year'] = year
            acs_year_df['GEOID'] = (acs_year_df['state'] +
                                    acs_year_df['county'].str.zfill(3) +
                                    acs_year_df['tract'].str.zfill(6))
            return acs_year_df
        except Exception as e:
            logger.error(f"Error fetching ACS data for {year}: {str(e)}")
            return pd.DataFrame()

    # Fetch ACS data
    with ThreadPoolExecutor() as executor:
        acs_dfs = list(executor.map(fetch_acs_data, years))

    acs_combined = pd.concat([df for df in acs_dfs if not df.empty],
                             ignore_index=True)
    if not acs_combined.empty:
        acs_combined.rename(columns=table_content['acs5'], inplace=True)

    # Fetch PL data
    try:
        pl_data = pd.DataFrame(c.pl.get(
            ('NAME', *list(table_content['pl'].keys())),
            {'for': 'tract:*', 'in': 'state:06'},
            year=2020
        ))
        pl_data['GEOID'] = (pl_data['state'] +
                            pl_data['county'].str.zfill(3) +
                            pl_data['tract'].str.zfill(6))
        pl_data.rename(columns=table_content['pl'], inplace=True)
    except Exception as e:
        logger.error(f"Error fetching PL data: {str(e)}")
        pl_data = pd.DataFrame()

    # Fetch TIGER boundaries
    try:
        latest_year = max(years)
        tiger_data = gpd.read_file(
            f"https://www2.census.gov/geo/tiger/TIGER{latest_year}/TRACT/tl_{latest_year}_06_tract.zip",
            columns=['GEOID', 'geometry']
        ).to_crs("EPSG:4326")
    except Exception as e:
        logger.error(f"Error fetching TIGER boundaries: {str(e)}")
        tiger_data = gpd.GeoDataFrame()

    # Create final GeoDataFrames
    acs_gdf = gpd.GeoDataFrame(
        acs_combined.merge(tiger_data, on='GEOID', how='left'),
        geometry='geometry'
    ) if not acs_combined.empty else gpd.GeoDataFrame()

    pl_gdf = gpd.GeoDataFrame(
        pl_data.merge(tiger_data, on='GEOID', how='left'),
        geometry='geometry'
    ) if not pl_data.empty else gpd.GeoDataFrame()

    return acs_gdf, pl_gdf

def merge_outage_census_data(
        outage_gdf: gpd.GeoDataFrame,
        acs_gdf: gpd.GeoDataFrame,
        pl_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Merge outage point data with census tract data.
    """
    try:
        # Reset indexes
        outage_gdf = outage_gdf.reset_index(drop=True)
        acs_gdf = acs_gdf.reset_index(drop=True)
        pl_gdf = pl_gdf.reset_index(drop=True)

        # Merge with ACS data
        outage_acs = gpd.sjoin(
            outage_gdf,
            acs_gdf,
            how="left",
            predicate="within"
        )

        # Merge with PL data
        outage_census = outage_acs.merge(
            pl_gdf.drop(columns=['geometry']),
            on='GEOID',
            how='left',
            suffixes=('_acs', '_pl')
        )

        # Clean up index columns
        index_cols = [col for col in outage_census.columns if col.startswith('index_')]
        outage_census = outage_census.drop(columns=index_cols)

        return outage_census

    except Exception as e:
        logger.error(f"Error merging data: {str(e)}")
        raise



if __name__ == "__main__":
    path1 = Path(r"C:\Users\wadamc\Desktop\pge_outages_pre_24\66011.2019-06-10211502-0700.912d8f835cdb23ce4bfe06329817d3781e54ca4e.pge-outages.json")
    path2 = Path(r"C:\Users\wadamc\Desktop\pge_outages_pre_24\0004.2022-04-21162821-0700.5485d67b7dd2596887c828536a93df5c92896409.pge-outages.json")
    df = import_single_file(path1)
    print(df)
    print(df.dtypes)
    # df = df.apply(pd.to_numeric, errors='ignore')
    # print(df.dtypes)
