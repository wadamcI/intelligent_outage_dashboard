# At the top of your file, after imports
from json_processing import *


def process_pipeline(folder_path: Path, zip_path: str, county_path: str,
                     max_files: int, census_api_key: str) -> pd.DataFrame:
    """
    Main data processing pipeline.
    Processes outage data, geographic data, weather data, and census data.
    Saves result as parquet file.
    """
    # Load geographic data
    logger.info("Loading geographic data...")
    zip_data, county_data = load_geographic_data(zip_path, county_path)

    # Get list of ZIP codes for census data
    zip_codes = zip_data['zip_code'].astype(int).tolist()

    # Process outage data
    logger.info("Processing outage files...")
    outage_data = multiprocess_files(folder_path, max_files)

    # Enrich with geographic data
    logger.info("Enriching with geographic information...")
    enriched_data = add_geographic_data_parallel(outage_data, zip_data, county_data)

    # Process weather data
    logger.info("Processing weather data...")
    try:
        enriched_data = process_counties_weather(
            data_gdf=enriched_data,
            county_gdf=county_data,
            chunk_size=5_000_000
        )
    except Exception as e:
        logger.error(f"Weather data processing failed: {str(e)}")

    # Fetch and merge census data
    logger.info("Fetching census data...")
    try:
        acs_gdf, pl_gdf = get_census_data(census_api_key)
        logger.info("Merging census data with outage data...")
        final_data = merge_outage_census_data(enriched_data, acs_gdf, pl_gdf)
        logger.info("Census data merge complete")
    except Exception as e:
        logger.error(f"Census data processing failed: {str(e)}")
        final_data = enriched_data

    # Convert geometry to WKT for parquet storage
    logger.info("Preparing data for parquet storage...")
    final_data_parquet = final_data.copy()
    final_data_parquet['geometry'] = final_data_parquet['geometry'].apply(lambda x: x.wkt if x else None)
    # Drop rows where all values are NaN/NaT
    final_data_parquet = final_data_parquet.dropna(how='all')

    # Convert numeric columns to appropriate types and handle NaN values
    for col in final_data_parquet.columns:
        # Skip geometry and known string columns
        if col == 'geometry' or final_data_parquet[col].dtype == 'object':
            continue

        # Try to convert to numeric, coercing errors to NaN
        try:
            # First check if datetime
            if pd.api.types.is_datetime64_any_dtype(final_data_parquet[col]):
                continue

            # Convert to numeric
            numeric_series = pd.to_numeric(final_data_parquet[col], errors='coerce')

            # Determine if int or float
            if numeric_series.notna().all() and numeric_series.eq(numeric_series.astype(int)).all():
                final_data_parquet[col] = numeric_series.astype(int)
            else:
                final_data_parquet[col] = numeric_series.astype(float)

        except (ValueError, TypeError):
            continue

    # Drop any remaining NaN/NaT values
    final_data_parquet = final_data_parquet.dropna()

    # Save processed data as parquet
    output_path = 'Datasets\processed_data.parquet'
    logger.info(f"Saving data to {output_path}...")
    final_data_parquet.to_parquet(
        output_path,
        index=False,
        compression='snappy',  # Good balance between compression and speed
        engine='pyarrow'
    )

    # Convert WKT back to geometry for return value
    return final_data


if __name__ == "__main__":

    # Configuration
    load_dotenv()
    FOLDER_PATH = Path(r"C:\Users\wadamc\Desktop\pge_outages_pre_24")
    ZIP_PATH = r"C:\Users\wadamc\Desktop\ZipCodes_-1049704744535259894.geojson"
    COUNTY_PATH = r"C:\Users\wadamc\Desktop\California_County_Boundaries_6550485670014028237.geojson"
    MAX_FILES = 0
    CENSUS_API_KEY = os.getenv("CENSUS_API_KEY")

    # Execute pipeline and measure performance
    start_time = time.time()
    result_df = process_pipeline(FOLDER_PATH, ZIP_PATH, COUNTY_PATH, MAX_FILES, CENSUS_API_KEY)
    execution_time = time.time() - start_time

    # result_df.to_csv('outage_processed_pge.csv', index=False)

    # Print DataFrame information
    print("\nComplete DataFrame Information:")
    print("=" * 80)
    print(f"\nShape: {result_df.shape}")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(f"\nColumns and Types:\n{result_df.dtypes}")
    print(f"\nNull Values:\n{result_df.isnull().sum()}")
    print(f"\nData Sample:\n{result_df.head()}")
    print(f"\nExecution time: {execution_time:.2f} seconds")

    # Demonstrate how to read back the parquet file
    print("\nReading back parquet file:")
    df_from_parquet = pd.read_parquet('Datasets\processed_data.parquet')
    # Convert WKT back to geometry if needed
    df_from_parquet['geometry'] = gpd.GeoSeries.from_wkt(df_from_parquet['geometry'])
    print(f"Parquet file shape: {df_from_parquet.shape}")
