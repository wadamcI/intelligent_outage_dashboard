import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def prepare_vulnerability_dataset(gdf):
    """
    Prepare the dataset for vulnerability prediction

    Parameters:
    -----------
    gdf : GeoDataFrame
        Input geodataframe with outage and demographic data

    Returns:
    --------
    X : DataFrame
        Features for prediction
    y : Series
        Target variable representing vulnerability
    """
    # Aggregate data by zip code
    zip_code_aggregation = gdf.groupby('zip_code').agg({
        # Outage-related features
        'outagenumber': 'count',
        'estcustaffected': 'sum',
        'duration': ['mean', 'max'],

        # Demographic vulnerability indicators
        'total_population': 'first',
        'median_household_income': 'first',
        'over_65_with_disability': 'first',
        'no_health_insurance': 'first',
        'no_vehicle': 'first',

        # Environmental and infrastructure factors
        'temperature_2m': 'mean',
        'wind_speed_10m': 'mean',
        'precipitation': 'sum',

        # Socioeconomic resilience indicators
        'income_below_poverty': 'first',
        'vacant_housing_units': 'first',
        'presence_of_internet': 'first'
    }).reset_index()

    # Flatten multi-level column names
    zip_code_aggregation.columns = [
        '_'.join(col).strip() if isinstance(col, tuple) else col
        for col in zip_code_aggregation.columns.values
    ]

    # Create vulnerability score
    # Higher value indicates more vulnerability
    zip_code_aggregation['vulnerability_score'] = (
        # Outage frequency and impact
            zip_code_aggregation['outagenumber'] *
            zip_code_aggregation['estcustaffected_sum'] *
            zip_code_aggregation['duration_max'] /

            # Normalize by population and income
            (zip_code_aggregation['total_population'] *
             np.log(zip_code_aggregation['median_household_income'] + 1))
    )

    # Prepare features and target
    features_columns = [
        'outagenumber', 'estcustaffected_sum',
        'duration_mean', 'duration_max',
        'total_population', 'median_household_income',
        'over_65_with_disability', 'no_health_insurance',
        'no_vehicle', 'temperature_2m_mean',
        'wind_speed_10m_mean', 'precipitation_sum',
        'income_below_poverty', 'vacant_housing_units',
        'presence_of_internet'
    ]

    X = zip_code_aggregation[features_columns]
    y = zip_code_aggregation['vulnerability_score']

    return X, y, zip_code_aggregation


def train_vulnerability_model(X_train, y_train):
    """
    Train a Random Forest Regressor for vulnerability prediction

    Parameters:
    -----------
    X_train : DataFrame
        Training features
    y_train : Series
        Training target variable

    Returns:
    --------
    Pipeline
        Trained model pipeline
    """
    # Preprocessing for numerical features
    numeric_features = X_train.columns.tolist()

    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features)
        ])

    # Create pipeline
    vulnerability_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10
        ))
    ])

    # Fit the model
    vulnerability_pipeline.fit(X_train, y_train)

    return vulnerability_pipeline


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the vulnerability prediction model

    Parameters:
    -----------
    model : Pipeline
        Trained model
    X_test : DataFrame
        Test features
    y_test : Series
        Test target variable

    Returns:
    --------
    dict
        Model performance metrics
    """
    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    return {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred)
    }


def identify_most_vulnerable_zip_codes(model, X, n_top=10):
    """
    Identify top most vulnerable zip codes

    Parameters:
    -----------
    model : Pipeline
        Trained vulnerability model
    X : DataFrame
        Features for prediction
    n_top : int, optional
        Number of top vulnerable zip codes to return

    Returns:
    --------
    DataFrame
        Top vulnerable zip codes with their scores
    """
    # Predict vulnerability scores
    vulnerability_scores = model.predict(X)

    # Create a DataFrame with zip codes and scores
    vulnerability_df = pd.DataFrame({
        'zip_code': X.index,
        'vulnerability_score': vulnerability_scores
    })

    # Sort and return top vulnerable zip codes
    return vulnerability_df.sort_values('vulnerability_score', ascending=False).head(n_top)


# Main workflow function
def main_vulnerability_analysis(gdf):
    """
    Comprehensive vulnerability analysis workflow

    Parameters:
    -----------
    gdf : GeoDataFrame
        Input geodataframe with outage and demographic data

    Returns:
    --------
    dict
        Analysis results including model, top vulnerable zip codes,
        and performance metrics
    """
    # Prepare dataset
    X, y, full_data = prepare_vulnerability_dataset(gdf)

    # Split data temporally (first 1.5 years for training, last 0.5 years for validation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=False, random_state=42
    )

    # Train model
    vulnerability_model = train_vulnerability_model(X_train, y_train)

    # Evaluate model
    performance_metrics = evaluate_model(vulnerability_model, X_test, y_test)

    # Identify most vulnerable zip codes
    top_vulnerable_zips = identify_most_vulnerable_zip_codes(
        vulnerability_model, X_test
    )

    return {
        'model': vulnerability_model,
        'top_vulnerable_zips': top_vulnerable_zips,
        'performance_metrics': performance_metrics,
        'full_vulnerability_data': full_data
    }

# Example usage (uncomment and modify as needed)
# results = main_vulnerability_analysis(gdf)
# print("Top Vulnerable Zip Codes:")
# print(results['top_vulnerable_zips'])
# print("\nModel Performance:")
# print(results['performance_metrics'])