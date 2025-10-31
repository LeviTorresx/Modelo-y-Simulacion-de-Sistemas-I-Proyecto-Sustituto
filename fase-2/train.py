import numpy as np
import pandas as pd
from datetime import datetime
from meteostat import Point, Hourly
from lightgbm import LGBMRegressor
import joblib

# =========================================================
# GLOBAL SETTINGS
# =========================================================
DATA_PATH = "./data/train.zip"

# New York coordinates (for the example dataset)
NYC_COORDS = Point(40.7128, -74.0060)

# Date range (adjust according to your data)
START_DATE = datetime(2015, 12, 31)
END_DATE = datetime(2016, 7, 31)

# =========================================================
# DATA LOADING
# =========================================================
import pandas as pd
import os

def load_data(path: str) -> pd.DataFrame:
    """
    Load dataset (supports .csv, .csv.gz, .zip).
    If it's a .zip, it must contain a single CSV file inside.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    # Detecta tipo de compresión
    if path.endswith(".gz"):
        compression = "gzip"
    elif path.endswith(".zip"):
        compression = "zip"
    else:
        compression = None

    print(f"Loading data from: {path} (compression={compression})")

    df = pd.read_csv(path, compression=compression)

    print(f"Data loaded: {df.shape}")
    print(df.info())
    print(df.describe())

    return df



# =========================================================
# DISTANCE CALCULATION
# =========================================================
def haversine(lat1, lon1, lat2, lon2):
    """Calculate the distance between two coordinates (km) using the Haversine formula."""
    R = 6371.0  #  Terrestrial radio in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def add_distance_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add a column with the distance of the trip."""
    df["distance_km"] = haversine(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"]
    )
    return df


# =========================================================
# TEMPORARY FEATURES
# =========================================================
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add columns related to date and time."""
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["pickup_day"] = df["pickup_datetime"].dt.day
    df["pickup_month"] = df["pickup_datetime"].dt.month
    df["pickup_year"] = df["pickup_datetime"].dt.year
    df["pickup_hour"] = df["pickup_datetime"].dt.hour
    df["pickup_week"] = df["pickup_datetime"].dt.isocalendar().week
    df["pickup_dayofweek"] = df["pickup_datetime"].dt.dayofweek
    df["pickup_datetime_hour_trunc"] = df["pickup_datetime"].dt.floor("h")
    return df


# =========================================================
# WEATHER DATA
# =========================================================
def fetch_weather_data() -> pd.DataFrame:
    """Download hourly weather data from Meteostat."""
    print("Fetching weather data-")
    data_hourly = Hourly(NYC_COORDS, START_DATE, END_DATE).fetch()
    data_hourly = data_hourly.reset_index().rename(columns={"time": "time_ny"})
    print(f"Weather data:{data_hourly.shape}")
    return data_hourly


def merge_weather(df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Combine travel data with weather data according to the time of day."""
    merged = df.merge(weather_df, how="left", left_on="pickup_datetime_hour_trunc", right_on="time_ny")
    print(f"Merge complete: {merged.shape}")
    return merged


# =========================================================
# FEATURE SELECTION AND TRAINING
# =========================================================
FEATURES = [
    "passenger_count",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "distance_km",
    "pickup_day",
    "pickup_hour",
    "pickup_dayofweek",
    "temp",
    "prcp"
]


def train_split(df: pd.DataFrame):
    """Divide the dataset into X (features) and Y (target)."""
    X = df[FEATURES]
    y = df["trip_duration"]
    return X, y


def train_model(X_train, y_train) -> LGBMRegressor:
    """Train the LightGBM model."""
    print("Training LightGBM model-")
    model = LGBMRegressor()
    model.fit(X_train, y_train)
    print("Training complete.")
    return model


# =========================================================
# MAIN FUNCTION
# =========================================================
def main():
    print("Starting training-\n")

    df = load_data(DATA_PATH)

    df = add_distance_feature(df)
    df = add_time_features(df)

    weather_df = fetch_weather_data()
    df = merge_weather(df, weather_df)

    X_train, y_train = train_split(df)
    model = train_model(X_train, y_train)

    
    joblib.dump(model, "./data/model_lgbm.pkl")
    print("Model saved in ./data/model_lgbm.pkl")


# =========================================================
# ENTRY POINT 
# =========================================================
if __name__ == "__main__":
    main()
