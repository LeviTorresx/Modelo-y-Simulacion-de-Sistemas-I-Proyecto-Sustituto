import pandas as pd
import joblib
import os
from train import (
    add_distance_feature,
    add_time_features,
    fetch_weather_data,
    merge_weather,
    load_data,
    FEATURES
)

# =========================================================
# CONFIG
# =========================================================
DATA_PATH = "./data/test.zip"
MODEL_PATH = "./data/model_lgbm.pkl"
OUTPUT_PATH = "./data/submission.csv"

# =========================================================
# PREDICTION FUNCTIONS
# =========================================================
def load_model(model_path: str):
    """Load the trained model from the .pkl file"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    print(f"Model loaded from: {model_path}")
    return joblib.load(model_path)


def make_predictions(model, df: pd.DataFrame) -> pd.DataFrame:
    """Generate predictions using the same features as in training"""
    X_pred = df[FEATURES]
    df["trip_duration"] = model.predict(X_pred)
    return df[["id", "trip_duration"]] 


# =========================================================
# MAIN FUNCTION
# =========================================================
def main():
    print("Starting prediction- \n")

   
    model = load_model(MODEL_PATH)

   
    test_df = test_df = load_data(DATA_PATH)
    print(f" Loaded data: {test_df.shape}")


    test_df = add_distance_feature(test_df)
    test_df = add_time_features(test_df)

    weather_df = fetch_weather_data()
    test_merged = merge_weather(test_df, weather_df)

    submission = make_predictions(model, test_merged)

    print(f" Predictions made: {submission.shape}")
    print(f" First rows:\n{submission.head()}")
  
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"Predictions saved in: {OUTPUT_PATH}")

# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    main()
