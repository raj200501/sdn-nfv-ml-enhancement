import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np

def engineer_features(filepath='data/network_data.csv', output_path='data/network_data_features.csv'):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Input file {filepath} not found. Skipping feature engineering.")
        return
    if df.empty:
        print(f"Input file {filepath} is empty. Skipping feature engineering.")
        return
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        print("No numeric columns available for feature engineering.")
        return
    engineered = numeric_df.copy()
    engineered["feature_sum"] = numeric_df.sum(axis=1)
    engineered["feature_mean"] = numeric_df.mean(axis=1)
    engineered.to_csv(output_path, index=False)
    print(f"Engineered features saved to {output_path}")

if __name__ == "__main__":
    engineer_features()
