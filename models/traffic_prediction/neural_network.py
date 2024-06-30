import pandas as pd
import numpy as np

def feature_engineering(filepath):
    df = pd.read_csv(filepath)
    # Example feature engineering: Add statistical features
    df['mean'] = df.mean(axis=1)
    df['std'] = df.std(axis=1)
    df['min'] = df.min(axis=1)
    df['max'] = df.max(axis=1)
    # Save enhanced data
    df.to_csv(filepath, index=False)
    print(f"Feature engineering applied and saved to {filepath}")

if __name__ == "__main__":
    feature_engineering('data/network_data.csv')
    feature_engineering('data/anomaly_data.csv')
    feature_engineering('data/performance_metrics.csv')
