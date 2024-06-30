import pandas as pd
import numpy as np

def clean_data(filepath):
    df = pd.read_csv(filepath)
    # Remove rows with missing values
    df.dropna(inplace=True)
    # Remove rows with non-numeric values
    df = df[df.applymap(lambda x: isinstance(x, (int, float)))]
    # Remove duplicate rows
    df.drop_duplicates(inplace=True)
    # Save cleaned data
    df.to_csv(filepath, index=False)
    print(f"Data cleaned and saved to {filepath}")

if __name__ == "__main__":
    clean_data('data/network_data.csv')
    clean_data('data/anomaly_data.csv')
    clean_data('data/performance_metrics.csv')
