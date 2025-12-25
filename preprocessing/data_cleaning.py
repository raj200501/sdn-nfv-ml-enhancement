import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np

def clean_data(filepath):
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        df = generate_placeholder_dataset(filepath)
        df.to_csv(filepath, index=False)
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

def generate_placeholder_dataset(filepath, rows=200, features=10):
    rng = np.random.default_rng(7)
    is_classification = "anomaly" in filepath
    X = rng.normal(size=(rows, features))
    if is_classification:
        y = (X.mean(axis=1) > 0).astype(int)
        label_name = "label"
    else:
        y = X.sum(axis=1) + rng.normal(scale=0.1, size=rows)
        label_name = "target"
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(features)])
    df[label_name] = y
    return df

if __name__ == "__main__":
    clean_data('data/network_data.csv')
    clean_data('data/anomaly_data.csv')
    clean_data('data/performance_metrics.csv')
