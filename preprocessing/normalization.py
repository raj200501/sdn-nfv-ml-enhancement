import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def normalize_data(filepath):
    df = pd.read_csv(filepath)
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df)
    normalized_df = pd.DataFrame(normalized_data, columns=df.columns)
    normalized_df.to_csv(filepath, index=False)
    print(f"Data normalized and saved to {filepath}")

if __name__ == "__main__":
    normalize_data('data/network_data.csv')
    normalize_data('data/anomaly_data.csv')
    normalize_data('data/performance_metrics.csv')
