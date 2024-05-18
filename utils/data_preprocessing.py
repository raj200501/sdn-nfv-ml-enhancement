import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_traffic_data(data_path):
    df = pd.read_csv(data_path)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df['traffic'].values.reshape(-1, 1))
    return data, scaler

def preprocess_anomaly_data(data_path):
    df = pd.read_csv(data_path)
    X = df.drop('anomaly', axis=1).values
    y = df['anomaly'].values
    return X, y
