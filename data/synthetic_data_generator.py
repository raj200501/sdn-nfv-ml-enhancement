import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, LeakyReLU

def generate_synthetic_data(samples=10000):
    noise_dim = 100
    data_dim = 784
    
    generator = build_generator(noise_dim, data_dim)
    noise = np.random.normal(0, 1, (samples, noise_dim))
    synthetic_data = generator.predict(noise)
    
    df = pd.DataFrame(synthetic_data)
    df.to_csv('data/network_data.csv', index=False)
    print("Synthetic data generated and saved.")

def build_generator(noise_dim, data_dim):
    model = Sequential()
    model.add(Dense(256, input_dim=noise_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(data_dim, activation='tanh'))
    return model

if __name__ == "__main__":
    generate_synthetic_data()

def generate_anomaly_data(samples=1000, features=10):
    rng = np.random.default_rng(123)
    X = rng.normal(size=(samples, features))
    y = (X.mean(axis=1) > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(features)])
    df["label"] = y
    df.to_csv('data/anomaly_data.csv', index=False)
    print("Anomaly data generated and saved.")

def generate_performance_metrics(samples=1000, features=5):
    rng = np.random.default_rng(456)
    X = rng.uniform(0, 1, size=(samples, features))
    y = X.sum(axis=1) + rng.normal(scale=0.1, size=samples)
    df = pd.DataFrame(X, columns=[f"metric_{i}" for i in range(features)])
    df["score"] = y
    df.to_csv('data/performance_metrics.csv', index=False)
    print("Performance metrics generated and saved.")

def ensure_auxiliary_datasets():
    anomaly_path = 'data/anomaly_data.csv'
    perf_path = 'data/performance_metrics.csv'
    if not os.path.exists(anomaly_path) or os.path.getsize(anomaly_path) == 0:
        generate_anomaly_data()
    if not os.path.exists(perf_path) or os.path.getsize(perf_path) == 0:
        generate_performance_metrics()

if __name__ == "__main__":
    ensure_auxiliary_datasets()
