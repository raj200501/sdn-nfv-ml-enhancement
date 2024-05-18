import numpy as np
import pandas as pd

def generate_traffic_data():
    # Generate synthetic traffic data
    time_steps = 1000
    traffic = np.sin(np.linspace(0, 50, time_steps)) + np.random.normal(0, 0.5, time_steps)
    df = pd.DataFrame({'timestamp': np.arange(time_steps), 'traffic': traffic})
    df.to_csv('data/traffic_data.csv', index=False)

def generate_anomaly_data():
    # Generate synthetic anomaly data
    time_steps = 1000
    anomalies = np.zeros(time_steps)
    anomalies[np.random.randint(0, time_steps, 20)] = 1  # Injecting anomalies
    df = pd.DataFrame({'timestamp': np.arange(time_steps), 'anomaly': anomalies})
    df.to_csv('data/anomaly_data.csv', index=False)

def generate_performance_metrics():
    # Generate synthetic performance metrics
    time_steps = 1000
    latency = np.random.normal(50, 5, time_steps)
    throughput = np.random.normal(1000, 100, time_steps)
    df = pd.DataFrame({'timestamp': np.arange(time_steps), 'latency': latency, 'throughput': throughput})
    df.to_csv('data/performance_metrics.csv', index=False)

if __name__ == "__main__":
    generate_traffic_data()
    generate_anomaly_data()
    generate_performance_metrics()
