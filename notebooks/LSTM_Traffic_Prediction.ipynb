import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pickle

# Load the model and scaler
model = load_model('../models/lstm_model.h5')
with open('../models/lstm_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load and preprocess the data
df = pd.read_csv('../data/traffic_data.csv')
data = scaler.transform(df['traffic'].values.reshape(-1, 1))

# Create sequences
sequence_length = 50
X, y = [], []
for i in range(len(data) - sequence_length):
    X.append(data[i:i+sequence_length])
    y.append(data[i+sequence_length])
X, y = np.array(X), np.array(y)

# Predict traffic
predicted_traffic = model.predict(X)

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(df['timestamp'][sequence_length:], scaler.inverse_transform(predicted_traffic), label='Predicted Traffic')
plt.plot(df['timestamp'][sequence_length:], scaler.inverse_transform(y), label='Actual Traffic')
plt.legend()
plt.xlabel('Timestamp')
plt.ylabel('Traffic')
plt.title('Traffic Prediction using LSTM')
plt.savefig('../results/traffic_prediction_results.png')
plt.show()
