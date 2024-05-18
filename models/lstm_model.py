import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm_model(data_path):
    # Load and preprocess data
    df = pd.read_csv(data_path)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df['traffic'].values.reshape(-1, 1))
    
    # Create sequences
    X, y = [], []
    sequence_length = 50
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    X, y = np.array(X), np.array(y)
    
    # Split into training and testing sets
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Create and train the LSTM model
    model = create_lstm_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
    
    return model, scaler

if __name__ == "__main__":
    model, scaler = train_lstm_model('data/traffic_data.csv')
    model.save('models/lstm_model.h5')
    with open('models/lstm_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
