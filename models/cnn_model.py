import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.model_selection import train_test_split

def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_cnn_model(data_path):
    # Load and preprocess data
    df = pd.read_csv(data_path)
    X = df.drop('anomaly', axis=1).values
    y = df['anomaly'].values
    
    # Reshape data for CNN
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the CNN model
    model = create_cnn_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
    
    return model

if __name__ == "__main__":
    model = train_cnn_model('data/anomaly_data.csv')
    model.save('models/cnn_model.h5')
