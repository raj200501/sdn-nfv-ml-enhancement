import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

def load_data(filepath):
    df = pd.read_csv(filepath)
    X = df.iloc[:, :-1].values  # All columns except the last one
    y = df.iloc[:, -1].values   # The last column
    return X, y

def preprocess_data(X, y):
    # Reshape data for CNN input (samples, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y

def build_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, epochs=10, batch_size=64):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    model.save('models/anomaly_detection/cnn_model.h5')
    print("CNN model trained and saved.")

if __name__ == "__main__":
    X, y = load_data('data/anomaly_data.csv')
    X, y = preprocess_data(X, y)

    cnn_model = build_cnn_model((X.shape[1], X.shape[2]))
    train_model(cnn_model, X, y)
