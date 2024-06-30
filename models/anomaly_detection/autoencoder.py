import pandas as pd
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def preprocess_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def create_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = Dense(16, activation='relu')(encoded)
    decoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)
    
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    return autoencoder

def train_autoencoder(model, X_train, epochs=50, batch_size=256):
    model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)
    model.save('models/anomaly_detection/autoencoder_model.h5')
    print("Autoencoder model trained and saved.")

def evaluate_model(model, X_test):
    reconstructions = model.predict(X_test)
    reconstruction_errors = np.mean(np.square(X_test - reconstructions), axis=1)
    threshold = np.mean(reconstruction_errors) + 2 * np.std(reconstruction_errors)
    
    y_pred = (reconstruction_errors > threshold).astype(int)
    return y_pred

if __name__ == "__main__":
    data = load_data('data/anomaly_data.csv')
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    X, scaler = preprocess_data(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    autoencoder = create_autoencoder(X_train.shape[1])
    train_autoencoder(autoencoder, X_train)

    y_pred = evaluate_model(autoencoder, X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
