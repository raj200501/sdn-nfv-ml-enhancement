import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from evaluation_metrics import evaluate_regression_model, evaluate_classification_model

def load_data(filepath):
    return pd.read_csv(filepath)

def evaluate_traffic_prediction_model():
    if not os.path.exists('models/traffic_prediction/lstm_model.h5'):
        print("LSTM model not found. Skipping traffic prediction evaluation.")
        return
    model = load_model('models/traffic_prediction/lstm_model.h5')
    data = load_data('data/network_data.csv')
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    X = X.values.reshape((X.shape[0], X.shape[1], 1))
    
    y_pred = model.predict(X)
    evaluate_regression_model(y, y_pred)

def evaluate_anomaly_detection_model():
    if not os.path.exists('models/anomaly_detection/cnn_model.h5'):
        print("CNN model not found. Skipping anomaly detection evaluation.")
        return
    model = load_model('models/anomaly_detection/cnn_model.h5')
    data = load_data('data/anomaly_data.csv')
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    X = X.values.reshape((X.shape[0], X.shape[1], 1))
    
    y_pred = (model.predict(X) > 0.5).astype("int32")
    evaluate_classification_model(y, y_pred)

if __name__ == "__main__":
    evaluate_traffic_prediction_model()
    evaluate_anomaly_detection_model()
