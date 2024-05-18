import tensorflow as tf
from tensorflow.keras.models import load_model

def load_lstm_model(model_path, scaler_path):
    model = load_model(model_path)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def load_cnn_model(model_path):
    model = load_model(model_path)
    return model

def load_dqn_model(model_path):
    model = load_model(model_path)
    return model
