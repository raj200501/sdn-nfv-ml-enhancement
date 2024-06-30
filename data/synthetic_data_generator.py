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
