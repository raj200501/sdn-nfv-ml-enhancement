from traffic_prediction.lstm import train_lstm_model
from anomaly_detection.cnn import train_cnn_model
from network_optimization.deep_q_network import DQNAgent
import pandas as pd
import numpy as np

def load_data(filepath):
    return pd.read_csv(filepath)

def train_models():
    # Train LSTM for traffic prediction
    traffic_data = load_data('data/network_data.csv')
    train_lstm_model(traffic_data)

    # Train CNN for anomaly detection
    anomaly_data = load_data('data/anomaly_data.csv')
    train_cnn_model(anomaly_data)

    # Train DQN for network optimization
    env = NetworkEnvironment()  # Assume NetworkEnvironment is defined elsewhere
    state_size = env.state_size
    action_size = env.action_size
    agent = DQNAgent(state_size, action_size)
    
    for e in range(1000):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"episode: {e}/{1000}, score: {time}, e: {agent.epsilon:.2}")
                break
            if len(agent.memory) > 32:
                agent.replay(32)
        if e % 10 == 0:
            agent.save(f"models/network_optimization/dqn_{e}.h5")

if __name__ == "__main__":
    train_models()
