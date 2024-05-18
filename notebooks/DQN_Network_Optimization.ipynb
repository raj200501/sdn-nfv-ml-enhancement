import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the model
model = load_model('../models/dqn_model.h5')

# Initialize environment
env = gym.make('CartPole-v1')

# Function to test the agent
def test_agent(env, model, episodes=100):
    total_rewards = []
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, env.observation_space.shape[0]])
        total_reward = 0
        for time in range(500):
            action = np.argmax(model.predict(state)[0])
            next_state, reward, done, _ = env.step(action)
            state = np.reshape(next_state, [1, env.observation_space.shape[0]])
            total_reward += reward
            if done:
                break
        total_rewards.append(total_reward)
    return total_rewards

# Test the agent
rewards = test_agent(env, model)

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(range(len(rewards)), rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('DQN Network Optimization Performance')
plt.savefig('../results/resource_utilization_results.png')
plt.show()
