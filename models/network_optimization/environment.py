import numpy as np


class NetworkEnvironment:
    """Simple synthetic environment for DQN training demos."""

    def __init__(self, state_size=4, action_size=3, max_steps=25, seed=42):
        self.state_size = state_size
        self.action_size = action_size
        self.max_steps = max_steps
        self.random = np.random.default_rng(seed)
        self.steps_taken = 0
        self.state = None
        self.target = self.random.normal(size=self.state_size)

    def reset(self):
        self.steps_taken = 0
        self.state = self.random.normal(size=self.state_size)
        return self.state

    def step(self, action):
        self.steps_taken += 1
        action_effect = (action - (self.action_size // 2)) * 0.1
        noise = self.random.normal(scale=0.05, size=self.state_size)
        self.state = self.state + noise + action_effect
        reward = -float(np.linalg.norm(self.state - self.target))
        done = self.steps_taken >= self.max_steps
        info = {"steps": self.steps_taken}
        return self.state, reward, done, info
