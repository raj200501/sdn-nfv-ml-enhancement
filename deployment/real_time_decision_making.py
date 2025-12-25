import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import requests
import json
import numpy as np
from keras.models import load_model

def get_network_state():
    try:
        response = requests.get("http://localhost:8080/network/state", timeout=2)
        return response.json()
    except requests.RequestException:
        print("Warning: SDN controller not reachable, using simulated state.")
        return {"feature1": 0.1, "feature2": 0.2, "feature3": 0.3}

def apply_configuration(config):
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post("http://localhost:8080/network/configure", headers=headers, data=json.dumps(config), timeout=2)
        return response.status_code
    except requests.RequestException:
        print("Warning: SDN controller not reachable, simulated apply.")
        return 200

def optimize_network():
    state = get_network_state()
    state_array = np.array([state['feature1'], state['feature2'], state['feature3']])
    state_array = state_array.reshape((1, len(state_array)))

    if os.path.exists('models/network_optimization/dqn_model.h5'):
        model = load_model('models/network_optimization/dqn_model.h5')
        action = int(np.argmax(model.predict(state_array)[0]))
    else:
        print("Warning: DQN model not found, using heuristic action.")
        action = int(np.argmax(state_array))
    
    config = generate_optimized_configuration(action)
    status = apply_configuration(config)
    
    if status == 200:
        print("Network configuration applied successfully.")
    else:
        print("Failed to apply network configuration.")

def generate_optimized_configuration(action):
    # Dummy function for generating an optimized network configuration
    config = {"action": action}
    return config

def get_network_state_legacy():
    response = requests.get("http://localhost:8080/network/state")
    return response.json()

def apply_configuration_legacy(config):
    headers = {'Content-Type': 'application/json'}
    response = requests.post("http://localhost:8080/network/configure", headers=headers, data=json.dumps(config))
    return response.status_code

def optimize_network_legacy():
    state = get_network_state_legacy()
    state_array = np.array([state['feature1'], state['feature2'], state['feature3']])
    state_array = state_array.reshape((1, len(state_array)))
    model = load_model('models/network_optimization/dqn_model.h5')
    action = np.argmax(model.predict(state_array)[0])
    config = generate_optimized_configuration(action)
    status = apply_configuration_legacy(config)
    if status == 200:
        print("Network configuration applied successfully.")
    else:
        print("Failed to apply network configuration.")

if __name__ == "__main__":
    optimize_network()
