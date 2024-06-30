import requests
import json
import numpy as np
from keras.models import load_model

def get_network_state():
    response = requests.get("http://localhost:8080/network/state")
    return response.json()

def apply_configuration(config):
    headers = {'Content-Type': 'application/json'}
    response = requests.post("http://localhost:8080/network/configure", headers=headers, data=json.dumps(config))
    return response.status_code

def optimize_network():
    state = get_network_state()
    state_array = np.array([state['feature1'], state['feature2'], state['feature3']])
    state_array = state_array.reshape((1, len(state_array)))
    
    model = load_model('models/network_optimization/dqn_model.h5')
    action = np.argmax(model.predict(state_array)[0])
    
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

if __name__ == "__main__":
    optimize_network()
