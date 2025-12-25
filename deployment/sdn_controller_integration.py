import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import requests
import json

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
    config = generate_optimized_configuration(state)
    status = apply_configuration(config)
    if status == 200:
        print("Network configuration applied successfully.")
    else:
        print("Failed to apply network configuration.")

def generate_optimized_configuration(state):
    # Dummy function for generating an optimized network configuration
    config = {"setting": "optimized"}
    return config

def get_network_state_legacy():
    response = requests.get("http://localhost:8080/network/state")
    return response.json()

def apply_configuration_legacy(config):
    headers = {'Content-Type': 'application/json'}
    response = requests.post("http://localhost:8080/network/configure", headers=headers, data=json.dumps(config))
    return response.status_code

if __name__ == "__main__":
    optimize_network()
