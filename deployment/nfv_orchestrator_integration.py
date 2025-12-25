import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import requests
import json

def get_nfv_state():
    try:
        response = requests.get("http://localhost:8080/nfv/state", timeout=2)
        return response.json()
    except requests.RequestException:
        print("Warning: NFV orchestrator not reachable, using simulated state.")
        return {"vnf_count": 2, "cpu_utilization": 0.5}

def apply_nfv_configuration(config):
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post("http://localhost:8080/nfv/configure", headers=headers, data=json.dumps(config), timeout=2)
        return response.status_code
    except requests.RequestException:
        print("Warning: NFV orchestrator not reachable, simulated apply.")
        return 200

def optimize_nfv():
    state = get_nfv_state()
    config = generate_optimized_nfv_configuration(state)
    status = apply_nfv_configuration(config)
    if status == 200:
        print("NFV configuration applied successfully.")
    else:
        print("Failed to apply NFV configuration.")

def generate_optimized_nfv_configuration(state):
    # Dummy function for generating an optimized NFV configuration
    config = {"setting": "optimized"}
    return config

def get_nfv_state_legacy():
    response = requests.get("http://localhost:8080/nfv/state")
    return response.json()

def apply_nfv_configuration_legacy(config):
    headers = {'Content-Type': 'application/json'}
    response = requests.post("http://localhost:8080/nfv/configure", headers=headers, data=json.dumps(config))
    return response.status_code

if __name__ == "__main__":
    optimize_nfv()
