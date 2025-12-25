#!/usr/bin/env bash
set -euo pipefail

echo "Running SDN/NFV ML smoke test..."
export SDN_NFV_LSTM_EPOCHS=1
export SDN_NFV_CNN_EPOCHS=1
export SDN_NFV_DQN_EPISODES=2
export SDN_NFV_DQN_STEPS=5

python data/synthetic_data_generator.py
python preprocessing/data_cleaning.py
python preprocessing/normalization.py
python preprocessing/feature_engineering.py
python models/model_training.py
python evaluation/model_evaluation.py

echo "Smoke test completed."
