#!/usr/bin/env bash
set -euo pipefail

export SDN_NFV_LSTM_EPOCHS=${SDN_NFV_LSTM_EPOCHS:-1}
export SDN_NFV_CNN_EPOCHS=${SDN_NFV_CNN_EPOCHS:-1}
export SDN_NFV_DQN_EPISODES=${SDN_NFV_DQN_EPISODES:-5}
export SDN_NFV_DQN_STEPS=${SDN_NFV_DQN_STEPS:-25}

python data/synthetic_data_generator.py
python preprocessing/data_cleaning.py
python preprocessing/normalization.py
python preprocessing/feature_engineering.py
python models/model_training.py
python evaluation/model_evaluation.py
python evaluation/results_visualization.py
python deployment/sdn_controller_integration.py
python deployment/nfv_orchestrator_integration.py
python deployment/real_time_decision_making.py
