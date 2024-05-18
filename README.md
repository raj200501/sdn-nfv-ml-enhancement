Leveraging Machine Learning for Enhanced Decision-Making in Software-Defined Networking and Network Function Virtualization
This repository contains the implementation of the research paper "Leveraging Machine Learning for Enhanced Decision-Making in Software-Defined Networking and Network Function Virtualization." The repository includes scripts for synthetic data generation, machine learning model training, evaluation, and necessary configurations.


Overview
The goal of this project is to leverage machine learning techniques to enhance decision-making processes in software-defined networking (SDN) and network function virtualization (NFV). The project is divided into three main components:

Traffic Prediction: Using Long Short-Term Memory (LSTM) networks to predict network traffic patterns.
Anomaly Detection: Implementing Convolutional Neural Networks (CNN) to identify anomalies in network traffic.
Network Optimization: Applying Deep Q-Network (DQN) for optimizing resource utilization and network performance.
Setup
Prerequisites
Ensure you have Python 3.7+ and pip installed. It's recommended to use a virtual environment to manage dependencies.

Installation
Clone the repository:

git clone https://github.com/yourusername/sdn-nfv-ml-enhancement.git
cd sdn-nfv-ml-enhancement
Create and activate a virtual environment:

python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install the required packages:

pip install -r requirements.txt
Generate synthetic data:

python data/generate_synthetic_data.py
Train the LSTM model:

python models/lstm_model.py
Train the CNN model:

python models/cnn_model.py
Train the DQN model:

python models/dqn_model.py
Usage
Notebooks
The repository includes Jupyter notebooks for training and evaluating the models. You can run these notebooks to see step-by-step implementations and visualizations:

notebooks/LSTM_Traffic_Prediction.ipynb: Predict network traffic using LSTM.
notebooks/CNN_Anomaly_Detection.ipynb: Detect anomalies in network traffic using CNN.
notebooks/DQN_Network_Optimization.ipynb: Optimize network resource allocation using DQN.
Scripts
Alternatively, you can directly run the scripts to generate results:

Traffic Prediction:

python models/lstm_model.py
Anomaly Detection:

python models/cnn_model.py
Network Optimization:

python models/dqn_model.py
Results
The results of the experiments are saved in the results/ directory. Key figures include:

Traffic Prediction Results: traffic_prediction_results.png
Anomaly Detection Results: anomaly_detection_results.png
Network Optimization Results: resource_utilization_results.png
