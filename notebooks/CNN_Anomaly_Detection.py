import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Load the model
model = load_model('../models/cnn_model.h5')

# Load and preprocess the data
df = pd.read_csv('../data/anomaly_data.csv')
X = df.drop('anomaly', axis=1).values
y = df['anomaly'].values

# Reshape data for CNN
X = X.reshape((X.shape[0], X.shape[1], 1))

# Predict anomalies
predicted_anomalies = model.predict(X).round()

# Evaluate performance
cm = confusion_matrix(y, predicted_anomalies)
cr = classification_report(y, predicted_anomalies)

print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(cr)

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(df['timestamp'], y, label='Actual Anomalies')
plt.plot(df['timestamp'], predicted_anomalies, label='Predicted Anomalies')
plt.legend()
plt.xlabel('Timestamp')
plt.ylabel('Anomaly')
plt.title('Anomaly Detection using CNN')
plt.savefig('../results/anomaly_detection_results.png')
plt.show()
