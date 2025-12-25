import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_predictions(y_true, y_pred, title):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_roc_curve(y_true, y_pred, title):
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, title):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

if __name__ == "__main__":
    if not os.path.exists('data/y_true.csv') or not os.path.exists('data/y_pred.csv'):
        rng = np.random.default_rng(99)
        y_true = rng.integers(0, 2, size=100)
        y_pred = rng.integers(0, 2, size=100)
        pd.DataFrame(y_true).to_csv('data/y_true.csv', index=False)
        pd.DataFrame(y_pred).to_csv('data/y_pred.csv', index=False)
    y_true = pd.read_csv('data/y_true.csv').values.flatten()
    y_pred = pd.read_csv('data/y_pred.csv').values.flatten()

    plot_predictions(y_true, y_pred, "Predicted vs Actual")
    plot_roc_curve(y_true, y_pred, "ROC Curve")
    plot_confusion_matrix(y_true, y_pred, "Confusion Matrix")

    plt.savefig('data/visualization_outputs.png')
    print("Visualization saved to data/visualization_outputs.png")
