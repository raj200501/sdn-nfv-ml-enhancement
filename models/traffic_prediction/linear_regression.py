import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def train_linear_regression_model(df):
    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    
    return model

if __name__ == "__main__":
    df = load_data('data/network_data.csv')
    df['target'] = df['mean']  # Example: Assume 'mean' as target for regression
    model = train_linear_regression_model(df)
    print("Linear Regression model trained and evaluated.")
