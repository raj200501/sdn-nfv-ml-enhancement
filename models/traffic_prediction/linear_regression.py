import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def train_linear_regression(data):
    X = data.iloc[:, :-1]  # All columns except the last one
    y = data.iloc[:, -1]   # The last column

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print_evaluation_metrics(y_test, y_pred)

    return model

def print_evaluation_metrics(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")

if __name__ == "__main__":
    data = load_data('data/network_data.csv')
    model = train_linear_regression(data)
    print("Linear regression model trained and evaluated.")
