from numpy import NDArray


def train_test_split(X, y, test_size=0.2, random_state=None):
    if isinstance(X, NDArray):
        X_data = X.data
    else:
        X_data = X
    y_data = y.data if hasattr(y, "data") else y
    total = len(X_data)
    split = int(total * (1 - test_size))
    X_train = X_data[:split]
    X_test = X_data[split:]
    y_train = y_data[:split]
    y_test = y_data[split:]
    return NDArray(X_train), NDArray(X_test), y_train, y_test
