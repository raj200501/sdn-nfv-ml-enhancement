from numpy import NDArray, array


class MinMaxScaler:
    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit_transform(self, X):
        if isinstance(X, NDArray):
            data = X.data
        elif hasattr(X, "data"):
            data = X.data
        else:
            data = X
        if not data:
            return array([])
        cols = list(zip(*data))
        self.min_ = [min(col) for col in cols]
        self.max_ = [max(col) for col in cols]
        scaled = []
        for row in data:
            scaled_row = []
            for value, min_v, max_v in zip(row, self.min_, self.max_):
                denom = (max_v - min_v) or 1
                scaled_row.append((value - min_v) / denom)
            scaled.append(scaled_row)
        return array(scaled)
