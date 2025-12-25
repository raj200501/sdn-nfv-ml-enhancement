import json
from numpy import NDArray


class Sequential:
    def __init__(self):
        self.layers = []
        self.compiled = False

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss=None, optimizer=None, metrics=None):
        self.compiled = True

    def fit(self, X, y, epochs=1, batch_size=32, validation_split=0.0, verbose=0):
        return {"loss": 0.0}

    def _output_units(self):
        for layer in reversed(self.layers):
            if hasattr(layer, "units") and layer.units is not None:
                return layer.units
        return 1

    def predict(self, X):
        if isinstance(X, NDArray):
            samples = len(X.data)
        elif isinstance(X, list):
            samples = len(X)
        else:
            samples = 1
        units = self._output_units()
        output = [[0.0 for _ in range(units)] for _ in range(samples)]
        return NDArray(output)

    def save(self, filepath):
        payload = {"units": self._output_units()}
        with open(filepath, "w") as handle:
            json.dump(payload, handle)

    def save_weights(self, filepath):
        self.save(filepath)

    def load_weights(self, filepath):
        return None


def load_model(filepath):
    model = Sequential()
    try:
        with open(filepath) as handle:
            payload = json.load(handle)
        model.add(_DummyLayer(payload.get("units", 1)))
    except Exception:
        model.add(_DummyLayer(1))
    return model


class _DummyLayer:
    def __init__(self, units):
        self.units = units
