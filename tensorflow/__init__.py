class _Adam:
    def __init__(self, learning_rate=0.001, lr=None):
        self.learning_rate = learning_rate if lr is None else lr


class _Optimizers:
    Adam = _Adam


class _Keras:
    optimizers = _Optimizers()


keras = _Keras()
