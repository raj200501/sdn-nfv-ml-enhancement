class Dense:
    def __init__(self, units, input_dim=None, activation=None):
        self.units = units
        self.input_dim = input_dim
        self.activation = activation


class LSTM:
    def __init__(self, units, return_sequences=False, input_shape=None):
        self.units = units
        self.return_sequences = return_sequences
        self.input_shape = input_shape


class Conv1D:
    def __init__(self, filters, kernel_size, activation=None, input_shape=None):
        self.units = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.input_shape = input_shape


class MaxPooling1D:
    def __init__(self, pool_size=2):
        self.pool_size = pool_size


class Flatten:
    def __init__(self):
        self.units = None


class BatchNormalization:
    def __init__(self, momentum=0.99):
        self.momentum = momentum


class LeakyReLU:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
