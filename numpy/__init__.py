import math
import random as _random


class NDArray:
    def __init__(self, data):
        self.data = data

    @property
    def shape(self):
        return _shape(self.data)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        flat = _flatten(self.data)
        reshaped = _reshape_list(flat, shape)
        return NDArray(reshaped)

    def astype(self, _dtype):
        return NDArray(_map(self.data, lambda x: int(x)))

    def flatten(self):
        return _flatten(self.data)

    def sum(self, axis=None):
        if axis is None:
            return sum(_flatten(self.data))
        if axis == 1:
            return NDArray([sum(row) for row in self.data])
        return sum(self.data)

    def mean(self, axis=None):
        if axis is None:
            flat = _flatten(self.data)
            return sum(flat) / len(flat) if flat else 0
        if axis == 1:
            return NDArray([sum(row) / len(row) if row else 0 for row in self.data])
        return self.sum(axis=axis) / len(self.data) if self.data else 0

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __gt__(self, other):
        return NDArray(_map(self.data, lambda x: 1 if x > other else 0))

    def __add__(self, other):
        if isinstance(other, NDArray):
            other = other.data
        if isinstance(other, list):
            return NDArray([a + b for a, b in zip(self.data, other)])
        return NDArray(_map(self.data, lambda x: x + other))

    def __sub__(self, other):
        if isinstance(other, NDArray):
            other = other.data
        if isinstance(other, list):
            return NDArray([a - b for a, b in zip(self.data, other)])
        return NDArray(_map(self.data, lambda x: x - other))

    def __repr__(self):
        return f"NDArray({self.data})"


def array(data):
    return NDArray(data)


def reshape(data, shape):
    if isinstance(data, NDArray):
        return data.reshape(shape)
    return NDArray(data).reshape(shape)


def argmax(data):
    if isinstance(data, NDArray):
        data = data.data
    return max(range(len(data)), key=lambda i: data[i])


def amax(data):
    if isinstance(data, NDArray):
        data = data.data
    if isinstance(data, list) and data and isinstance(data[0], list):
        return max(max(row) for row in data)
    return max(data)


class _RandomModule:
    def rand(self):
        return _random.random()

    def normal(self, loc=0.0, scale=1.0, size=None):
        return _random_array(lambda: _random.gauss(loc, scale), size)

    def uniform(self, low=0.0, high=1.0, size=None):
        return _random_array(lambda: _random.uniform(low, high), size)

    def default_rng(self, seed=None):
        return _Generator(seed)


class _Generator:
    def __init__(self, seed=None):
        self._random = _random.Random(seed)

    def normal(self, size=None, scale=1.0):
        return _random_array(lambda: self._random.gauss(0, scale), size)

    def uniform(self, low=0.0, high=1.0, size=None):
        return _random_array(lambda: self._random.uniform(low, high), size)

    def integers(self, low, high=None, size=None):
        if high is None:
            high = low
            low = 0
        return _random_array(lambda: self._random.randint(low, high - 1), size)


random = _RandomModule()
number = (int, float)


class _LinalgModule:
    def norm(self, vector):
        if isinstance(vector, NDArray):
            vector = vector.data
        return math.sqrt(sum(x * x for x in vector))


linalg = _LinalgModule()


def _shape(data):
    if isinstance(data, NDArray):
        data = data.data
    if isinstance(data, list):
        if not data:
            return (0,)
        if isinstance(data[0], list):
            return (len(data),) + _shape(data[0])
        return (len(data),)
    return ()


def _flatten(data):
    if isinstance(data, NDArray):
        data = data.data
    if isinstance(data, list):
        flat = []
        for item in data:
            if isinstance(item, list):
                flat.extend(_flatten(item))
            else:
                flat.append(item)
        return flat
    return [data]


def _reshape_list(flat, shape):
    if not shape:
        return flat[0]
    if len(shape) == 1:
        return flat[: shape[0]]
    step = int(len(flat) / shape[0]) if shape[0] else 0
    return [
        _reshape_list(flat[i * step:(i + 1) * step], shape[1:])
        for i in range(shape[0])
    ]


def _map(data, func):
    if isinstance(data, NDArray):
        data = data.data
    if isinstance(data, list):
        return [_map(item, func) for item in data]
    return func(data)


def _random_array(generator, size):
    if size is None:
        return generator()
    if isinstance(size, int):
        size = (size,)
    flat = [generator() for _ in range(int(math.prod(size)))]
    return NDArray(_reshape_list(flat, size))
