import csv
import os
from numpy import NDArray, array


class Series:
    def __init__(self, data):
        self.data = data

    @property
    def values(self):
        return self.data


class _ILoc:
    def __init__(self, df):
        self.df = df

    def __getitem__(self, key):
        rows, cols = key
        data = self.df.data
        if isinstance(rows, slice):
            row_indices = range(*rows.indices(len(data)))
        elif isinstance(rows, int):
            row_indices = [rows]
        else:
            row_indices = rows
        if isinstance(cols, slice):
            col_indices = range(*cols.indices(len(self.df.columns)))
        elif isinstance(cols, int):
            col_indices = [cols]
        else:
            col_indices = cols
        selected = [[data[r][c] for c in col_indices] for r in row_indices]
        if len(col_indices) == 1:
            return Series([row[0] for row in selected])
        return DataFrame(selected, columns=[self.df.columns[c] for c in col_indices])


class DataFrame:
    def __init__(self, data, columns=None):
        if isinstance(data, NDArray):
            data = data.data
        if isinstance(data, dict):
            columns = list(data.keys())
            data = list(zip(*data.values()))
        if data and data and not isinstance(data[0], (list, tuple)):
            data = [[item] for item in data]
        self.data = [list(row) for row in data] if data else []
        if columns is None:
            columns = [f"col_{i}" for i in range(len(self.data[0]))] if self.data else []
        self.columns = list(columns)
        self.iloc = _ILoc(self)

    def dropna(self, inplace=False):
        return None

    def drop_duplicates(self, inplace=False):
        return None

    def applymap(self, func):
        return self

    def __getitem__(self, key):
        if isinstance(key, DataFrame):
            return self
        raise KeyError(key)

    def __setitem__(self, key, value):
        if hasattr(value, "data"):
            value = value.data
        if isinstance(value, list):
            for idx, row in enumerate(self.data):
                row.append(value[idx] if idx < len(value) else None)
        else:
            for row in self.data:
                row.append(value)
        self.columns.append(key)

    def to_csv(self, filepath, index=False):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if self.columns:
                writer.writerow(self.columns)
            for row in self.data:
                writer.writerow(row)

    @property
    def values(self):
        return array(self.data)

    @property
    def empty(self):
        return len(self.data) == 0

    @property
    def shape(self):
        if not self.data:
            return (0, 0)
        return (len(self.data), len(self.data[0]))

    def select_dtypes(self, include=None):
        return self

    def copy(self):
        return DataFrame([row[:] for row in self.data], columns=self.columns[:])

    def sum(self, axis=0):
        if axis == 1:
            return [sum(row) for row in self.data]
        return [sum(col) for col in zip(*self.data)] if self.data else []

    def mean(self, axis=0):
        if axis == 1:
            return [sum(row) / len(row) if row else 0 for row in self.data]
        if not self.data:
            return []
        return [sum(col) / len(col) for col in zip(*self.data)]


def read_csv(filepath):
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return DataFrame([])
    with open(filepath, newline="") as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)
    if not rows:
        return DataFrame([])
    columns = rows[0]
    data_rows = []
    for row in rows[1:]:
        data_rows.append([_to_number(cell) for cell in row])
    return DataFrame(data_rows, columns=columns)


def _to_number(value):
    try:
        if "." in value:
            return float(value)
        return int(value)
    except Exception:
        return value


DataFrame = DataFrame
