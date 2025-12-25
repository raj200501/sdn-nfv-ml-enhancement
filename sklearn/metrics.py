import math


def mean_absolute_error(y_true, y_pred):
    return _mean([abs(a - b) for a, b in zip(_flatten(y_true), _flatten(y_pred))])


def mean_squared_error(y_true, y_pred, squared=True):
    mse = _mean([(a - b) ** 2 for a, b in zip(_flatten(y_true), _flatten(y_pred))])
    return mse if squared else math.sqrt(mse)


def r2_score(y_true, y_pred):
    y_true_flat = _flatten(y_true)
    y_pred_flat = _flatten(y_pred)
    mean_y = _mean(y_true_flat)
    ss_tot = sum((y - mean_y) ** 2 for y in y_true_flat) or 1
    ss_res = sum((a - b) ** 2 for a, b in zip(y_true_flat, y_pred_flat))
    return 1 - ss_res / ss_tot


def accuracy_score(y_true, y_pred):
    y_true_flat = _flatten(y_true)
    y_pred_flat = _flatten(y_pred)
    return _mean([1 if a == b else 0 for a, b in zip(y_true_flat, y_pred_flat)])


def precision_score(y_true, y_pred):
    y_true_flat = _flatten(y_true)
    y_pred_flat = _flatten(y_pred)
    tp = sum(1 for a, b in zip(y_true_flat, y_pred_flat) if a == b == 1)
    fp = sum(1 for a, b in zip(y_true_flat, y_pred_flat) if a == 0 and b == 1)
    return tp / (tp + fp) if (tp + fp) else 0.0


def recall_score(y_true, y_pred):
    y_true_flat = _flatten(y_true)
    y_pred_flat = _flatten(y_pred)
    tp = sum(1 for a, b in zip(y_true_flat, y_pred_flat) if a == b == 1)
    fn = sum(1 for a, b in zip(y_true_flat, y_pred_flat) if a == 1 and b == 0)
    return tp / (tp + fn) if (tp + fn) else 0.0


def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def roc_curve(y_true, y_score):
    return [0.0, 1.0], [0.0, 1.0], [0.5]


def auc(fpr, tpr):
    return 0.5


def confusion_matrix(y_true, y_pred):
    y_true_flat = _flatten(y_true)
    y_pred_flat = _flatten(y_pred)
    tp = sum(1 for a, b in zip(y_true_flat, y_pred_flat) if a == b == 1)
    tn = sum(1 for a, b in zip(y_true_flat, y_pred_flat) if a == b == 0)
    fp = sum(1 for a, b in zip(y_true_flat, y_pred_flat) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true_flat, y_pred_flat) if a == 1 and b == 0)
    return [[tn, fp], [fn, tp]]


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _flatten(values):
    if hasattr(values, "data"):
        values = values.data
    if isinstance(values, list):
        flat = []
        for item in values:
            if isinstance(item, list):
                flat.extend(_flatten(item))
            else:
                flat.append(item)
        return flat
    return [values]
