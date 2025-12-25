import os


def figure(figsize=None):
    return None


def plot(*args, **kwargs):
    return None


def title(_title):
    return None


def legend(*args, **kwargs):
    return None


def xlabel(_label):
    return None


def ylabel(_label):
    return None


def xlim(_bounds):
    return None


def ylim(_bounds):
    return None


def show():
    return None


def savefig(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as handle:
        handle.write("stub plot")
