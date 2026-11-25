"""
Pure-NumPy federated learning building blocks (no network, no extra deps):
  * logistic regression trained with local mini-batch SGD
  * FedAvg aggregation (weights proportional to local sample counts)
"""
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def predict_proba(w, b, X):
    return sigmoid(X @ w + b)


def log_loss(y, p, eps=1e-9):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def local_train(w, b, X, y, epochs=1, lr=0.05, batch_size=32, l2=1e-3, seed=0):
    """One client's local update starting from the global weights."""
    rng = np.random.default_rng(seed)
    w, b = w.copy(), float(b)
    n = len(X)
    for _ in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, batch_size):
            j = idx[s:s + batch_size]
            err = predict_proba(w, b, X[j]) - y[j]
            w -= lr * (X[j].T @ err / len(j) + l2 * w)
            b -= lr * float(err.mean())
    return w, b


def fed_avg(updates):
    """updates: list of (w, b, n_samples) -> weighted-average (w, b)."""
    total = sum(n for _, _, n in updates)
    w = sum(wi * (n / total) for wi, _, n in updates)
    b = sum(bi * (n / total) for _, bi, n in updates)
    return w, float(b)


class HospitalClient:
    """A hospital: holds private data, only ever shares model weights."""

    def __init__(self, name, X_train, y_train, X_val, y_val, seed=0):
        self.name, self.seed = name, seed
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val

    @property
    def n_samples(self):
        return len(self.X_train)

    def fit(self, w, b, rnd, epochs=1, lr=0.05):
        w2, b2 = local_train(w, b, self.X_train, self.y_train,
                             epochs=epochs, lr=lr, seed=self.seed * 1000 + rnd)
        return w2, b2, self.n_samples

    def evaluate(self, w, b):
        p = predict_proba(w, b, self.X_val)
        return log_loss(self.y_val, p), float(((p >= .5) == self.y_val).mean())
