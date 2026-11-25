"""Create the non-IID hospital partitions used by the federated simulation."""
import numpy as np
from sklearn.model_selection import train_test_split


def dirichlet_partition(y, n_clients=3, alpha=1.0, seed=42, min_size=40):
    """
    Label-skewed split: for each class, indices are divided between hospitals
    with proportions ~ Dirichlet(alpha). Smaller alpha = more heterogeneous.
    Retries seeds until every hospital has >= min_size samples of BOTH classes.
    """
    for attempt in range(200):
        rng = np.random.default_rng(seed + attempt)
        parts = [[] for _ in range(n_clients)]
        for c in np.unique(y):
            idx = rng.permutation(np.where(y == c)[0])
            props = rng.dirichlet([alpha] * n_clients)
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for k, chunk in enumerate(np.split(idx, cuts)):
                parts[k].extend(chunk.tolist())
        ok = all(len(p) >= min_size and 0 < y[p].mean() < 1 and
                 min(y[p].sum(), len(p) - y[p].sum()) >= 8 for p in parts)
        if ok:
            return [np.array(sorted(p)) for p in parts]
    raise RuntimeError("Could not build a valid partition - lower min_size or raise alpha")


def build_federation(X, y, alpha=1.0, seed=42):
    """
    X must already be a numeric array. Returns:
      hospitals : list of dicts with train/val data per hospital
      (X_test, y_test): global held-out test set (never used for training)
      scaler stats : (mean, std) computed on pooled TRAIN data
    Note: pooled mean/std stands in for a secure-aggregation step.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                              random_state=seed, stratify=y)
    mean, std = X_tr.mean(0), X_tr.std(0) + 1e-8
    X_tr, X_te = (X_tr - mean) / std, (X_te - mean) / std

    hospitals = []
    for name, idx in zip('ABC', dirichlet_partition(y_tr, 3, alpha, seed)):
        Xh, yh = X_tr[idx], y_tr[idx]
        Xa, Xv, ya, yv = train_test_split(Xh, yh, test_size=0.2,
                                          random_state=seed, stratify=yh)
        hospitals.append({'name': name, 'X_train': Xa, 'y_train': ya,
                          'X_val': Xv, 'y_val': yv})
    return hospitals, (X_te, y_te), (mean, std)
