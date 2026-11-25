import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from module2_tabular.preprocess import load_raw
from module5_federated.fedavg import (HospitalClient, fed_avg, local_train,
                                      log_loss, predict_proba)
from module5_federated.split_data import build_federation


def run_federated_simulation(num_rounds=15, local_epochs=1, lr=0.05, alpha=1.0,
                             csv_path='data/wisconsin.csv', output_dir='models',
                             seed=42, verbose=True):
    """
    FedAvg across 3 hospitals with NON-IID data, starting from zero weights so
    the learning curve is real. The global model is scored on a shared held-out
    test set. Two baselines are reported for context:
      * centralized : all training data pooled (upper reference, no privacy)
      * local_only  : each hospital trains alone (what FL is meant to beat)
    """
    X_df, y, _ = load_raw(csv_path)
    hospitals, (X_te, y_te), (mean, std) = build_federation(X_df.values, y, alpha, seed)
    clients = [HospitalClient(h['name'], h['X_train'], h['y_train'],
                              h['X_val'], h['y_val'], seed=i)
               for i, h in enumerate(hospitals)]
    d = X_te.shape[1]
    w, b = np.zeros(d), 0.0

    hist = {'rounds': [], 'global_accuracy': [], 'global_auc': [], 'global_loss': [],
            'hospital_A_acc': [], 'hospital_B_acc': [], 'hospital_C_acc': []}

    if verbose:
        print(f"FedAvg | hospitals=3 | rounds={num_rounds} | local_epochs={local_epochs} | alpha={alpha}")
        for c in clients:
            print(f"  Hospital {c.name}: train={c.n_samples} "
                  f"malignant={y_share(c.y_train):.0%}")

    for r in range(1, num_rounds + 1):
        w, b = fed_avg([c.fit(w, b, r, local_epochs, lr) for c in clients])
        p = predict_proba(w, b, X_te)
        hist['rounds'].append(r)
        hist['global_accuracy'].append(round(float(((p >= .5) == y_te).mean()), 4))
        hist['global_auc'].append(round(float(roc_auc_score(y_te, p)), 4))
        hist['global_loss'].append(round(log_loss(y_te, p), 4))
        for c in clients:
            hist[f'hospital_{c.name}_acc'].append(round(c.evaluate(w, b)[1], 4))
        if verbose:
            print(f"  Round {r:>2}: acc={hist['global_accuracy'][-1]*100:.2f}% "
                  f"auc={hist['global_auc'][-1]:.4f} loss={hist['global_loss'][-1]:.4f}")

    # ---- baselines ------------------------------------------------------
    steps = num_rounds * local_epochs
    Xc = np.vstack([c.X_train for c in clients]); yc = np.concatenate([c.y_train for c in clients])
    wc, bc = local_train(np.zeros(d), 0.0, Xc, yc, epochs=steps, lr=lr, seed=seed)
    local_only = {}
    for c in clients:
        wl, bl = local_train(np.zeros(d), 0.0, c.X_train, c.y_train, epochs=steps, lr=lr, seed=seed)
        local_only[f'hospital_{c.name}'] = round(float(((predict_proba(wl, bl, X_te) >= .5) == y_te).mean()), 4)
    baselines = {
        'centralized_accuracy': round(float(((predict_proba(wc, bc, X_te) >= .5) == y_te).mean()), 4),
        'local_only_accuracy': local_only,
        'federated_accuracy': hist['global_accuracy'][-1],
    }

    hist['baselines'] = baselines
    hist['config'] = {'rounds': num_rounds, 'local_epochs': local_epochs, 'lr': lr,
                      'dirichlet_alpha': alpha, 'algorithm': 'FedAvg (NumPy logistic regression)',
                      'hospital_sizes': {c.name: c.n_samples for c in clients},
                      'hospital_malignant_share': {c.name: round(y_share(c.y_train), 3) for c in clients}}

    os.makedirs(output_dir, exist_ok=True)
    with open(f'{output_dir}/fl_metrics.json', 'w') as f:
        json.dump(hist, f, indent=2)
    # version-independent model file (plain arrays, no pickle)
    np.savez(f'{output_dir}/federated_model.npz', w=w, b=b, mean=mean, std=std)

    if verbose:
        print("\nBaselines on the shared test set:")
        print(f"  Centralized (no privacy) : {baselines['centralized_accuracy']*100:.2f}%")
        print(f"  Federated (FedAvg)       : {baselines['federated_accuracy']*100:.2f}%")
        for k, v in local_only.items():
            print(f"  Local only {k:<11}: {v*100:.2f}%")
    return hist


def y_share(y):
    return float(np.mean(y))
