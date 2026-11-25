"""
Module 7 - rigorous evaluation of the clinical model.
Writes models/evaluation_report.json (read by the dashboard Performance page).
  * stratified 5-fold cross-validation (mean +/- std), scaler fit inside each fold
  * ROC curve, confusion matrix and Brier score on the held-out test set
"""
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.base import clone
from sklearn.metrics import brier_score_loss, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from module2_tabular.preprocess import load_raw
from module2_tabular.train_xgboost import binary_metrics, get_classifier


def evaluate_tabular(csv_path='data/wisconsin.csv', folds=5, out='models/evaluation_report.json'):
    X_df, y, names = load_raw(csv_path)
    X = X_df.values
    base, model_name = get_classifier(early_stopping=False)   # no eval_set in CV

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    rows = []
    for tr, te in skf.split(X, y):
        pipe = make_pipeline(StandardScaler(), clone(base)).fit(X[tr], y[tr])
        rows.append(binary_metrics(y[te], pipe.predict_proba(X[te])[:, 1]))

    def ms(key):
        v = np.array([r[key] for r in rows], dtype=float)
        return round(float(v.mean()), 2 if key != 'auc_roc' else 4), round(float(v.std()), 2 if key != 'auc_roc' else 4)

    cv = {'folds': folds}
    for key, short in (('accuracy', 'accuracy'), ('sensitivity', 'sensitivity'),
                       ('specificity', 'specificity'), ('auc_roc', 'auc')):
        cv[f'{short}_mean'], cv[f'{short}_std'] = ms(key)

    # held-out test set: ROC, confusion matrix, calibration
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe = make_pipeline(StandardScaler(), clone(base)).fit(X_tr, y_tr)
    prob = pipe.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, prob)
    test = binary_metrics(y_te, prob)

    report = {
        'tabular_model': model_name,
        'tabular_cv': cv,
        'tabular_test': {k: v for k, v in test.items() if k != 'confusion_matrix'},
        'confusion_matrix': test['confusion_matrix'],
        'brier_score': round(float(brier_score_loss(y_te, prob)), 4),
        'roc': {'fpr': [round(float(x), 4) for x in fpr], 'tpr': [round(float(x), 4) for x in tpr]},
    }
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"{folds}-fold CV ({model_name})")
    print(f"  Accuracy    : {cv['accuracy_mean']}% +/- {cv['accuracy_std']}")
    print(f"  Sensitivity : {cv['sensitivity_mean']}% +/- {cv['sensitivity_std']}")
    print(f"  Specificity : {cv['specificity_mean']}% +/- {cv['specificity_std']}")
    print(f"  AUC-ROC     : {cv['auc_mean']} +/- {cv['auc_std']}")
    print(f"Brier score (test): {report['brier_score']}")
    return report


if __name__ == '__main__':
    evaluate_tabular()
