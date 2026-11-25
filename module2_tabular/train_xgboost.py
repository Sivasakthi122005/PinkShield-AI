import json
import os
import warnings

import joblib
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split


def get_classifier(early_stopping=True):
    """XGBoost if installed, otherwise a scikit-learn gradient-boosting fallback."""
    try:
        from xgboost import XGBClassifier
        kwargs = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8,
                      eval_metric='logloss', random_state=42, verbosity=0)
        if early_stopping:
            kwargs['early_stopping_rounds'] = 20
        return XGBClassifier(**kwargs), 'XGBoost'
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        warnings.warn("xgboost not installed - using HistGradientBoostingClassifier fallback.")
        return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05,
                                              max_iter=300, random_state=42), 'HistGradientBoosting (fallback)'


def binary_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        'accuracy': round(100 * (tp + tn) / (tp + tn + fp + fn), 2),
        'sensitivity': round(100 * tp / max(tp + fn, 1), 2),   # recall on malignant
        'specificity': round(100 * tn / max(tn + fp, 1), 2),
        'precision': round(100 * tp / max(tp + fp, 1), 2),
        'auc_roc': round(float(roc_auc_score(y_true, y_prob)), 4),
        'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def train_xgboost(X_train, X_test, y_train, y_test,
                  model_path='models/xgboost_model.pkl'):
    """
    Early stopping uses a validation slice carved out of the TRAINING data,
    so the test set stays untouched until the final evaluation.
    """
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train)

    model, name = get_classifier()
    if name == 'XGBoost':
        model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
        best_iter = int(getattr(model, 'best_iteration', 0) or 0)
    else:
        model.fit(X_train, y_train)
        best_iter = int(getattr(model, 'n_iter_', 0))

    y_prob = model.predict_proba(X_test)[:, 1]
    m = binary_metrics(y_test, y_prob)

    print("=" * 40)
    print(f"  MODULE 2 RESULTS ({name})")
    print("=" * 40)
    print(f"  Accuracy    : {m['accuracy']}%")
    print(f"  Sensitivity : {m['sensitivity']}%   (malignant cases caught)")
    print(f"  Specificity : {m['specificity']}%")
    print(f"  AUC-ROC     : {m['auc_roc']}")
    print(classification_report(y_test, (y_prob >= .5).astype(int),
                                target_names=['Benign', 'Malignant']))

    os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
    joblib.dump(model, model_path)

    metrics = {'model': name, 'best_iteration': best_iter, **m}
    with open('models/tabular_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    return model, metrics
