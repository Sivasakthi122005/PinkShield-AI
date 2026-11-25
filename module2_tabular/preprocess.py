import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_raw(csv_path='data/wisconsin.csv'):
    """Return (X DataFrame, y array, feature_names). Auto-creates the CSV if missing."""
    if not os.path.exists(csv_path):
        from module2_tabular.prepare_data import prepare_wisconsin
        prepare_wisconsin(csv_path)

    df = pd.read_csv(csv_path)
    df = df.drop(columns=[c for c in ('id', 'Unnamed: 32') if c in df.columns])
    y = df['diagnosis'].map({'M': 1, 'B': 0}).values      # 1 = malignant
    X = df.drop(columns='diagnosis')
    return X, y, list(X.columns)


def load_and_preprocess(csv_path='data/wisconsin.csv',
                        scaler_path='models/scaler.pkl',
                        features_path='models/feature_names.json'):
    """
    Split FIRST, then fit the scaler on the training portion only
    (the old version scaled the full dataset before splitting = data leakage).
    """
    X, y, feature_names = load_raw(csv_path)
    print(f"Features: {len(feature_names)} | Samples: {len(X)} | "
          f"Malignant: {int(y.sum())} | Benign: {int((y == 0).sum())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    os.makedirs(os.path.dirname(scaler_path) or '.', exist_ok=True)
    joblib.dump(scaler, scaler_path)
    with open(features_path, 'w') as f:
        json.dump(feature_names, f)
    print(f"Scaler saved to {scaler_path} | Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train_s, X_test_s, y_train, y_test, feature_names, scaler
