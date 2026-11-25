"""
Create data/wisconsin.csv WITHOUT any download.

The Wisconsin Diagnostic Breast Cancer (WDBC) dataset ships inside
scikit-learn, so the project can be reproduced fully offline.
Column names / order match the popular Kaggle CSV (id, diagnosis M/B,
radius_mean ... fractal_dimension_worst).
"""
import os
import pandas as pd
from sklearn.datasets import load_breast_cancer


def _rename(col: str) -> str:
    """'mean radius' -> 'radius_mean', 'worst concave points' -> 'concave points_worst'."""
    col = col.strip()
    if col.startswith('mean '):
        return col[5:] + '_mean'
    if col.startswith('worst '):
        return col[6:] + '_worst'
    if col.endswith(' error'):
        return col[:-6] + '_se'
    return col


def prepare_wisconsin(output_path: str = 'data/wisconsin.csv') -> pd.DataFrame:
    data = load_breast_cancer(as_frame=True)
    df = data.data.copy()
    df.columns = [_rename(c) for c in df.columns]
    # sklearn target: 0 = malignant, 1 = benign  ->  M / B
    df.insert(0, 'diagnosis', data.target.map({0: 'M', 1: 'B'}))
    df.insert(0, 'id', range(1, len(df) + 1))

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Wisconsin dataset written: {len(df)} rows, "
          f"{df.shape[1] - 2} features -> {output_path}")
    return df


if __name__ == '__main__':
    prepare_wisconsin()
