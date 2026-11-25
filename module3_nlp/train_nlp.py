import json
import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline


def build_pipeline():
    return Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=8000,
                                  min_df=1, sublinear_tf=True)),
        ('clf', LogisticRegression(max_iter=1000, C=0.5, random_state=42,
                                   class_weight='balanced')),
    ])


def train_nlp_pipeline(csv_path='data/symptom_data.csv',
                       model_path='models/nlp_pipeline.pkl'):
    df = pd.read_csv(csv_path)
    X, y = df['text'], df['label']

    # (a) optimistic: random split - near-duplicate sentences leak into test
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          random_state=42, stratify=y)
    p = build_pipeline().fit(Xtr, ytr)
    random_acc = accuracy_score(yte, p.predict(Xte))

    # (b) honest: whole templates are held out, so the model must generalise
    if 'template_id' in df.columns:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        tr, te = next(gss.split(X, y, groups=df['template_id']))
        pg = build_pipeline().fit(X.iloc[tr], y.iloc[tr])
        prob = pg.predict_proba(X.iloc[te])[:, 1]
        grouped_acc = accuracy_score(y.iloc[te], prob >= .5)
        grouped_auc = roc_auc_score(y.iloc[te], prob)
    else:
        grouped_acc = grouped_auc = None

    # final model: trained on all synthetic data
    pipeline = build_pipeline().fit(X, y)
    os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
    joblib.dump(pipeline, model_path)

    metrics = {
        'accuracy': round(100 * (grouped_acc if grouped_acc is not None else random_acc), 2),
        'auc_roc': round(grouped_auc if grouped_auc is not None else
                         roc_auc_score(yte, p.predict_proba(Xte)[:, 1]), 4),
        'random_split_accuracy': round(100 * random_acc, 2),
        'evaluation': 'held-out templates' if grouped_acc is not None else 'random split',
        'data': 'synthetic (template generated) - NOT real patient notes',
    }
    with open('models/nlp_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print("=" * 40)
    print("  MODULE 3 RESULTS (synthetic data)")
    print("=" * 40)
    print(f"  Random-split accuracy      : {metrics['random_split_accuracy']}%  (random split)")
    print(f"  Held-out-template accuracy : {metrics['accuracy']}%  (held-out)")
    print(f"  AUC-ROC (held-out)         : {metrics['auc_roc']}")
    return pipeline, metrics
