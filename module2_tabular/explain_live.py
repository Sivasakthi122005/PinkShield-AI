"""Per-patient SHAP explanation (computed for the values actually entered)."""
import json
import os

import numpy as np


def load_feature_names(path='models/feature_names.json'):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def explain_single(model, scaler, features, feature_names=None, max_display=10):
    """
    Returns a matplotlib Figure (SHAP waterfall) or None if shap is unavailable.
    Never raises - the dashboard falls back gracefully.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import shap

        feature_names = feature_names or load_feature_names() \
            or [f'f{i}' for i in range(len(features))]
        x = scaler.transform(np.array(features, dtype=float).reshape(1, -1))

        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(x)
        if isinstance(sv, list):          # older shap: one array per class
            sv = sv[1]
        sv = np.asarray(sv)[0]
        base = explainer.expected_value
        base = float(np.ravel(base)[-1])

        fig = plt.figure(figsize=(8, 5))
        shap.waterfall_plot(
            shap.Explanation(values=sv, base_values=base, data=x[0],
                             feature_names=feature_names),
            max_display=max_display, show=False)
        plt.tight_layout()
        return fig
    except Exception:
        return None
