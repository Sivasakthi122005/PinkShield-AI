import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module2_tabular.preprocess import load_and_preprocess
from module2_tabular.train_xgboost import train_xgboost

print("Starting Module 2 Training...\n")

X_train, X_test, y_train, y_test, feature_names, scaler = load_and_preprocess()
model, metrics = train_xgboost(X_train, X_test, y_train, y_test)

print("\nGenerating SHAP explanations...")
try:
    from module2_tabular.shap_explain import generate_shap_explanation, generate_shap_summary
    generate_shap_explanation(model, X_test[:1], feature_names)
    generate_shap_summary(model, X_test, feature_names)
except Exception as e:          # shap missing / version issue should not kill training
    print(f"  (SHAP charts skipped: {e})")

print("\nModule 2 Complete!")
print(f"Accuracy : {metrics['accuracy']}%  |  AUC-ROC : {metrics['auc_roc']}")
