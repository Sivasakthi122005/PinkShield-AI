import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torchvision import models

from module2_tabular.predict_tabular import load_tabular_model, predict_tabular
from module3_nlp.predict_nlp import load_nlp_model, predict_text
from module4_fusion.fusion import (weighted_fusion, confidence_weighted_fusion,
                                   modality_agreement)
from module4_fusion.risk_score import compute_risk_score
from module4_fusion.report_generator import generate_report


def load_all_models(models_dir='models'):
    """Load all 3 trained models at once."""
    print("Loading all models...")

    # Module 2 — Tabular
    tabular_model, scaler = load_tabular_model(
        model_path=f'{models_dir}/xgboost_model.pkl',
        scaler_path=f'{models_dir}/scaler.pkl'
    )

    # Module 3 — NLP
    nlp_pipeline = load_nlp_model(
        model_path=f'{models_dir}/nlp_pipeline.pkl'
    )

    print("All models loaded successfully!")
    return tabular_model, scaler, nlp_pipeline


def run_fusion_pipeline(clinical_features: list,
                         symptom_text: str,
                         image_prob,
                         tabular_model, scaler, nlp_pipeline,
                         use_confidence_weighting=False) -> dict:
    """
    Full end-to-end fusion pipeline.

    Args:
        clinical_features : list of 30 Wisconsin feature values
        symptom_text      : plain English symptom description
        image_prob        : malignant probability from Module 1 (0-1)
        tabular_model     : loaded XGBoost model
        scaler            : loaded StandardScaler
        nlp_pipeline      : loaded NLP pipeline
        use_confidence_weighting: if True use adaptive weights

    Returns:
        Complete result dictionary with all module outputs + fusion
    """

    # --- Module 1 result (passed in from dashboard) ---
    img_ok = image_prob is not None
    p_img = image_prob if img_ok else 0.5
    image_result = {
        'probability': p_img,
        'label': 'Malignant' if p_img > 0.5 else 'Benign',
        'confidence': round(max(p_img, 1 - p_img) * 100, 1),
        'available': img_ok,
    }

    # --- Module 2 --- tabular prediction
    tabular_result = predict_tabular(clinical_features, tabular_model, scaler)

    # --- Module 3 --- NLP prediction
    nlp_result = predict_text(symptom_text, nlp_pipeline)
    nlp_result['available'] = nlp_result['label'] != 'Insufficient Input'
    img_p = image_result['probability'] if img_ok else None
    nlp_p = nlp_result['probability'] if nlp_result['available'] else None

    # --- Module 4 --- Fusion
    if use_confidence_weighting:
        fused_prob = confidence_weighted_fusion(
            img_p, tabular_result['probability'], nlp_p
        )
    else:
        fused_prob = weighted_fusion(img_p, tabular_result['probability'], nlp_p)

    # --- Risk Score ---
    agreement = modality_agreement(img_p, tabular_result['probability'], nlp_p)
    risk = compute_risk_score(fused_prob)

    # --- Report ---
    report = generate_report(image_result, tabular_result, nlp_result, risk)

    return {
        'image':    image_result,
        'tabular':  tabular_result,
        'nlp':      nlp_result,
        'fusion':   {'probability': fused_prob},
        'risk':     risk,
        'agreement': agreement,
        'report':   report
    }
