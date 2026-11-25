# PinkShield-AI
# Multi-Modal Breast Cancer Risk Analysis

## Project Overview

OncAI is a multi-modal AI-based breast cancer risk analysis and awareness platform that combines histopathology image analysis, clinical measurements, and symptom-text classification to generate a unified 0–100 risk score.

The platform integrates explainable AI techniques and a simulated federated learning approach to demonstrate collaborative model training across three hospitals. It also provides an interactive dashboard with breast cancer awareness information and healthcare resources.

## Key Features

- **Image Analysis:** Uses EfficientNet-B0 to classify breast histopathology images.
- **Clinical Prediction:** Uses XGBoost to analyze 30 clinical measurement features.
- **Symptom Analysis:** Applies TF-IDF and Logistic Regression to classify symptom descriptions.
- **Risk Score Fusion:** Combines available model predictions into a unified 0–100 risk score.
- **Explainable AI:** Integrates Grad-CAM heatmaps and SHAP explanations to interpret model predictions.
- **Federated Learning:** Simulates FedAvg-based collaborative training across three hospitals.
- **Performance Dashboard:** Displays model evaluation metrics, ROC curves, and confusion matrices.
- **Awareness & Resources:** Provides breast cancer awareness information and links to trusted healthcare organizations.
- **Donation Section:** Connects users to official cancer charity donation pages.

## Technologies Used

- **Programming Language:** Python
- **Deep Learning:** PyTorch, EfficientNet-B0
- **Machine Learning:** XGBoost, Scikit-learn
- **Natural Language Processing:** TF-IDF, Logistic Regression
- **Explainable AI:** Grad-CAM, SHAP
- **Federated Learning:** FedAvg, NumPy
- **Data Processing:** Pandas, NumPy
- **Data Visualization:** Matplotlib
- **Web Framework:** Streamlit

## System Architecture

1. **Image Processing:** Extracts features from histopathology images using EfficientNet-B0.
2. **Clinical Analysis:** Predicts risk using XGBoost trained on Wisconsin Breast Cancer Dataset (WDBC).
3. **Text Processing:** Analyzes symptom descriptions using TF-IDF and Logistic Regression.
4. **Multi-Modal Fusion:** Combines available prediction probabilities using weighted fusion.
5. **Risk Assessment:** Generates a risk score, category, recommendation, and report.
6. **Federated Learning:** Evaluates model training across three simulated hospital clients using FedAvg.

## Datasets

- **BreakHis:** Histopathological breast cancer image dataset.
- **Wisconsin Diagnostic Breast Cancer (WDBC):** Clinical measurement dataset.
- **Synthetic Symptom Dataset:** Template-generated symptom descriptions for NLP model development.

## Project Structure

- `module1_image/` – Image classification and Grad-CAM.
- `module2_tabular/` – Clinical data prediction and SHAP.
- `module3_nlp/` – Symptom-text classification.
- `module4_fusion/` – Multi-modal fusion and risk report generation.
- `module5_federated/` – Federated learning simulation.
- `module6_dashboard/` – Streamlit application and awareness pages.
- `module7_evaluation/` – Model evaluation and performance analysis.
- `models/` – Trained models and evaluation metrics.

## Installation and Execution

1. Clone the repository:

   ```bash
   git clone YOUR_GITHUB_REPOSITORY_LINK
   cd breast-cancer-final
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:

   ```bash
   streamlit run module6_dashboard/app.py
   ```

4. To execute the complete training and evaluation pipeline:

   ```bash
   python run_all.py
   ```

## Important Note

This project is developed for academic and research demonstration purposes only. It is not a medical diagnostic device. The NLP dataset is synthetic, and federated learning is simulated on a single machine.

## Project Objective

To demonstrate how multiple AI models, explainable AI techniques, and federated learning can be integrated into a unified breast cancer risk analysis and awareness platform.
