# Project Report - Multi-Modal Breast Cancer Risk Analysis

## 1. Objective
Build a screening-support prototype that combines three evidence sources - a
histopathology image, clinical measurements and a free-text symptom description -
into a single explainable risk score, and study how federated learning can train
a model across hospitals without sharing patient records.

## 2. System architecture
Image -> EfficientNet-B0 -> P(malignant) -> Grad-CAM heatmap
Clinical (30 WDBC features) -> XGBoost -> P(malignant) -> SHAP waterfall
Symptom text -> TF-IDF + LR -> P(high risk)
All available probabilities -> weighted fusion -> 0-100 risk score -> category,
recommendation and report. A conflict detector flags disagreeing modalities.

## 3. Methods and design decisions
* **Image (Module 1).** Transfer learning, last layers fine-tuned. Train/validation split is
  by *patient* (parsed from BreakHis filenames) because slides of the same patient are
  near-duplicates; a random split inflates accuracy. Augmentation is applied to the
  training set only (an earlier version accidentally disabled it).
* **Clinical (Module 2).** Split first, scale on the training part only, early stopping on a
  validation slice of the training data - the test set is untouched until the end.
  Reported: accuracy, sensitivity, specificity, AUC, plus 5-fold CV mean +/- std (Module 7).
  Sensitivity matters most in screening (a missed cancer is worse than a false alarm).
* **Text (Module 3).** Synthetic template sentences. Even with whole templates held out the
  score saturates, because the two classes use different vocabulary. This shows the pipeline,
  not clinical validity, so it has the lowest fusion weight (0.15).
* **Fusion (Module 4).** Weights 0.45 / 0.40 / 0.15 are expert-set, not learned, because no
  public dataset contains all three modalities for the same patient. A missing modality is
  removed and the remaining weights re-normalised (never replaced by 0.5).
* **Federated (Module 5).** Three hospitals (sizes ~40/35/25 %), label-skewed (non-IID) via a
  Dirichlet partition, FedAvg from zero-initialised weights, 15 rounds. The global model is
  scored on a shared held-out test set. Baselines: centralized training and local-only.

## 4. Results (see the Performance page for live numbers)
| Module | Metric | Value | Note |
|---|---|---|---|
| Image | Val accuracy | ~97.7 % | Original run used a random split; retrain with the patient-level split for the honest figure. |
| Clinical | Test accuracy / AUC | 96.5 % / 0.997 | Shipped XGBoost model. Run `run_all.py` for CV mean +/- std. |
| Text | Accuracy | ~100 % | Synthetic data - not a clinical result. |
| Federated | Global accuracy | 96.5 % | Equals centralized (96.5 %); loss falls every round. |

Federated finding: FedAvg matches centralized training without pooling data. Local-only
models also score ~96-97 % because WDBC is an easy dataset, so the contribution is privacy
at no accuracy cost, not an accuracy gain.

## 5. Limitations (state these in the viva)
1. WDBC measures cells from fine-needle aspirates; BreakHis are biopsy slides. They are
   different patients, so fusion combines unrelated cases - it demonstrates mechanics, not
   joint clinical performance.
2. Text data is synthetic.
3. Fusion weights are heuristic and uncalibrated; probabilities are not guaranteed to be
   well calibrated (Brier score is reported).
4. Federated learning is simulated on one machine; there is no secure aggregation or
   differential privacy.
5. No external validation set and no regulatory approval; not for clinical use.

## 6. Future work
Paired multimodal data and learned fusion (stacking); probability calibration;
differential privacy / secure aggregation; external validation; multi-magnification
BreakHis models; real clinical notes with a transformer encoder.

## 7. Likely viva questions
* *Why drop a missing modality instead of using 0.5?* A neutral value drags every score toward
  "moderate" and fabricates evidence.
* *Why patient-level split?* Slides from one patient are near-identical; random splits leak.
* *Why is federated accuracy not higher than centralized?* FedAvg approximates centralized
  training; the benefit is privacy.
* *Why 100 % on text?* Synthetic, keyword-separable data; hence the 15 % weight.
* *What would you change for real clinical use?* Paired data, calibration, external
  validation, privacy guarantees, regulatory review.
