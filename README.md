# ChurnSense-AI 📊

An end-to-end machine learning pipeline built to identify at-risk telecommunication customers, assess churn probability, and prioritize retention interventions using Scikit-Learn.

---

## 📌 Project Overview

Customer churn poses a major revenue risk in the subscription telecom sector. **ChurnSense-AI** predicts whether a customer will discontinue their service based on account demographics, contract structures, and service usage patterns.

Rather than optimizing solely for raw accuracy on an imbalanced dataset (~73% non-churn vs. ~27% churn), this pipeline emphasizes **ROC-AUC** and **Recall**, catching over **76% of actual churners** via tuned classification thresholds.

---

## 🚀 Key Results (Holdout Test Set)

| Metric | Score | Business Interpretation |

| **ROC-AUC** | **0.8440** | Strong separation and ranking of customer churn risk |
| **Recall (Churn / Class 1)** | **76.74%** | Captures 287 out of 374 actual churners |
| **Precision (Churn / Class 1)** | **52.66%** | Roughly 1 in 2 flagged customers are confirmed churners |
| **F1-Score (Churn / Class 1)** | **0.6246** | Balanced harmonic mean between precision and recall |
| **Overall Accuracy** | **75.51%** | Total correct predictions across all test observations |

### Confusion Matrix Breakdown (1,409 Test Samples)
* **True Negatives (TN):** 777 (Loyal customers correctly classified)
* **False Positives (FP):** 258 (Loyal customers flagged for retention check)
* **False Negatives (FN):** 87 (Churners missed by the model)
* **True Positives (TP):** 287 (Churners successfully intercepted)

---

## 🛠️ Pipeline Architecture

The workflow is cleanly modularized into three analytical notebooks:

1. **`notebooks/data_clean.ipynb` (Data Sanitization):**
   * Normalizes column names and lowercase string tokens.
   * Coerces numeric parsing errors in `TotalCharges` and imputes using median values.
   * Maps binary fields (`Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, `Churn`) into numeric indicators (0/1).

2. **`notebooks/eda.ipynb` (Exploratory Analysis & Feature Selection):**
   * Computes **Mutual Information (MI)** scores against the target to measure non-linear feature dependency.
   * Prunes low-importance and identifier columns (`customerID`, `gender`, `MultipleLines`).

3. **`notebooks/model.ipynb` (Model Training & Evaluation):**
   * Employs stratified train/validation/test splitting (60% / 20% / 20%) to prevent data leakage.
   * Uses an object-oriented pipeline encapsulation combining `DictVectorizer`, `StandardScaler`, and `LogisticRegression`.
   * Evaluates baseline stability via 10-Fold Cross-Validation on the training split.
   * Sweeps classification decision thresholds to optimize the F1/Recall operational point.
   * Serializes the complete production pipeline bundle into `model.bin`.

---

## 📁 Repository Structure

```text
CHURNSENSE-AI/
│
├── data/
│   ├── raw/                   # Raw Telco-Customer-Churn dataset
│   └── processed/             # Cleaned and feature-selected datasets
│
├── notebooks/
│   ├── data_clean.ipynb       # Data ingestion, typecasting, and imputation
│   ├── eda.ipynb              # Mutual information and feature selection
│   ├── model.ipynb            # Model training, CV, threshold tuning, and evaluation
│   └── model.bin              # Exported inference pipeline artifact
│
├── .gitignore                 # Excluded environments, caches, and checkpoint files
├── requirements.txt           # Environment dependencies
└── README.md                  # Project documentation

