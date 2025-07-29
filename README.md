# 🚦 FraudPulse

## 🚀 Project Overview

**FraudPulse** is a fraud detection project for e-commerce and banking transactions.
The goal is to accurately identify fraudulent transactions by analyzing complex transaction data, engineering meaningful features, and applying robust machine learning models — all while addressing the challenge of highly imbalanced datasets.

This repository contains the complete pipeline for:

* ✅ **Task 1:** Data Analysis and Preprocessing
* ✅ **Task 2:** Model Building and Evaluation
* 🔜 **Task 3:** Model Explainability with SHAP

---

## 📁 Dataset Description

The project uses the following datasets:

* `Fraud_Data.csv` — E-commerce transactions with user, device, purchase, and IP details, labeled as fraud or legitimate.
* `IpAddress_to_Country.csv` — IP address ranges mapped to countries for geolocation analysis.
* `creditcard.csv` — Bank credit card transactions dataset with anonymized features for fraud detection.

Raw data files are stored in **`data/raw/`**.
Processed and cleaned data are saved in **`data/processed/`**.

---

## 📂 Repository Structure

```
FraudPulse/
├── .github/workflows/         # CI workflow files
├── data/
│   ├── raw/                   # Original raw datasets (unmodified)
│   └── processed/             # Cleaned & feature-engineered datasets
├── notebooks/                 # Jupyter notebooks for EDA, preprocessing, explainability
├── scripts/                   # Python scripts for preprocessing & training
├── models/                    # Saved models & scalers
├── .gitignore                 # Git ignore rules
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## 🛠️ Setup Instructions

1. **Clone this repository**

   ```bash
   git clone https://github.com/yourusername/FraudPulse.git
   cd FraudPulse
   ```

2. **Create a virtual environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Place raw dataset files**
   Place `Fraud_Data.csv`, `IpAddress_to_Country.csv`, and `creditcard.csv` inside **`data/raw/`**.

5. **Run notebooks or scripts**

   ```bash
   jupyter notebook notebooks/01_fraud_data_eda_preprocessing.ipynb
   ```

---

## ✅ **1️⃣ Task 1: Data Analysis & Preprocessing**

This task covers:

* Univariate & bivariate EDA to discover fraud patterns.
* Handling missing values, duplicates, and type corrections.
* Merging IP geolocation data.
* Feature engineering:

  * Geolocation mapping.
  * Time-based features: `time_since_signup`, `hour_of_day`, `day_of_week`.
  * Transaction frequency and velocity.
* Addressing class imbalance with techniques like SMOTE.
* Scaling numeric features and encoding categorical features.
* Saving final, ready-to-train datasets.

---

## ✅ **2️⃣ Task 2: Model Building & Evaluation**

This task is **complete** and includes:

* Loading processed datasets.
* Building and comparing **Logistic Regression** (baseline) and a **powerful ensemble model** (Random Forest for Fraud\_Data, XGBoost for creditcard.csv).
* Training on both **Fraud\_Data** and **creditcard.csv**.
* Evaluating using robust metrics for imbalanced data: Confusion Matrix, ROC AUC, AUC-PR, Precision, Recall, F1-Score.
* Saving the best-performing models for the next step.

---

## 📝 **Key Insights (so far)**

* The fraud label is extremely imbalanced (\~9.4% fraud).
* Fraudulent transactions cluster around certain purchase values, time delays, and user behaviors.
* IP geolocation reveals country-based fraud patterns.
* Certain transaction times and signup behaviors are linked with higher fraud risk.

(For full visualizations and interpretations, see the `notebooks/`.)


# Model Comparison & Justification

This document outlines the performance comparison between **Logistic Regression** and **Random Forest** models on two datasets:  
1. **Credit Card Fraud Dataset (`creditcard.csv`)**  
2. **E-commerce Fraud Dataset (`Fraud_Data.csv`)**  

Based on the evaluation metrics, the best-performing model for each dataset is selected.

---

## 📊 **Credit Card Dataset (`creditcard.csv`)**  

| Metric               | Logistic Regression | Random Forest |
|----------------------|---------------------|---------------|
| **ROC AUC**          | 0.967               | 0.969         |
| **AUC-PR**           | 0.706               | 0.828         |
| **F1-Score**         | 0.12                | 0.83          |
| **Recall (Fraud)**   | 0.88                | 0.78          |
| **Precision (Fraud)**| 0.06                | 0.89          |

### **Insight:**  
- **Logistic Regression** achieves high recall (88%) but suffers from extremely low precision (6%), leading to many false positives.  
- **Random Forest** provides a better balance:  
  - High precision (**89%**)  
  - Strong recall (**78%**)  
  - Better **F1-Score (0.83)** and **AUC-PR (0.828)**  

### **Final Choice:** ✅ **Random Forest**  
> Handles class imbalance more robustly, reducing false positives while maintaining strong fraud detection.  

---

## 🛒 **E-commerce Dataset (`Fraud_Data.csv`)**  

| Metric               | Logistic Regression | Random Forest |
|----------------------|---------------------|---------------|
| **ROC AUC**          | 0.772               | 0.771         |
| **AUC-PR**           | 0.632               | 0.628         |
| **F1-Score**         | 0.69                | 0.60          |
| **Recall (Fraud)**   | 0.54                | 0.55          |
| **Precision (Fraud)**| 0.96                | 0.66          |

### **Insight:**  
- **Logistic Regression** slightly outperforms Random Forest in:  
  - **AUC-PR (0.632 vs. 0.628)**  
  - **F1-Score (0.69 vs. 0.60)**  
- It also achieves **much higher precision (96% vs. 66%)** with similar recall (~55%).  

### **Final Choice:** ✅ **Logistic Regression**  
> Provides clearer separation of fraudulent transactions with fewer false positives.  

---

## ✅ **Summary of Best Models**  
| Dataset                  | Best Model         | Key Reason                          |
|--------------------------|--------------------|-------------------------------------|
| `creditcard.csv`         | **Random Forest**  | Better precision-recall trade-off   |
| `Fraud_Data.csv`         | **Logistic Regression** | Higher precision & F1-Score    |
---


## 🔜 **3️⃣ Task 3: Model Explainability**

Applied SHAP (Shapley Additive Explanations) to:
- Understand global feature importance using summary plots
- Explain individual predictions using force plots

SHAP is used for:
- Credit Card Fraud: Random Forest model
- E-commerce Fraud: Logistic Regression model

Generated SHAP plots are saved in `plots/` for reporting.

## 📝 Key Insights
- The datasets are highly imbalanced; fraud cases are rare
- Specific transaction amounts, time delays, and device usage patterns indicate fraud signals
- SHAP analysis reveals which features most influence fraud predictions — giving practical insights for fraud prevention strategies


## 👤 Author
FraudPulse — by [Daniel Tujuma]
---


**Let’s detect fraud — one transaction at a time! 🚦✨**
