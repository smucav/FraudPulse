# FraudPulse 

## 🚀 Project Overview

**FraudPulse** is a fraud detection project for e-commerce and banking transactions
The goal is to accurately identify fraudulent transactions by analyzing complex transaction data, engineering meaningful features, and applying machine learning models — all while addressing the challenge of highly imbalanced datasets.

This repository contains code, analysis, and documentation for Task 1: Data Analysis and Preprocessing.



## 📁 Dataset Description

The project uses the following datasets:

- `Fraud_Data.csv`: E-commerce transactions with user, device, purchase, and IP details, labeled as fraud or legitimate.
- `IpAddress_to_Country.csv`: IP address ranges mapped to countries for geolocation analysis.
- `creditcard.csv.zip`: Bank credit card transactions dataset with anonymized features for fraud detection.

Raw data files are stored in `data/raw/`.
Processed and cleaned data are saved in `data/processed/`.



## 📂 Repository Structure
```
FraudPulse/
├── .github/workflows/         # CI workflow files
├── data/
│   ├── raw/                   # Original raw datasets (not modified)
│   └── processed/             # Cleaned and feature-engineered datasets
├── notebooks/                 # Jupyter notebooks for EDA and preprocessing
├── scripts/                   # Modular scripts (future use)
├── .gitignore                 # Git ignore rules
├── requirements.txt           # Python dependencies
└── README.md                  # Project 
```

## 🛠️ Setup Instructions

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/FraudPulse.git
   cd FraudPulse
   ```
2. Create a Python virtual environment and activate it:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Place the raw dataset files inside **data/raw/** if not already present.

5. Open and run the EDA and preprocessing notebook:
```
jupyter notebook notebooks/01_fraud_data_eda_preprocessing.ipynb
```

## 1: Data Analysis and Preprocessing
This task includes:

- Comprehensive exploratory data analysis (EDA) focusing on fraud patterns.

- Data cleaning: handling missing values, duplicates, and correct data types.

### Feature engineering:

- Geolocation mapping from IP addresses.

- Time-based features like time_since_signup, hour_of_day, and day_of_week.

- Transaction frequency counts.

Handling class imbalance through analysis and strategy proposal.

Normalization and encoding of features for model readiness.


## 📝 Key Insights (from EDA)
- The dataset is highly imbalanced, with fraudulent transactions accounting for ~9.4%.

- Fraudulent transactions show distinct patterns in purchase values, user location, and transaction timing.

- The United States accounts for the highest fraud count, followed by other countries with significant variation.

- Time-based features reveal transaction activity trends linked to fraud.

(For detailed plots, tables, and interpretations, see the notebook.)

### 🔜 Next Steps
- Task 2: Model building and evaluation using Logistic Regression and ensemble methods.

- Task 3: Model explainability using SHAP to understand key fraud drivers.

- Finalize comprehensive report and visualization dashboards.
