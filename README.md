# Loan Approval Prediction using Machine Learning

This project predicts whether a loan application will be approved or rejected using machine learning techniques.  
It follows an **end-to-end ML pipeline** including preprocessing, feature engineering, imbalance handling, model tuning, and deployment-ready inference.

---

## 🚀 Features
- Leak-free ML pipeline
- Feature engineering for credit risk
- SMOTE for class imbalance
- Gradient Boosting classifier
- ROC-AUC optimized
- Model persistence with Joblib
- Batch & single application prediction

---

## 📊 Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn
- Matplotlib, Seaborn

---

## 📁 Project Structure
loan-approval-prediction-ml/
│
├── data/
│   ├── credit_risk_dataset.csv
│   ├── new_applications.csv
│
├── models/
│   ├── loan_approval_model.pkl
│
├── notebooks/
│   ├── EDA.ipynb
│
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── preprocessing.py
│
├── outputs/
│   ├── submission.csv
│   ├── new_applications_with_predictions.csv
│
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE


## 📊 Exploratory Data Analysis

- Correlation heatmaps used to analyze numerical feature relationships
- Identified strong predictors such as:
  - loan_percent_income
  - income_to_loan_ratio
- Helped verify absence of severe multicollinearity

<img width="1920" height="1140" alt="image" src="https://github.com/user-attachments/assets/900eaecd-37a1-4ca0-9dc2-3ae2b63c3081" />

<img width="1920" height="1140" alt="image" src="https://github.com/user-attachments/assets/edd0770b-ef2e-4d88-b85c-865bdc9483ed" />

<img width="1920" height="1140" alt="image" src="https://github.com/user-attachments/assets/db88d335-f781-441d-8bfd-9ba27531b41c" />






