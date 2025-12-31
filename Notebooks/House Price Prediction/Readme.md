# 🏠 House Prices – End-to-End Regression Pipeline

## Overview
This project demonstrates a complete **end-to-end machine learning workflow** for predicting residential property sale prices using structured tabular data.

The focus of the project is not just model performance, but **correct ML practices**:
- thoughtful exploratory data analysis
- semantic handling of missing values
- leakage-safe preprocessing using pipelines
- appropriate evaluation metrics
- model ensembling

The solution is implemented using **scikit-learn** and is designed to be clear, reproducible, and interview-ready.

---

## Problem Statement
The objective is to predict the final sale price of residential homes based on a mix of numerical and categorical features describing their properties.

This is a **supervised regression problem** evaluated using **Root Mean Squared Log Error (RMSLE)**, which penalizes relative prediction errors and motivates training in log-transformed target space.

---

## Dataset
The dataset contains housing attributes such as:
- living area and lot size
- quality and condition metrics
- year built and remodel information
- garage and basement characteristics
- neighborhood and zoning categories

The data used in this project is publicly available and commonly used for benchmarking regression models.

> **Note:** The dataset itself is not included in this repository and is subject to Kaggle’s dataset license.

---

## Project Structure
```
house-prices-ml-pipeline/
│
├── notebooks/
│   └── House_Prices_End_to_End_Regression_Pipeline.ipynb
    └── README.md
│
├── requirements.txt
```

---

## Approach

### 1. Exploratory Data Analysis (EDA)
- Examined target distribution and identified right skew
- Inspected numerical vs categorical feature types
- Analyzed missing value patterns
- Visualized key relationships (e.g., living area vs price)

### 2. Feature Engineering
Added domain-informed features to simplify learning:
- **TotalSF** – total usable living area
- **HouseAge** – age of the house at time of sale
- **RemodAge** – years since last remodel
- **HasGarage** – binary indicator for garage presence
- **HasBasement** – binary indicator for basement presence

### 3. Preprocessing Pipeline
Implemented using `Pipeline` and `ColumnTransformer` to avoid data leakage:
- Numerical features where missing implies absence → filled with `0`
- Remaining numerical features → median imputation
- Categorical features → `"None"` imputation + one-hot encoding

### 4. Modeling
Models trained on `log1p(SalePrice)`:
- Random Forest Regressor
- Gradient Boosting Regressor
- Ridge Regression

### 5. Evaluation
- Validation performed using **RMSE in log space**
- Metric choice aligns with RMSLE behavior
- Prevented common pitfalls such as double-logging

### 6. Ensembling
- Combined predictions using a weighted average
- Ensembling reduced variance and improved stability

---

## Results
- Validation RMSE (log space): **~0.13**
- Demonstrates a strong baseline with correct modeling practices

---

## Key Takeaways
- Proper preprocessing and evaluation matter more than model choice alone
- Pipelines help prevent subtle data leakage issues
- Feature engineering based on domain understanding provides consistent gains
- Ensembling improves robustness even with simple models

---

## Possible Improvements
- K-Fold cross-validated ensembling
- Gradient boosting libraries such as LightGBM or XGBoost
- Hyperparameter tuning via cross-validation
- Additional interaction and neighborhood-level features

---

## Tools & Libraries
- Python
- pandas, numpy
- scikit-learn
- matplotlib

---

## Author
This project was developed as part of a hands-on effort to strengthen practical machine learning skills and build a production-minded regression workflow.
