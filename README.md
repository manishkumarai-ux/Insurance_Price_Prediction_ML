# Insurance_Price_Prediction_ML
Predict insurance premium charges based on customer demographics and lifestyle factors.
# Insurance Expense Prediction using Machine Learning

## Project Overview

This project focuses on predicting individual medical insurance expenses using Machine Learning Regression techniques. The objective is to estimate insurance expenses based on demographic and lifestyle-related factors such as age, gender, BMI, number of children, smoking status, and region.

The project follows a complete Machine Learning lifecycle including data exploration, feature engineering, data preprocessing, model training, evaluation, and prediction.

---

## Problem Statement

Insurance companies need accurate estimates of medical insurance expenses to assess risk and determine appropriate premium pricing.

The goal of this project is to build a Machine Learning model capable of predicting medical insurance expenses based on customer information.

---

## Business Objective

* Predict medical insurance expenses accurately.
* Support data-driven premium estimation.
* Improve customer risk assessment.
* Compare multiple regression algorithms and select the best-performing model.

---

## Dataset Information

### Features

| Feature  | Description                                  |
| -------- | -------------------------------------------- |
| age      | Age of the insured person                    |
| sex      | Gender of the insured person                 |
| bmi      | Body Mass Index                              |
| children | Number of dependent children                 |
| smoker   | Smoking status (Yes/No)                      |
| region   | Residential region                           |
| expenses | Medical insurance expenses (Target Variable) |

### Dataset Summary

* Total Records: 1338
* Total Features: 6
* Target Variable: expenses
* Problem Type: Supervised Machine Learning (Regression)

---

## Project Architecture

Data Collection
↓
Data Understanding
↓
Exploratory Data Analysis (EDA)
↓
Data Cleaning
↓
Categorical Encoding
↓
Train-Test Split
↓
Log Transformation of Target Variable
↓
Feature Scaling
↓
Model Training
↓
Model Evaluation
↓
Best Model Selection
↓
Expense Prediction

---

## Exploratory Data Analysis

The following analyses were performed:

* Data quality checks
* Missing value analysis
* Duplicate record analysis
* Distribution analysis
* Outlier analysis
* Correlation analysis
* Feature relationship analysis

### Visualizations

* Insurance Expense Distribution
* Age Distribution
* BMI Distribution
* Smoker vs Expenses
* Region vs Expenses
* Correlation Heatmap
* Model Performance Comparison

---

## Data Preprocessing

### Categorical Encoding

Applied One-Hot Encoding on:

* sex
* smoker
* region

### Log Transformation

Applied logarithmic transformation on the target variable (`expenses`) to reduce skewness and improve model performance.

```python
y_train_log = np.log(y_train)
```

### Feature Scaling

Applied StandardScaler to standardize numerical features before training.

```python
scaler = StandardScaler()
```

---

## Machine Learning Models Used

The following regression algorithms were trained and evaluated:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. ElasticNet Regression
5. Decision Tree Regressor
6. Random Forest Regressor
7. Gradient Boosting Regressor
8. XGBoost Regressor

---

## Evaluation Metrics

The models were evaluated using:

* R² Score
* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)

---

## Technology Stack

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost

### Development Environment

* Jupyter Notebook
* VS Code

---

## Project Structure

Insurance-Expense-Prediction/


├── data/
│ └── insurance.csv

│
├── notebooks/
│ └── Insurance_Expense_Prediction.ipynb
│
├── src/
│ ├── data_preprocessing.py
│ ├── train_model.py
│ ├── evaluate_model.py
│ └── predict.py
│
├── models/
│ └── best_model.pkl
│
├── images/
│ ├── expense_distribution.png
│ ├── correlation_heatmap.png
│ └── model_comparison.png
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore

---

## Key Achievements

* Completed end-to-end Machine Learning pipeline.
* Performed comprehensive exploratory data analysis.
* Applied log transformation to improve prediction accuracy.
* Compared multiple regression algorithms.
* Selected the best-performing model using evaluation metrics.
* Developed a reusable prediction workflow.

---

## Future Enhancements

* Hyperparameter Tuning
* Streamlit Web Application
* Docker Containerization
* AWS Cloud Deployment
* CI/CD Pipeline Integration
* Model Monitoring Dashboard

---

## Author

Manish Singla

Project Manager | AI/ML Enthusiast | Telecom & IT Professional

---
