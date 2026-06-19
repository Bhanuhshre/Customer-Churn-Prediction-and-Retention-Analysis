# 📊 Customer Churn Prediction and Retention Analysis

A machine learning project that predicts whether a telecom customer will churn (leave the service) using classification algorithms such as Logistic Regression and Random Forest. Includes data analysis, preprocessing, visualization, and model evaluation.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 📝 Overview

Customer churn refers to the phenomenon where customers stop using a company's service. In the telecom industry, retaining customers is crucial for business growth. This project builds a predictive system using machine learning to identify customers at risk of churning based on various attributes such as tenure, monthly charges, internet service, and contract type.

Multiple supervised ML algorithms were used, their performance was compared, and visualizations were created to understand churn patterns and help businesses retain their customers.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## ✨ Project Features

- Load and analyze real-world telecom customer churn dataset

- Clean and preprocess data using encoding and scaling

- Perform Exploratory Data Analysis (EDA) with visualizations

- Train and evaluate multiple ML classification models

- Compare model accuracy in a clear tabular form

- Visualize confusion matrix and ROC curve for each model

- Feature importance analysis to identify key churn factors

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 📌 Dataset Features

- **Source**: [Kaggle - Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

The dataset contains telecom customer details and whether they churned or not.

| Feature | Description |
|---|---|
| `gender` | Male or Female |
| `SeniorCitizen` | Whether customer is a senior citizen (1/0) |
| `Partner` | Whether customer has a partner (Yes/No) |
| `Dependents` | Whether customer has dependents (Yes/No) |
| `tenure` | Number of months with the company |
| `PhoneService` | Whether customer has phone service (Yes/No) |
| `MultipleLines` | Whether customer has multiple lines |
| `InternetService` | DSL / Fiber optic / No |
| `OnlineSecurity` | Whether customer has online security |
| `Contract` | Month-to-month / One year / Two year |
| `PaperlessBilling` | Whether customer uses paperless billing |
| `PaymentMethod` | Payment method used by customer |
| `MonthlyCharges` | Monthly bill amount |
| `TotalCharges` | Total amount charged |
| `Churn` | Target label (Yes = Churned, No = Retained) |

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 🛠️ Tools & Technologies Used

- **Python**
- **Pandas**, **NumPy**
- **Matplotlib**, **Seaborn**
- **Scikit-learn** (ML models, preprocessing, evaluation)

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 🤖 Machine Learning Models & Accuracy Comparison

| Machine Learning Model | Accuracy |
|---|---|
| Logistic Regression | ~80% |
| Random Forest Classifier | ~82% |

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 📈 Project Workflow

1. Import Required Libraries
2. Load Dataset (CSV file)
3. Perform Data Exploration and Analysis
4. Data Cleaning (Fix TotalCharges, Drop customerID)
5. Data Visualization using Seaborn and Matplotlib
6. Label Encoding for Categorical Columns
7. Train/Test Split (80% Train, 20% Test)
8. Feature Scaling using StandardScaler
9. Train Machine Learning Models
10. Evaluate Models (Accuracy, Confusion Matrix, ROC Curve)
11. Feature Importance Analysis

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## ✅ Results

- Random Forest achieved the highest accuracy among the models used.

- Data visualization helped in understanding churn trends and feature correlations.

- Customers with **Fiber optic** internet and **Month-to-month** contracts showed the highest churn rate.

- **Tenure** and **Monthly Charges** were found to be the most important features for predicting churn.

- The code is modular and reusable, making it easy to scale or improve further.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 📚 References

- https://www.kaggle.com/datasets/blastchar/telco-customer-churn

- https://www.analyticsvidhya.com/blog/2022/09/bank-customer-churn-prediction-using-machine-learning/

- https://medium.com/@allanouko17/customer-churn-prediction-using-machine-learning-ddf4cd7c9fd4

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 👩‍💻 Author

Varikuti Bhanuhshre
BTech Computer Science and Engineering


------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 📎 License

This project is open-source and available under the [MIT License](LICENSE).
