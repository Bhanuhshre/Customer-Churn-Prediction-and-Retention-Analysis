# 📊 Customer-Churn-Prediction-and-Rentention-Analysis

This project predicts whether a telecom customer is likely to leave (churn) 
using Machine Learning models like Logistic Regression and Random Forest. 
It analyzes customer data such as tenure, monthly charges, and internet service 
to find patterns that lead to churn and help businesses retain their customers.

---

## 📁 Project Structure
```
telecom-churn-project/
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Dataset
├── telecom_churn.py                         # Main code
└── README.md                                # Project info
```

---

## 📌 About the Dataset
- **Source:** IBM / Kaggle
- **Rows:** 7,043 customers
- **Columns:** 21 features
- **Target:** Churn (Yes / No)

### Features Include:
| Feature | Description |
|---|---|
| `gender` | Male or Female |
| `SeniorCitizen` | Whether customer is senior (1/0) |
| `tenure` | Months with the company |
| `InternetService` | DSL / Fiber optic / No |
| `Contract` | Month-to-month / One year / Two year |
| `MonthlyCharges` | Monthly bill amount |
| `TotalCharges` | Total amount charged |
| `Churn` | Whether customer left (Yes/No) |

---

## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## 🤖 Models Used
| Model | Description |
|---|---|
| Logistic Regression | Simple classification model |
| Random Forest | Ensemble tree based model |

---

## 📈 Project Steps
1. Load Dataset
2. Data Exploration
3. Data Cleaning
4. Data Visualization
5. Label Encoding
6. Train/Test Split (80/20)
7. Feature Scaling
8. Model Training
9. Model Evaluation
10. ROC Curve
11. Feature Importance

---

## 🚀 How to Run
```bash
# Install libraries
pip install pandas numpy matplotlib seaborn scikit-learn

# Run the code
python telecom_churn.py
```

---

## 📊 Output Charts
- Churn Distribution
- Churn by Internet Service
- Tenure vs Churn
- Monthly Charges vs Churn
- Confusion Matrix
- ROC Curve
- Feature Importance

---

## 👨‍💻 Author
Made with ❤️ for learning Machine Learning
