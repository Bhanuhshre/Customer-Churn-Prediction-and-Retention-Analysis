# =============================================================
#         TELECOM CUSTOMER CHURN PREDICTION PROJECT
# =============================================================

# -----------------------------------------------
# STEP 1 - IMPORT LIBRARIES
# -----------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, roc_auc_score, roc_curve
)

import warnings
warnings.filterwarnings('ignore')

print("=" * 50)
print("   TELECOM CUSTOMER CHURN PREDICTION")
print("=" * 50)


# -----------------------------------------------
# STEP 2 - LOAD DATASET
# -----------------------------------------------
df = pd.read_csv('telecom_churn.csv')

print("\n✅ Dataset Loaded Successfully!")
print(f"   Rows    : {df.shape[0]}")
print(f"   Columns : {df.shape[1]}")


# -----------------------------------------------
# STEP 3 - DATA EXPLORATION
# -----------------------------------------------
print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Churn Distribution ---")
print(df['Churn'].value_counts())


# -----------------------------------------------
# STEP 4 - DATA CLEANING
# -----------------------------------------------

# Fix TotalCharges column (it has spaces, convert to number)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill missing TotalCharges with median
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Drop customerID (not useful for prediction)
df.drop('customerID', axis=1, inplace=True)

print("\n✅ Data Cleaning Done!")


# -----------------------------------------------
# STEP 5 - VISUALIZATION
# -----------------------------------------------

# Plot 1 - Churn Count
plt.figure(figsize=(5, 4))
df['Churn'].value_counts().plot(kind='bar', color=['steelblue', 'tomato'])
plt.title('Churn Distribution')
plt.xlabel('Churn')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('churn_distribution.png')
plt.show()
print("✅ Saved: churn_distribution.png")

# Plot 2 - Churn by Internet Service
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='InternetService', hue='Churn', palette='Set2')
plt.title('Churn by Internet Service')
plt.tight_layout()
plt.savefig('churn_by_internet.png')
plt.show()
print("✅ Saved: churn_by_internet.png")

# Plot 3 - Tenure vs Churn
plt.figure(figsize=(6, 4))
sns.histplot(data=df, x='tenure', hue='Churn', bins=30, palette='Set1')
plt.title('Tenure vs Churn')
plt.tight_layout()
plt.savefig('tenure_vs_churn.png')
plt.show()
print("✅ Saved: tenure_vs_churn.png")

# Plot 4 - Monthly Charges vs Churn
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x='Churn', y='MonthlyCharges', palette='coolwarm')
plt.title('Monthly Charges vs Churn')
plt.tight_layout()
plt.savefig('monthly_charges_vs_churn.png')
plt.show()
print("✅ Saved: monthly_charges_vs_churn.png")


# -----------------------------------------------
# STEP 6 - ENCODE CATEGORICAL COLUMNS
# -----------------------------------------------
le = LabelEncoder()

# Get all object (text) columns
cat_cols = df.select_dtypes(include='object').columns

for col in cat_cols:
    df[col] = le.fit_transform(df[col])

print("\n✅ Label Encoding Done!")


# -----------------------------------------------
# STEP 7 - SPLIT DATA INTO FEATURES AND TARGET
# -----------------------------------------------
X = df.drop('Churn', axis=1)   # Features
y = df['Churn']                 # Target

# Split into Train (80%) and Test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n✅ Data Split Done!")
print(f"   Training samples : {X_train.shape[0]}")
print(f"   Testing samples  : {X_test.shape[0]}")


# -----------------------------------------------
# STEP 8 - SCALE THE FEATURES
# -----------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print("\n✅ Feature Scaling Done!")


# -----------------------------------------------
# STEP 9 - TRAIN MODELS
# -----------------------------------------------

# --- Model 1: Logistic Regression ---
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# --- Model 2: Random Forest ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("\n✅ Models Trained!")


# -----------------------------------------------
# STEP 10 - EVALUATE MODELS
# -----------------------------------------------
def evaluate_model(name, y_test, y_pred, model, X_test):
    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")
    print(f"  Accuracy  : {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"  ROC-AUC   : {roc_auc_score(y_test, y_pred):.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    filename = f"confusion_matrix_{name.replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.show()
    print(f"✅ Saved: {filename}")

evaluate_model("Logistic Regression", y_test, lr_pred, lr_model, X_test)
evaluate_model("Random Forest",       y_test, rf_pred, rf_model, X_test)


# -----------------------------------------------
# STEP 11 - ROC CURVE
# -----------------------------------------------
plt.figure(figsize=(7, 5))

for model, name in [(lr_model, 'Logistic Regression'),
                    (rf_model, 'Random Forest')]:
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.tight_layout()
plt.savefig('roc_curve.png')
plt.show()
print("✅ Saved: roc_curve.png")


# -----------------------------------------------
# STEP 12 - FEATURE IMPORTANCE (Random Forest)
# -----------------------------------------------
feature_names = df.drop('Churn', axis=1).columns
importances = rf_model.feature_importances_
feat_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(data=feat_df, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importance - Random Forest')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()
print("✅ Saved: feature_importance.png")

print("\n" + "=" * 50)
print("   PROJECT COMPLETE!")
print("=" * 50)
