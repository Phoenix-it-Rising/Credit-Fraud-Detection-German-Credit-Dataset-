# Title: Credit Fraud Detection using XGBoost
# Author: Kiana Lang
# Course: CS492 - Machine Learning
# Date: 2025-09-21
# Description: This script uses the german_cleaned.csv dataset to train an XGBoost model for predicting credit fraud. It includes data preprocessing, model training, evaluation, and visualization.
# Resources/References:
# - https://xgboost.readthedocs.io/
# - https://scikit-learn.org/stable/
# - https://peps.python.org/pep-0008/
# - https://eng.uber.com/transforming-financial-forecasting-machine-learning/
# - https://coloradotech.idm.oclc.org/login?qurl=https%3a%2f%2fsearch.ebscohost.com%2flogin.aspx%3fdirect%3dtrue%26db%3dmnh%26AN%3d33267817%26site%3dehost-live%26scope%3dsite%26custid%3dns214714
# - https://coloradotech.idm.oclc.org/login?url=https://www.proquest.com/scholarly-journals/long-term-forecasting-electrical-loads-kuwait/docview/2434995470/se-2?accountid=144789
# - https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
# - https://www.python.org/dev/peps/pep-0008/

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# -----------------------------
# Step 1: Load the dataset
# -----------------------------
data = pd.read_csv("german_cleaned.csv")

# Display basic information
print("Dataset shape:", data.shape)
print("First 5 rows:")
print(data.head())

# -----------------------------
# Step 2: Preprocess the data
# -----------------------------
# Check for missing values
print("\nMissing values per column:")
print(data.isnull().sum())

# Remap target labels for binary classification
# Original: 1 = non-fraud, 2 = fraud
# New: 0 = non-fraud, 1 = fraud
data['Target'] = data['Target'].map({1: 0, 2: 1})

# Separate features and target
X = data.drop(columns=['Target'])
y = data['Target']

# -----------------------------
# Step 3: Split the dataset
# -----------------------------
# Split into training (70%), validation (15%), and test (15%) sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print("\nTraining set size:", X_train.shape)
print("Validation set size:", X_val.shape)
print("Test set size:", X_test.shape)

# -----------------------------
# Step 4: Train the XGBoost model
# -----------------------------
# Initialize the classifier
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Fit the model to training data
xgb_model.fit(X_train, y_train)

# -----------------------------
# Step 5: Evaluate on validation set
# -----------------------------
# Predict on validation data
y_val_pred = xgb_model.predict(X_val)

# Calculate accuracy
val_accuracy = accuracy_score(y_val, y_val_pred)
print("\nValidation Accuracy:", val_accuracy)

# Generate classification report
val_report = classification_report(y_val, y_val_pred)
print("\nClassification Report (Validation):\n", val_report)

# Confusion matrix
val_conf_matrix = confusion_matrix(y_val, y_val_pred)

# Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(val_conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Validation Set')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig("confusion_matrix_validation.png")
plt.close()

# -----------------------------
# Step 6: Final evaluation on test set
# -----------------------------
# Predict on test data
y_test_pred = xgb_model.predict(X_test)

# Calculate accuracy
test_accuracy = accuracy_score(y_test, y_test_pred)
print("\nTest Accuracy:", test_accuracy)

# Generate classification report
test_report = classification_report(y_test, y_test_pred)
print("\nClassification Report (Test):\n", test_report)
