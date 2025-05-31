import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    classification_report
)

# 1. Load your CSV data
df = pd.read_csv("data.csv")

# 2. Select features and target (update this)
X = df.drop(columns=["Team 1_Bermuda"])  # Replace 'target_column' with your actual target name
y = df["Team 1_Bermuda"]

# 3. Train/test split and standardize
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# 4. Fit Logistic Regression model
model = LogisticRegression()
model.fit(X_train_std, y_train)

# 5. Evaluate model
y_pred = model.predict(X_test_std)
y_prob = model.predict_proba(X_test_std)[:, 1]

# Confusion matrix and metrics
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label="Logistic Regression (AUC = {:.2f})".format(roc_auc_score(y_test, y_prob)))
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

# 6. Tune threshold
custom_threshold = 0.3
y_custom = (y_prob >= custom_threshold).astype(int)
print("\nCustom Threshold (0.3) Confusion Matrix:\n", confusion_matrix(y_test, y_custom))
print("Precision at 0.3:", precision_score(y_test, y_custom))
print("Recall at 0.3:", recall_score(y_test, y_custom))

# 7. Sigmoid function explanation
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace(-10, 10, 100)
sig = sigmoid(z)

plt.plot(z, sig)
plt.title("Sigmoid Function")
plt.xlabel("z")
plt.ylabel("sigmoid(z)")
plt.grid()
plt.show()
