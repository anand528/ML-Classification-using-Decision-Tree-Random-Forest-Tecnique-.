# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load dataset
data = pd.read_csv('titanic.csv')  # Replace with actual path

# Basic preprocessing: drop rows with missing values (for simplicity)
data = data[['Pclass', 'Sex', 'Age', 'Fare', 'Survived']].dropna()

# Convert categorical 'Sex' to numeric
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# Features and target
X = data[['Pclass', 'Sex', 'Age', 'Fare']]
y = data['Survived']

# Step 2: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Fit Logistic Regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Step 4: Evaluate the model
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# ROC-AUC
roc_score = roc_auc_score(y_test, y_prob)
print(f'ROC-AUC Score: {roc_score:.4f}')

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_score:.2f})')
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid()
plt.show()

# Step 5: Tune threshold and explain sigmoid
threshold = 0.6  # Custom threshold
y_pred_custom = (y_prob >= threshold).astype(int)

print(f"Confusion Matrix with threshold = {threshold}")
print(confusion_matrix(y_test, y_pred_custom))

# Sigmoid function (for reference)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z_example = np.linspace(-10, 10, 100)
plt.plot(z_example, sigmoid(z_example))
plt.title('Sigmoid Function')
plt.xlabel('z')
plt.ylabel('sigmoid(z)')
plt.grid()
plt.show()
