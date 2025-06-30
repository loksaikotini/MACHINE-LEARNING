import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load CSV data
df = pd.read_csv('iris.csv')  # Replace with your path if needed

# Prepare features and binary target
X = df[['petal_length']].values  # Using petal length only
y = (df['species'] != 'setosa').astype(int)  # Binary: 0 for setosa, 1 for others

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)

# Predict test set
y_pred = log_reg.predict(x_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of the model: {accuracy * 100:.2f}%\n')
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Predict on new custom petal lengths
new_data = np.array([[1.5], [4.5], [6.5]])
new_pred = log_reg.predict(new_data)

for i, val in enumerate(new_data):
    print(f'Petal Length: {val[0]} cm => Prediction: {"Not Setosa" if new_pred[i]==1 else "Setosa"}')

# Plotting
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, log_reg.predict(X), color='red', label='Logistic Regression Line')

# Custom points
plt.scatter(new_data[:, 0], new_pred, color='green', marker='x', s=100, label='New Data Points')

plt.xlabel('Petal Length (cm)')
plt.ylabel('Class (0 = Setosa, 1 = Not Setosa)')
plt.title('Logistic Regression: Petal Length vs Iris Class')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
