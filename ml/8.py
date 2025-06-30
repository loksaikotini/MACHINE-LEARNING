# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, confusion_matrix

# # Load dataset
# iris = load_iris()
# X = iris.data
# y = iris.target

# print("Feature names:", iris.feature_names)
# print("Target classes:", iris.target_names)

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create and train the model
# model = LogisticRegression(max_iter=200)
# model.fit(X_train, y_train)

# # Predict and evaluate
# y_pred = model.predict(X_test)
# print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))

# # Predict on a new sample
# sample = [[5.1, 3.5, 1.4, 0.2]]  # Just an example
# predicted_class = model.predict(sample)
# print("\nPredicted class for sample {}: {}".format(sample[0], iris.target_names[predicted_class[0]]))

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, confusion_matrix

# # Load Iris dataset
# iris = load_iris()
# X = iris.data[:, 2:]  # Use only first 2 features for 2D plotting
# y = iris.target
# class_names = iris.target_names

# # Show feature and class names
# print("Feature names (used for plotting):", iris.feature_names[:2])
# print("Target classes:", class_names)

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train Logistic Regression model
# model = LogisticRegression(max_iter=200)
# model.fit(X_train, y_train)

# # Predictions
# y_pred = model.predict(X_test)

# # Evaluation
# print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))

# # Predict a custom sample (ensure shape is correct for 2D input)
# sample = np.array([[5.1, 3.5]])  # only 2 features used here
# predicted_class = model.predict(sample)[0]
# print("\nPredicted class for sample {}: {}".format(sample[0], class_names[predicted_class]))

# # -------------------------------------
# # 📊 Plotting test set predictions
# # -------------------------------------
# colors = ['red', 'green', 'blue']
# plt.figure(figsize=(8, 6))

# # Plot test points colored by predicted class
# for i, color in enumerate(colors):
#     idx = np.where(y_pred == i)
#     plt.scatter(X_test[idx, 0], X_test[idx, 1], label=class_names[i], color=color, edgecolor='black', s=80)

# # Plot the new predicted sample
# plt.scatter(sample[0, 0], sample[0, 1], color='yellow', edgecolor='black', label='New Sample', marker='X', s=150)

# plt.xlabel("Sepal Length (cm)")
# plt.ylabel("Sepal Width (cm)")
# plt.title("Logistic Regression - Iris Classification (2D)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()



import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
iris = load_iris()
X = iris.data[:, [2]]  # Petal length (1 feature for plotting)
y = (iris.target != 0).astype(int)  # Setosa = 0, Not Setosa = 1

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
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
new_data = np.array([[1.5], [4.5], [6.5]])  # Petal lengths
new_pred = log_reg.predict(new_data)

for i, val in enumerate(new_data):
    print(f'Petal Length: {val[0]} cm => Prediction: {"Not Setosa" if new_pred[i]==1 else "Setosa"}')

# Plotting
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, log_reg.predict(X), color='red', label='Logistic Regression Line')

# Custom new data points
plt.scatter(new_data[:, 0], new_pred, color='green', marker='x', s=100, label='New Data Points')

plt.xlabel('Petal Length (cm)')
plt.ylabel('Class (0 = Setosa, 1 = Not Setosa)')
plt.title('Logistic Regression: Petal Length vs Iris Class')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
