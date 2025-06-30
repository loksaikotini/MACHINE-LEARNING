from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

# iris=load_iris()
# X=iris.data[:,[2]]
# y=iris.target
X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([40, 50, 60, 70, 80, 90])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# Create model and train
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Show predictions
# for i in range(len(X_test)):
print(f"Experience: {X_test[0][0]} years => Predicted Salary: ${y_pred[0]*1000:.2f}")
print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)

# Plot
plt.scatter(X, y, color='green', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel("Years of Experience")
plt.ylabel("Salary ($1000s)")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()