import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset from CSV
data = pd.read_csv("data.csv")  # Make sure this file is in the same folder or give full path

# Extract features and target
X = data[['Experience']].values  # 2D array for scikit-learn
y = data['Salary'].values        # 1D array

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Print predictions
print(f"Experience: {X_test[0][0]} years => Predicted Salary: ${y_pred[0]*1000:.2f}")
print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)

# Plotting
plt.scatter(X, y, color='green', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel("Years of Experience")
plt.ylabel("Salary ($1000s)")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()
