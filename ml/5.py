# from sklearn.datasets import fetch_california_housing
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score
# import numpy as np


# data = fetch_california_housing()
# X = data.data
# y = data.target  
# feature_names = data.feature_names

# print("Features used for prediction:")
# for name in feature_names:
#     print("-", name)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model = LinearRegression()
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)
# print("\nModel Performance:")
# print("R² Score:", round(r2_score(y_test, y_pred), 3))

# print("\nSample Predictions (in $100,000s):")
# for i in range(3):
#     print(f"Predicted: {y_pred[i]:.2f}, Actual: {y_test[i]:.2f}")

# print("\nTry predicting for a custom house:")
# sample_input = np.array([[8.0, 41, 5.0, 1.0, 1000, 3.0, 34.0, -118.0]]) 
# predicted_price = model.predict(sample_input)[0]
# print(f"Predicted house price: ${predicted_price * 100000:.2f}")


from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data = fetch_california_housing()
X = data.data
y = data.target
feature_names = data.feature_names
for name in feature_names:
    print(name)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate model
print("Model Performance:")
print("R² Score:", round(r2_score(y_test, y_pred), 3))
print("Mean Squared Error:", round(mean_squared_error(y_test, y_pred), 3))
print("Mean Absolute Error:", round(mean_absolute_error(y_test, y_pred), 3))

# Show some predictions
print("\nSample Predictions (in $100,000s):")
for i in range(3):
    print(f"Predicted: {y_pred[i]:.2f}, Actual: {y_test[i]:.2f}")

# Predict a custom input
sample_input = np.array([[8.0, 41.0, 5.0, 1.0, 1000.0, 3.0, 34.0, -118.0]])  # Example values
predicted_price = model.predict(sample_input)[0]
print(f"\nPredicted custom house price: ${predicted_price * 100000:.2f}")
 
 
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual Price ($100,000s)')
plt.ylabel('Predicted Price ($100,000s)')
plt.title('Actual vs Predicted House Prices')
plt.grid(True)
plt.tight_layout()
plt.show()