import pandas as pd
import matplotlib.pyplot as plt
print("Using Pandas to create and manipulate a dataset")

data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [10, 20, 30],
    'Score': [33,28, 49]
}

df = pd.DataFrame(data)
print("\n1. DataFrame:\n", df)


# Access column
print("\n3. Scores column:\n", df['Score'])

# Filter rows
print("\n4. Age > 25:\n", df[df['Age'] > 25])
print("\n📘 Using Matplotlib to visualize the data")

# Line Plot
plt.figure(figsize=(6, 4))
plt.plot(df['Age'], df['Score'], marker='o', color='blue')
plt.title("Age vs Score")
plt.xlabel("Age")
plt.ylabel("Score")
plt.grid(True)
plt.tight_layout()
plt.show()

# Bar Chart
plt.figure(figsize=(6, 4))
plt.bar(df['Name'], df['Score'], color='orange')
plt.title("Scores of Students")
plt.xlabel("Name")
plt.ylabel("Score")
plt.grid(True)
plt.tight_layout()
plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import fetch_california_housing
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, mean_squared_error

# # Load the California Housing dataset
# data = fetch_california_housing()
# X = data.data
# y = data.target
# feature_names = data.feature_names

# # Display the feature names used in the model
# print("Features used for prediction:")
# for name in feature_names:
#     print("-", name)

# # Split dataset into training and testing sets (80% train, 20% test)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create and train the Linear Regression model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Predict values using the test data
# y_pred = model.predict(X_test)

# # Evaluate model performance
# print("\nModel Performance:")
# print("R² Score:", round(r2_score(y_test, y_pred), 3))
# print("Mean Squared Error:", round(mean_squared_error(y_test, y_pred), 3))

# # Show sample predictions
# print("\nSample Predictions (in $100,000s):")
# for i in range(3):
#     print(f"Predicted: {y_pred[i]:.2f}, Actual: {y_test[i]:.2f}")

# # Predict a custom house input
# print("\nTry predicting for a custom house:")
# sample_input = np.array([[8.0, 41, 5.0, 1.0, 1000, 3.0, 34.0, -118.0]])  # Feature order matches dataset
# predicted_price = model.predict(sample_input)[0]
# print(f"Predicted house price: ${predicted_price * 100000:.2f}")

# # ----------------------------------------
# # 📊 Simple Plot: Actual vs Predicted Prices
# # ----------------------------------------
# plt.scatter(y_test, y_pred, color='skyblue', edgecolor='black')
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
# plt.xlabel('Actual Prices ($100,000s)')
# plt.ylabel('Predicted Prices ($100,000s)')
# plt.title('Actual vs Predicted House Prices')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

