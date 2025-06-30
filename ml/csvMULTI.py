import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


df = pd.read_csv(r"C:\Users\Sai Nishitha\Desktop\nishi\3-2\ml\housing.csv")
df = df.dropna()

print("Columns in your dataset:")
print(df.columns.tolist())
df = df.drop('ocean_proximity', axis=1)

X = df.drop('median_house_value', axis=1)
y = df['median_house_value']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("\nModel Performance:")
print("R² Score:", round(r2_score(y_test, y_pred), 3))
print("Mean Squared Error:", round(mean_squared_error(y_test, y_pred), 3))
print("Mean Absolute Error:", round(mean_absolute_error(y_test, y_pred), 3))

# Sample predictions
print("\nSample Predictions:")
for i in range(3):
    print(f"Predicted: {y_pred[i]:.2f}, Actual: {y_test.iloc[i]:.2f}")

# Predict from custom input
print("\nEnter custom values for prediction:")
feature_values = []
for col in X.columns:
    val = float(input(f"{col}: "))
    feature_values.append(val)

sample_input = np.array([feature_values])
predicted_price = model.predict(sample_input)[0]
print(f"\nPredicted Price: ${predicted_price:.2f}")
