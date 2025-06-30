import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  
y = np.array([52, 60, 68, 77, 85])                
model = LinearRegression()
model.fit(x, y)

y_pred = model.predict(x)

print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)

plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='blue', label='Actual Data')
plt.plot(x, y_pred, color='red', label='Regression Line')
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Simple Linear Regression")
plt.legend()
plt.grid(True)
plt.show()
