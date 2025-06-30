import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the dataset (use only 2 features for 2D plotting)
iris = load_iris()
X = iris.data[:, 2:]  # Sepal length and sepal width
y = iris.target

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Evaluate on test data
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy on test set: {accuracy:.2f}")

# Predict for a new sample point
new_point = np.array([[5.8,3.8]])
predicted_class = knn.predict(new_point)[0]
print(f"Predicted class for point {new_point[0]}: {iris.target_names[predicted_class]}")

# Plotting
plt.figure(figsize=(8, 6))

# Plot the training data
colors = ['red', 'green', 'blue']
for i in range(3):
    plt.scatter(X_train[y_train == i, 0], X_train[y_train == i, 1], 
                color=colors[i], label=iris.target_names[i], edgecolor='k')

# Plot the new point
plt.scatter(new_point[0][0], new_point[0][1], marker='X', s=200, 
            color='black', label='New Point')

# Labels and legend
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')
plt.title('KNN Classification (k=3)')
plt.legend()
plt.grid(True)
plt.show()