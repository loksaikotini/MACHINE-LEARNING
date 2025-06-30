import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset from CSV
df = pd.read_csv('iris.csv')  # 👈 Replace with your file path if needed

# Select only petal length and petal width for 2D plotting
X = df[['petal_length', 'petal_width']].values
y = df['species']

# Encode target labels (setosa → 0, versicolor → 1, virginica → 2)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Evaluate on test data
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy on test set: {accuracy:.2f}")

# Predict for a new sample point
new_point = np.array([[5.8, 3.8]])
predicted_class = knn.predict(new_point)[0]
print(f"Predicted class for point {new_point[0]}: {le.inverse_transform([predicted_class])[0]}")

# Plotting
plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue']

# Plot the training data
for i in range(3):
    plt.scatter(X_train[y_train == i, 0], X_train[y_train == i, 1],
                color=colors[i], label=le.inverse_transform([i])[0], edgecolor='k')

# Plot the new point
plt.scatter(new_point[0][0], new_point[0][1], marker='X', s=200,
            color='black', label='New Point')

# Labels and legend
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('KNN Classification (k=3)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
