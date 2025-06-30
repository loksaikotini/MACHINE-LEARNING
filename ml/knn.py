from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
import numpy as np

# Step 1: Load dataset
iris = load_iris()
X = iris.data      # features
y = iris.target    # labels

# Step 2: Split dataset into training and testing (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Create KNN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=5)

# Step 4: Train the classifier
knn.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = knn.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy:{accuracy*100:.2f}%")
new_data=np.array([[3.2,7.3,6.5,0.1]])
new_pred=knn.predict(new_data)
classname=iris.target_names
print(f"Preicted classname for new_data:{classname[new_pred[0]]}")
# Step 6: Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

