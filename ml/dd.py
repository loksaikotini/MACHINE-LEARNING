import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['Species'] = iris.target  # Target values (0, 1, 2)

# Features and target
X = df.drop("Species", axis=1)
y = df["Species"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Create and train the Decision Tree
model = DecisionTreeClassifier(criterion='gini',max_depth=3, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("  Max Depth       :", model.get_params()['max_depth'])
print("  Criterion       :", model.get_params()['criterion'])

print("  Random State    :", model.get_params()['random_state'])
print("Classification Report:\n", classification_report(y_test, y_pred))


# Plot the tree
plt.figure(figsize=(10, 5))
plot_tree(model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree for Iris Dataset")
plt.show()
