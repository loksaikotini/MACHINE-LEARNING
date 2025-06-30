import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model and parameter grid
model = DecisionTreeClassifier(random_state=42)
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 3, 4, 5],
    'min_samples_split': [2, 3]
}

# Grid search
grid = GridSearchCV(model, param_grid, cv=3)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# Predict and evaluate
y_pred = best_model.predict(X_test)
print("✅ Best Parameters:", grid.best_params_)
print("🎯 Accuracy:", round(accuracy_score(y_test, y_pred), 2))

# Plot the decision tree
plt.figure(figsize=(12, 6))
plot_tree(best_model, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.title("🌳 Best Decision Tree Visualization")
plt.show()
