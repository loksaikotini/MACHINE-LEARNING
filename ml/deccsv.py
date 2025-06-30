import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load CSV file
df = pd.read_csv('iris.csv')  # Replace with your path if needed

# Features and target
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

# Encode target labels (setosa → 0, versicolor → 1, virginica → 2)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

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
plot_tree(best_model,
          filled=True,
          feature_names=X.columns,
          class_names=le.classes_)
plt.title("🌳 Best Decision Tree Visualization")
plt.show()
