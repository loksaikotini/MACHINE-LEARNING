import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load CSV data
df = pd.read_csv('iris.csv')  # Replace with your file path

# Select only sepal length and sepal width for clustering
X = df[['sepal_length', 'sepal_width']].values

# Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Get labels and centroids
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Plot results
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow', label='Data Points')
plt.scatter(centers[:, 0], centers[:, 1], color='black', marker='x', s=100, label='Centroids')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('K-Means Clustering from CSV (Iris Dataset)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
