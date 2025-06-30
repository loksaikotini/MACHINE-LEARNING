#HAS PLOT WITHOUT CENTROIDS

# from sklearn.cluster import KMeans
# from sklearn.datasets import load_iris
# import matplotlib.pyplot as plt

# # Load data
# iris = load_iris()
# X = iris.data

# # Create KMeans model with 3 clusters
# kmeans = KMeans(n_clusters=3, random_state=42)

# # Fit model
# kmeans.fit(X)

# # Get cluster labels
# labels = kmeans.labels_

# # Plot first 2 features with cluster coloring
# plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title('K-Means Clustering')
# plt.show()


#HAS PLOT WITH CENTROIDS

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load Iris dataset
iris = load_iris()
X = iris.data[:, :2]  # Only use sepal length (0) and sepal width (1)

# Create and fit KMeans model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Get cluster labels and centers
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Plot
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow')
plt.scatter(centers[:, 0], centers[:, 1], color='black', marker='x', s=100, label='Centroids')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('K-Means Clustering (Iris Dataset)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
