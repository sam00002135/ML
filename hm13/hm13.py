import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 1. Generate some sample data
X, y_true = make_blobs(n_samples=300, centers=3,
                       cluster_std=0.60, random_state=0)

# 2. Apply K-Means
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# 3. Show the clusters
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

# 4. Show the cluster centers
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red',
            s=200, alpha=0.75, marker='X')
plt.title('K-Means Clustering')
plt.show()
