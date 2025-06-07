from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 產生虛擬資料
X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

# 分群模型
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# 顯示分群結果
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)
plt.title("KMeans Clustering Example")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
