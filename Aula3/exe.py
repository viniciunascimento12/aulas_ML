import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Gerando alguns dados de exemplo aleatórios
X = np.random.rand(100, 2)

# Criação do modelo KMeans com 3 clusters
kmeans = KMeans(n_clusters=4)

# Treinamento do modelo
kmeans.fit(X)

# Obtendo os rótulos das amostras
labels = kmeans.labels_

# Obtendo os centróides dos clusters
centroids = kmeans.cluster_centers_

# Plotando os dados e os clusters
plt.scatter(X[:,0], X[:,1], c=labels)
plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=200, linewidths=3, color='r')
plt.show()
