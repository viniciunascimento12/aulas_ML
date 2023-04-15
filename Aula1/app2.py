# Importar as bibliotecas necessárias
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Gerar dados de exemplo para clusterização
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Criar o modelo de clusterização K-Means
model = KMeans(n_clusters=4)

# Treinar o modelo usando os dados gerados
model.fit(X)

# Visualizar os resultados
plt.scatter(X[:, 0], X[:, 1], c=model.labels_)
plt.show()
