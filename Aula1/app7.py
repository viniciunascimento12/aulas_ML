import numpy as np
from sklearn.cluster import KMeans

# Dados de exemplo
X = np.array([[1, 2], [1, 4], [2, 2], [2, 3], [4, 1], [4, 2], [4, 4], [5, 4]])

# Criação do modelo K-means
kmeans = KMeans(n_clusters=2)  # Número de clusters desejado

# Treinamento do modelo
kmeans.fit(X)

# Obtém as etiquetas (rótulos) dos clusters e os centroides
etiquetas = kmeans.labels_
centroides = kmeans.cluster_centers_

# Impressão dos resultados
print("Etiquetas (rótulos) dos clusters: ", etiquetas)
print("Coordenadas dos centroides: ", centroides)
