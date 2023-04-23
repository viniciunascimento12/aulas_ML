import numpy as np
from sklearn.cluster import KMeans

# Criando os dados de entrada
X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])

# Criando o modelo KMeans
model = KMeans(n_clusters=2, random_state=0)

# Treinando o modelo com os dados de entrada
model.fit(X)

# Obtendo os rótulos para cada exemplo de entrada
labels = model.labels_

# Obtendo as coordenadas dos centróides dos grupos
centroids = model.cluster_centers_

# Imprimindo os rótulos e os centróides
print("Rótulos:", labels)
print("Centróides:", centroids)
