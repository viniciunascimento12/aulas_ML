
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Dados de exemplo dos pacientes
idade = [45, 62, 32, 47, 71, 52, 60, 39, 46, 54]
pressao_sanguinea = [140, 115, 122, 128, 134, 120, 145, 130, 126, 138]
frequencia_cardiaca = [75, 83, 92, 70, 60, 65, 80, 85, 88, 95]
nivel_de_acucar = [0, 1, 1, 0, 1, 1, 0, 0, 1, 0]
colesterol = [233, 270, 157, 235, 246, 234, 300, 280, 223, 220]
ecg = [0, 1, 0, 1, 0, 2, 2, 1, 1, 0]
pico_exercicio = [0, 1, 1, 0, 0, 0, 1, 2, 2, 0]
st_depressao = [2.3, 1.5, 0.6, 0.0, 0.0, 0.1, 1.4, 0.2, 0.0, 0.0]
thal = [3, 2, 1, 2, 3, 1, 2, 1, 3, 2]

# Juntando as características dos pacientes em uma matriz
X = np.array([idade, pressao_sanguinea, frequencia_cardiaca, nivel_de_acucar, colesterol, ecg, pico_exercicio, st_depressao, thal]).T
# Normalizando os dados
scaler = StandardScaler()
X = scaler.fit_transform(X)
# Criando o modelo KMeans com 5 clusters
kmeans = KMeans(n_clusters=5, random_state=42)
# Treinando o modelo com os dados normalizados
kmeans.fit(X)
# Adicionando os rótulos dos clusters aos dados originais
cluster_labels = kmeans.labels_

# Plotando um gráfico dos clusters
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis')
plt.xlabel('Idade')
plt.ylabel('Pressão Sanguínea')
plt.title('Clusters de Pacientes')
plt.show()
# Imprimindo as informações dos clusters
for cluster in np.unique(cluster_labels):
    print("Cluster", cluster, ":")
    cluster_data = X[cluster_labels == cluster]
    print(np.mean(cluster_data, axis=0))
