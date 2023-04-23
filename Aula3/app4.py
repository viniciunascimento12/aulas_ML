import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Criando um conjunto de dados simulado
n_clients = 100
n_features = 3

n_compras = np.random.randint(1, 20, size=n_clients)
valor_gasto = np.random.uniform(0, 1000, size=n_clients)
categoria = np.random.choice(["A", "B", "C"], size=n_clients)

data = np.column_stack((n_compras, valor_gasto, categoria))

# Criando um DataFrame com os dados simulados
df = pd.DataFrame(data, columns=["n_compras", "valor_gasto", "categoria"])

# Selecionando as colunas que serão usadas para agrupar os clientes
features = ["n_compras", "valor_gasto"]

# Normalizando os dados de entrada
scaler = StandardScaler()
X = scaler.fit_transform(df[features])

# Criando o modelo KMeans com 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)

# Treinando o modelo com os dados normalizados
kmeans.fit(X)

# Adicionando os rótulos dos clusters aos dados originais
df["cluster"] = kmeans.labels_

# Plotando os clusters em um gráfico de dispersão
plt.scatter(df["n_compras"], df["valor_gasto"], c=df["cluster"], cmap="viridis")
plt.xlabel("Número de compras")
plt.ylabel("Valor gasto")
plt.title("Agrupamento de clientes")
plt.show()

# Imprimindo as informações dos clusters
for cluster in df["cluster"].unique():
    print("Cluster", cluster, ":")
    print(df[df["cluster"] == cluster][features].describe())
