# importando as bibliotecas necessárias
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# carregando o conjunto de dados iris
iris = load_iris()

# separando os dados em conjunto de treinamento e conjunto de teste
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=0)

# criando o modelo KNN com k=1
knn = KNeighborsClassifier(n_neighbors=1)

# treinando o modelo com o conjunto de treinamento
knn.fit(X_train, y_train)

# avaliando o desempenho do modelo com o conjunto de teste
print("Acurácia do modelo com k=1: {:.2f}".format(knn.score(X_test, y_test)))
