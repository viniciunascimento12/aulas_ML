from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Gerando um conjunto de dados sintético com 1000 amostras e 10 variáveis independentes
X, y = make_regression(n_samples=1000, n_features=10, random_state=42)

# Dividindo o conjunto de dados em treino e teste (70% treino, 30% teste)
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=42)

# Criando um modelo de regressão linear
modelo = LinearRegression()

# Treinando o modelo com o conjunto de treino
modelo.fit(X_treino, y_treino)

# Avaliando o desempenho do modelo com o conjunto de teste
previsoes = modelo.predict(X_teste)
erro = mean_squared_error(y_teste, previsoes)
print('Erro quadrático médio:', erro)

