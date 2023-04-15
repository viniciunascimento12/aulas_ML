import numpy as np
from sklearn.linear_model import LinearRegression

# Dados de treinamento
X_train = np.array([[5, 3], [10, 8], [15, 7], [20, 18], [25, 22]]) # Pares de números de entrada
y_train = np.array([[2], [2], [8], [2], [3]]) # Resultados da subtração

# Criação do modelo de regressão linear
model = LinearRegression()

# Treinamento do modelo
model.fit(X_train, y_train)

# Dados de teste
X_test = np.array([[1, 4], [16, 6], [1, 8]]) # Pares de números de entrada para teste

# Fazer previsões
y_pred = model.predict(X_test)

# Impressão das previsões
print("Previsões:", y_pred)
