import numpy as np
from sklearn.linear_model import LinearRegression

# Dados de entrada
X = np.array([[2]]) # Valor de 'x'
y = np.array([8]) # Resultado esperado da equação

# Treinando o modelo
reg = LinearRegression().fit(X, y)

# Testando o modelo para a equação 2x + 3 = 7
X_test = np.array([[2]])
y_pred = reg.predict(X_test)
print("Resultado da equação:", y_pred[0])