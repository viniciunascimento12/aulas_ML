# Importar as bibliotecas necessárias
import numpy as np
from sklearn.linear_model import LinearRegression

# Definir os dados de treinamento
X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
y_train = np.array([3, 7, 11, 15, 19])

# Criar o modelo de regressão linear
model = LinearRegression()

# Treinar o modelo usando os dados de treinamento
model.fit(X_train, y_train)

# Fazer uma previsão com o modelo treinado
X_test = np.array([[100, 100], [2, 2], [125, 5]])
y_pred = model.predict(X_test)

# Imprimir as previsões
print(y_pred)