from sklearn.linear_model import LinearRegression
import numpy as np

# criar dados de treinamento
X_train = np.random.uniform(-5, 5, size=(10,1))
y_train = 2*X_train + 1

# criar e treinar modelo
reg = LinearRegression()
reg.fit(X_train, y_train)

# fazer previsões
X_test = np.array([[3], [5], [-2], [0]])
y_pred = reg.predict(X_test)

# imprimir previsões
print(y_pred)