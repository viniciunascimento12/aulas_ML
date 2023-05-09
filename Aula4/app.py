import numpy as np
from sklearn.linear_model import LinearRegression

# Gerar dados sintéticos
X = np.array([[1000, 3], [2000, 4], [3000, 5], [4000, 6], [5000, 7]])
y = np.array([50000, 80000, 110000, 140000, 170000])


# Criar modelo de regressão linear
model = LinearRegression()

# Treinar modelo nos dados de treinamento
model.fit(X, y)

# Fazer previsões em novos dados
X_new = np.array([[6000, 8], [7000, 9]])
y_pred = model.predict(X_new)

# Exibir previsões
print(y_pred)

