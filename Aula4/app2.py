import numpy as np
from sklearn.linear_model import LinearRegression

# Gerar dados sintéticos
X = np.array([[1000], [2000], [3000], [4000], [5000]])
y = np.array([50000, 80000, 110000, 140000, 170000])


# Criar modelo de regressão linear com apenas um coeficiente
model = LinearRegression()

# Treinar modelo nos dados de treinamento
model.fit(X, y)

# Fazer previsões em novos dados
X_new = np.array([[6000]])
y_pred = model.predict(X_new)

# Exibir previsões
print(y_pred)
