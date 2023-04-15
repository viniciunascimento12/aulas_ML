import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dados de exemplo
areas = np.array([80, 100, 120, 140, 160, 180, 200, 220, 240, 260])
precos = np.array([400000, 500000, 600000, 700000, 800000, 900000, 1000000, 1100000, 1200000, 1300000])

# Criar modelo de regressão linear
reg = LinearRegression()

# Ajustar o modelo aos dados
reg.fit(areas.reshape(-1, 1), precos)

# Prever o preço para uma nova área
nova_area = np.array([500]).reshape(-1, 1)
preco_predito = reg.predict(nova_area)

# Plotar os dados e a linha de regressão
plt.scatter(areas, precos)
plt.plot(areas, reg.predict(areas.reshape(-1, 1)), color='red')
plt.xlabel('Área')
plt.ylabel('Preço')
plt.title('Regressão Linear')
plt.show()

print(f'O preço para uma área é de R${preco_predito[0]:,.2f}')
