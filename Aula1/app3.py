import numpy as np  # Importa a biblioteca NumPy para operações numéricas
from sklearn.linear_model import LinearRegression  # Importa o modelo de regressão linear do scikit-learn

# Passo 1: Preparação dos dados
# Cria um array de valores de entrada (variável independente) X com 10 amostras de 0 a 9
X = np.array([i for i in range(10)]).reshape((-1, 1))  
# Cria um array de valores de saída (variável dependente) Y correspondentes a X adicionados de ruído aleatório
Y = np.array([2*i + np.random.normal(0, 1) for i in range(10)]).reshape((-1, 1))  

# Passo 2: Criação e treinamento do modelo
model = LinearRegression()  # Cria uma instância do modelo de regressão linear
model.fit(X, Y)  # Treina o modelo com os dados de entrada X e saída Y

# Passo 3: Previsões e avaliação do modelo
# Faz uma previsão para um novo valor de entrada X_test
X_test = np.array([[11]])  
Y_pred = model.predict(X_test)  # Calcula a previsão do modelo para X_test
print("Previsão para X_test:", Y_pred)  # Imprime a previsão do modelo

# Passo 4: Visualização dos resultados
import matplotlib.pyplot as plt  # Importa a biblioteca Matplotlib para visualização de gráficos
# Plota os dados de entrada X e saída Y
plt.scatter(X, Y, color='blue', label='Dados de treinamento')
# Plota a reta de regressão encontrada pelo modelo
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regressão linear')
plt.scatter(X_test, Y_pred, color='green', label='Previsão para X_test')
plt.legend()  # Adiciona uma legenda ao gráfico
plt.xlabel('X')  # Adiciona rótulo ao eixo x
plt.ylabel('Y')  # Adiciona rótulo ao eixo y
plt.show()  # Mostra o gráfico
