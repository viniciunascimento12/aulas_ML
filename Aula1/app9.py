from sklearn.linear_model import LinearRegression

# Dados de exemplo (características e preços das casas)
X = [[1, 140], [2, 160], [3, 180], [4, 200], [5, 220]] # Características das casas (número de quartos e área do terreno)
y = [250, 300, 350, 400, 450] # Preços das casas

# Criação do modelo de regressão linear
modelo = LinearRegression()

# Treinamento do modelo
modelo.fit(X, y)

# Predição do preço de uma casa com 6 quartos e área do terreno de 240 m2
caracteristicas_casa_nova = [[6, 240]]
preco_predito = modelo.predict(caracteristicas_casa_nova)

# Impressão do resultado
print(f"O preço estimado é: R$ {preco_predito[0]:.2f}")
