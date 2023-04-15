
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Carregando o conjunto de dados de imagens
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Normalizando as imagens
X_train = X_train / 255.0
X_test = X_test / 255.0

# Criando o modelo de classificação de imagem
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compilando o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treinando o modelo
model.fit(X_train, y_train, epochs=5)

# Avaliando o modelo
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Acurácia do modelo:', test_acc)

# Fazendo previsões
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

# Mostrando exemplos de previsões corretas e incorretas
for i in range(10):
    print("Previsão:", predicted_labels[i])
    print("Rótulo verdadeiro:", y_test[i])
