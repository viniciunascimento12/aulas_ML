# Importar as bibliotecas necessárias
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Definir os dados de treinamento e teste
X_train = ["O filme é ótimo", "Não gostei do filme", "A trama é intrigante", "A atuação é fraca"]
y_train = ["positivo", "negativo", "positivo", "negativo"]
X_test = ["melhor filme que já vi", "recomendo o filme"]

# Criar o vetorizador de palavras
vectorizer = CountVectorizer()

# Transformar os dados de treinamento e teste em vetores de palavras
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Criar o modelo de classificação Naive Bayes
model = MultinomialNB()

# Treinar o modelo usando os dados de treinamento
model.fit(X_train_vec, y_train)

# Fazer uma previsão com o modelo treinado
y_pred = model.predict(X_test_vec)

# Imprimir as previsões
print(y_pred)
