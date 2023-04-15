# Importando as bibliotecas necessárias
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Dados de treinamento
avaliacoes = ["Ótimo filme!", "Gostei bastante", "Não gostei do final", "Que filme horrível", "Não recomendo"]

# Classes correspondentes
classes = ["positivo", "positivo", "negativo", "negativo", "negativo"]

# Vetorização dos dados de treinamento
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(avaliacoes)

# Divisão dos dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, classes, test_size=0.2, random_state=42)

# Criação e treinamento do modelo
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Teste do modelo com novos dados
novas_avaliacoes = ["Amei o filme!", "Excelente da atuação", "Filme ótimo", "Excelente filme"]
X_novas_avaliacoes = vectorizer.transform(novas_avaliacoes)
y_pred = clf.predict(X_novas_avaliacoes)

# Resultado da classificação das novas avaliações
print(y_pred)
