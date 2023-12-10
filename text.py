import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import class_weight

# Carregando dados de treinamento e teste
train = pd.read_csv('ReutersGrain-train.csv', escapechar='\\', quotechar="'")
test = pd.read_csv('ReutersGrain-test.csv', escapechar='\\', quotechar="'")

# Pré-processamento da base
test['Text'] = test['Text'].str.replace('\n', ' ').str.lower()
train['Text'] = train['Text'].str.replace('\n', ' ').str.lower()

X_train, y_train = train['Text'], train['class-att']
X_test, y_test = test['Text'], test['class-att']

# Vetorização usando TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Calcular pesos das classes
class_weights = class_weight.compute_class_weight('balanced', classes=pd.unique(y_train), y=y_train)

# Instanciar o classificador Naive Bayes com pesos de classe
classifier = MultinomialNB(class_prior=class_weights)

# Treinar o classificador Naive Bayes
classifier.fit(X_train_tfidf, y_train)

# Predições no conjunto de teste
y_pred = classifier.predict(X_test_tfidf)

# Avaliar o desempenho do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Exibir relatório de classificação
print('Classification Report:\n', classification_report(y_test, y_pred))

