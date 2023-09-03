#árvore com todos os atributos alterados para nominais não ordinais.
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt

base = pd.read_csv('restaurante.csv', sep=',', encoding='utf-8') 

line = base.shape[0]
column = base.shape[1]

#contando linhas e colunas
print(f"linhas: {line} colunas: {column}")

#Instancia,Alternativo,Bar,Sex/Sab,Fome,Cliente,Preco,Chuva,Res,Tipo,Tempo,Conclusao
#só funciona se igualar duas variáveis
base['Conclusao'], _= pd.factorize(base['Conclusao'])
base['Alternativo'], _ = pd.factorize(base['Alternativo'])
base['Bar'], _ = pd.factorize(base['Bar'])
base['Sex/Sab'], _ = pd.factorize(base['Sex/Sab'])
base['Fome'], _ = pd.factorize(base['Fome'])
base['Cliente'], _ = pd.factorize(base['Cliente'])
base['Preco'], _ = pd.factorize(base['Preco'])
base['Chuva'], _ = pd.factorize(base['Chuva'])
base['Res'], _ = pd.factorize(base['Res'])
base['Tipo'], _ = pd.factorize(base['Tipo'])
base['Tempo'], _ = pd.factorize(base['Tempo'])
base['Conclusao'], _ = pd.factorize(base['Conclusao'])


#testando se todos os atributos foram convertidos
print(base)

x_prev = base.iloc[:, 1:11].values
y_classe = base.iloc[:, 11].values

#base de dados dividida entre teste e treino
X_train, X_teste, Y_train, Y_teste = train_test_split(x_prev, y_classe, test_size = 0.30, random_state = 25)

#clacificacao da árvore de decisão
#calculo de entropia e dos niveis da arvore
classifier = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)

#modela um classificador de decisao dos treinos x_train e y_train
modelo = classifier.fit(X_train, Y_train)

#testa o modelo
previsao = modelo.predict(X_teste)

print(accuracy_score(Y_teste, previsao))

confusion_matrix(Y_teste, previsao)

cm = ConfusionMatrix(modelo)
cm.fit(X_train, Y_train)
cm.score(X_teste, Y_teste)
cm.show()

#avore simples
print(classification_report(Y_teste, previsao))
tree.plot_tree(modelo)
plt.show()

previsores = ['Frances', 'Hamburguer', 'Italiano', 'Tailandes', 'Alternativo', 'Bar', 'SextaSabado', 'Fome', 'Cliente', 'Preco','Chuva','Res','Tipo', 'Tempo']
figura, eixos = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
tree.plot_tree(classifier, feature_names=previsores, class_names=[str(cls) for cls in classifier.classes_.tolist()], filled=True)




