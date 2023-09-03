import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier

base = pd.read_csv('restaurante.csv', usecols=['Alternativo','Bar','Sex/Sab','Fome','Cliente','Preco','Chuva','Res','Tipo','Tempo','Conclusao'])

base

np.unique(base['Conclusao'], return_counts=True)

sns.countplot(x = base['Conclusao'])

X_prev = base.iloc[:, 0:10].values

X_prev

X_prev[:,5]

y_classe = base.iloc[:, 10].values

y_classe

from sklearn.preprocessing import LabelEncoder

X_prev[:,5]

X_prev

lb = LabelEncoder()

X_prev[:,0] = lb.fit_transform(X_prev[:,0])
X_prev[:,1] = lb.fit_transform(X_prev[:,1])
X_prev[:,2] = lb.fit_transform(X_prev[:,2])
X_prev[:,3] = lb.fit_transform(X_prev[:,3])
X_prev[:,4] = lb.fit_transform(X_prev[:,4])
X_prev[:,5] = lb.fit_transform(X_prev[:,5])
X_prev[:,6] = lb.fit_transform(X_prev[:,6])
X_prev[:,7] = lb.fit_transform(X_prev[:,7])
X_prev[:,9] = lb.fit_transform(X_prev[:,9])

X_prev

len(np.unique(base['Cliente']))

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

X_prev

X_prev[:,0:9]

onehotencoder_restaurante = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [8])], remainder='passthrough')

X_prev= onehotencoder_restaurante.fit_transform(X_prev)

X_prev

X_prev.shape

from sklearn.model_selection import train_test_split

X_prev

y_classe.shape

X_treino, X_teste, y_treino, y_teste = train_test_split(X_prev, y_classe, test_size = 0.20, random_state = 23)

X_treino.shape

X_teste.shape

modelo = DecisionTreeClassifier(criterion='entropy')
Y = modelo.fit(X_treino, y_treino)

previsoes = modelo.predict(X_teste)

y_teste

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(accuracy_score(y_teste,previsoes))

from yellowbrick.classifier import ConfusionMatrix
confusion_matrix(y_teste, previsoes)

cm = ConfusionMatrix(modelo)
cm.fit(X_treino, y_treino)
cm.score(X_teste, y_teste)
cm.show()

print(classification_report(y_teste, previsoes))

from sklearn import tree
tree.plot_tree(Y)
plt.show()

from sklearn import tree
previsores = ['Frances', 'Hamburguer', 'Italiano', 'Tailandes', 'Alternativo', 'Bar', 'SextaSabado', 'Fome', 'Cliente', 'Preco','Chuva','Res', 'Tempo']
figura, eixos = plt.subplots(nrows=1, ncols=1, figsize=(13,13))
tree.plot_tree(modelo, feature_names=previsores, filled=True);

