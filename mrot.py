import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import seaborn as sns
import re

# Carregando os conjuntos de treino e teste
treino_df = pd.read_csv('/Users/caiolima/Documents/inteligência artificial/Lista EXTRA 3 - Mineração de textos/jigsaw-toxic-comment-classification-challenge/train.csv')
teste_df = pd.read_csv('/Users/caiolima/Documents/inteligência artificial/Lista EXTRA 3 - Mineração de textos/jigsaw-toxic-comment-classification-challenge/test.csv')

# Exibindo uma amostra aleatória de 5 linhas do conjunto de treino
treino_df.sample(5)

# Colunas alvo (rótulos) que queremos prever
cols_alvo = ['obscene','insult','toxic','severe_toxic','identity_hate','threat']

# Verificando valores ausentes em colunas numéricas
treino_df.describe()

# Comentários sem nenhum rótulo (sem toxicidade)
sem_rotulo_em_todos = treino_df[(treino_df['toxic']!=1) & (treino_df['severe_toxic']!=1) & (treino_df['obscene']!=1) & 
                            (treino_df['threat']!=1) & (treino_df['insult']!=1) & (treino_df['identity_hate']!=1)]
print('Porcentagem de comentários sem rótulos é ', len(sem_rotulo_em_todos)/len(treino_df)*100)

# Verificando se há algum comentário nulo no conjunto de treino
sem_comentario = treino_df[treino_df['comment_text'].isnull()]
len(sem_comentario)

# Exibindo as primeiras linhas do conjunto de teste
teste_df.head()

# Verificando se há algum comentário nulo no conjunto de teste
sem_comentario_teste = teste_df[teste_df['comment_text'].isnull()]
sem_comentario_teste

# Exibindo o total de linhas nos conjuntos de treino e teste, e os números para as diferentes categorias (rótulos)
print('Total de linhas no teste é {}'.format(len(teste_df)))
print('Total de linhas no treino é {}'.format(len(treino_df)))
print(treino_df[cols_alvo].sum())

# Analisando o comprimento dos comentários no conjunto de treino e plotando um histograma
treino_df['comprimento'] = treino_df['comment_text'].apply(lambda x: len(str(x)))
sns.set()
treino_df['comprimento'].hist()
plt.show()

# Analisando a correlação entre as features e os rótulos no conjunto de treino
data = treino_df[cols_alvo]
colormap = plt.cm.plasma
plt.figure(figsize=(7, 7))
plt.title('Correlação entre features e rótulos', y=1.05, size=14)
sns.heatmap(data.astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap,
           linecolor='white', annot=True)

# Analisando o comprimento dos comentários no conjunto de teste e plotando um histograma
teste_df['comprimento'] = teste_df['comment_text'].apply(lambda x: len(str(x)))
plt.figure()
plt.hist(teste_df['comprimento'])
plt.show()

# Função para limpar o texto dos comentários
def limpar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"what's", "what is ", texto)
    texto = re.sub(r"\'s", " ", texto)
    texto = re.sub(r"\'ve", " have ", texto)
    texto = re.sub(r"can't", "cannot ", texto)
    texto = re.sub(r"n't", " not ", texto)
    texto = re.sub(r"i'm", "i am ", texto)
    texto = re.sub(r"\'re", " are ", texto)
    texto = re.sub(r"\'d", " would ", texto)
    texto = re.sub(r"\'ll", " will ", texto)
    texto = re.sub(r"\'scuse", " excuse ", texto)
    texto = re.sub('\W', ' ', texto)
    texto = re.sub('\s+', ' ', texto)
    texto = texto.strip(' ')
    return texto

# Aplicando a limpeza nos comentários do conjunto de treino
treino_df['comment_text'] = treino_df['comment_text'].map(lambda com: limpar_texto(com))

# Aplicando a limpeza nos comentários do conjunto de teste
teste_df['comment_text'] = teste_df['comment_text'].map(lambda com: limpar_texto(com))

# Removendo a coluna 'comprimento' do conjunto de treino
treino_df = treino_df.drop('comprimento', axis=1)

# Separando os conjuntos de treino e teste
X_treino = treino_df.comment_text
X_teste = teste_df.comment_text

print(X_treino.shape, X_teste.shape)

# Importando e instanciando o TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(max_features=5000, stop_words='english')

# Aprendendo o vocabulário nos dados de treino e criando uma matriz documento-termo
X_treino_dtm = vect.fit_transform(X_treino)

# Transformando os dados de teste usando o vocabulário aprendido, criando uma matriz documento-termo
X_teste_dtm = vect.transform(X_teste)

# Importando e instanciando o modelo de Regressão Logística
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
logreg = LogisticRegression(C=12.0)

# Criando um arquivo de submissão
submissao_binaria = pd.read_csv('/Users/caiolima/Documents/inteligência artificial/Lista EXTRA 3 - Mineração de textos/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')

# Treinando o modelo e gerando previsões para cada rótulo
for rotulo in cols_alvo:
    print('... Processando {}'.format(rotulo))
    y = treino_df[rotulo]
    # Treinando o modelo usando X_treino_dtm & y
    logreg.fit(X_treino_dtm, y)
    # Calculando a acurácia do treino
    y_pred_X = logreg.predict(X_treino_dtm)
    print('Acurácia do treino é {}'.format(accuracy_score(y, y_pred_X)))
    # Calculando as probabilidades previstas para X_teste
