import pandas as pd
from apyori import apriori

base= pd.read_csv('/Users/caiolima/Documents/inteligência artificial/lista 5/base.csv', encoding='cp1252')

singulares = set(item for sublist in base.values for item in sublist if item == item)  # item == item filters out NaN

transacoes = []
for i in range(len(base)):
    transaction = [str(base.values[i, j]) for j in range(base.shape[1])]
    for item in singulares:
        if item not in transaction:
            transaction.append('no_' + str(item))
    transacoes.append(transaction)

print(transacoes)
base.shape

transacoes = []
for i in range(len(base)):
  #print(i)
  #print(base_mercado1.values[i, 0])
  transacoes.append([str(base.values[i, j]) for j in range(base.shape[1])])

transacoes

type(transacoes)

regras = apriori(transacoes, min_support = 0.6, min_confidence = 0.7)
saida = list(regras) 
print(len(saida))
print(saida)

print(saida[0])

print(saida[1])

print(saida[2])

Antecedente = []
Consequente = []
suporte = []
confianca = []
lift = []

for resultado in saida:
  s = resultado[1]
  result_rules = resultado[2]
  for result_rule in result_rules:
    a = list(result_rule[0])
    b = list(result_rule[1])
    c = result_rule[2]
    l = result_rule[3]
    if 'nan' in a or 'nan' in b: continue
    if len(a) == 0 or len(b) == 0: continue
    Antecedente.append(a)
    Consequente.append(b)
    suporte.append(s)
    confianca.append(c)
    lift.append(l)
    RegrasFinais = pd.DataFrame({'Antecedente': Antecedente, 'Consequente': Consequente, 'suporte': suporte, 'confianca': confianca, 'lift': lift})

RegrasFinais

RegrasFinais.sort_values(by='lift', ascending =False)
