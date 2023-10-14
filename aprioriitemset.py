import pandas as pd
from apyori import apriori


base = pd.read_csv('/Users/caiolima/Documents/inteligência artificial/lista 5/base.csv', encoding='cp1252')

transacoes = []
for i in range(len(base)):
    transacoes.append([str(base.values[i, j]) for j in range(base.shape[1])])


regras = apriori(transacoes, min_support=0.6, min_confidence=0.7)
saida = list(regras)

# mostrar itemset
for item in saida:
    mostrar = item[0] 
    items = [x for x in mostrar]
    if 'nan' not in items:
        print("Itemset: ", items)


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
print(RegrasFinais.sort_values(by='lift', ascending=False))
