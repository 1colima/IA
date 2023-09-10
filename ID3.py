import pandas as pd
from chefboost import Chefboost as chef
from chefboost.training import Training
base = pd.read_csv('/content/drive/MyDrive/Teste1.csv', sep=',', encoding='utf-8')
config = {'algorithm': 'ID3'}
gains = Training.findGains(base, config)
print(base)
print(gains)

model = chef.fit(base, config = config, target_label='Decision')
