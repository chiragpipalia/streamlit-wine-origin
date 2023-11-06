from xgboost import XGBClassifier
import pandas as pd
import sklearn
import numpy as np

df = pd.read_csv('wine.data')
columns = ['Target','Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols'
      ,'Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline'] 
df.columns = columns
df['Target'] = df['Target'].map({1:0, 2:1, 3:2})
# df = df[['Target', 'Alcohol','Malic acid','Ash','Alcalinity of ash']].copy()
X = df.drop(columns = 'Target').copy()
y = df['Target']

model = XGBClassifier()
model.fit(X, y)
predictions = model.predict(X)
imp_dict = dict(zip(columns[1:], model.feature_importances_))

imp_dict = sorted(imp_dict.items(), key=lambda x:x[1], reverse=True)
print(imp_dict)

print(np.unique(predictions, return_counts=True))
print(y.value_counts())

top_features = ['OD280/OD315 of diluted wines', 'Color intensity', 'Flavanoids', 'Proline']
X = df[top_features].copy()
y = df['Target']

model = XGBClassifier()
model.fit(X, y)
predictions = model.predict(X)


model.save_model('xgb_model.json')

print('done')
