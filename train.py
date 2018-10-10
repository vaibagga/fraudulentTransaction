"""
Created on Mon Dec 18 22:12:13 2017

@author: Sahil Sulekhiya
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RF
#from sklearn.linear_model import LinearRegression
dataset = pd.read_csv('train.csv')
print(dataset.shape)
dataset.fillna(-99999, inplace = True)
x = []
for _ in dataset['transaction_id']:
    x.append(_[3:])
dataset['corrected_id'] = x
print(dataset.shape)
for i in range(1,19):
    dummy_temp = pd.get_dummies(dataset['cat_var_' + str(i)])
    dataset = pd.concat([dataset, dummy_temp],axis = 1)
    dataset.drop(['cat_var_' + str(i)], inplace = True, axis = 1)   
print(dataset.shape)
y = np.array(dataset['target'], dtype = float)
print(y.shape)    
print(dataset.shape)
dataset.drop(['transaction_id'],axis = 1, inplace = True)
dataset.drop(['target'],axis = 1, inplace = True)
print(dataset.shape)
X = np.array(dataset, dtype = float)
clf = RF()
clf.fit(X,y)
dataset = pd.read_csv('test.csv')
sample = pd.read_csv('sample_submissions.csv')
print(dataset.shape)
dataset.fillna(-99999, inplace = True)
x = []
for _ in dataset['transaction_id']:
    x.append(_[3:])
dataset['corrected_id'] = x
for i in range(1,19):
    dummy_temp = pd.get_dummies(dataset['cat_var_' + str(i)])
    dataset = pd.concat([dataset, dummy_temp],axis = 1)
    dataset.drop(['cat_var_' + str(i)],axis = 1,inplace = True)
print(dataset.shape)
dataset.drop(['transaction_id'],axis = 1, inplace = True)
print(dataset.shape)
X = np.array(dataset, dtype = float)
result = clf.predict_proba(X)
ptr = open('submission3.csv','a+')
ptr.write('transaction_id,target\n')
for i in range(dataset.shape[0]):
    ptr.write(str(sample['transaction_id'][i]))
    ptr.write(',')
    ptr.write(str(result[i][1]))
    ptr.write('\n')
ptr.close()


