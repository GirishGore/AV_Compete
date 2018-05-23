# -*- coding: utf-8 -*-
"""
Created on Wed May 23 13:03:36 2018

@author: Girish
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_train = pd.read_csv("E:\\Work\\AV_Compete\\BlackFriday\\train.csv")
data_train['isTrain'] = 1
data_test = pd.read_csv("E:\\Work\\AV_Compete\\BlackFriday\\test.csv")
data_test['isTrain'] = 0

data_train.info()
data_test.info()
Y = data_train['Purchase']
del data_train['Purchase']

uids = data_test['User_ID']
pids = data_test['Product_ID']
data = [ data_train , data_test]
data = pd.concat(data)
# to change use .astype() 
data['User_ID'] = data.User_ID.astype(object)
data.fillna(0)
data['Product_Category_2'] = data.Product_Category_2.astype(object)
data['Product_Category_3'] = data.Product_Category_3.astype(object)

data.info()


### One lable encoding of categorical variables
for ctype in data.columns[data.dtypes == 'object']: 	
#    if(ctype != 'User_ID' and ctype != 'Product_ID'):
        data[ctype] = data[ctype].factorize()[0];
## A final looka at teh data
data.describe(include='all')
data.head(5)

## Split back to test and train
data_test = data.loc[data['isTrain'] == 0]
data_train = data.loc[data['isTrain'] == 1]

## Applying linear regression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_train, Y , test_size=0.2, random_state=0)  

from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train)  

y_pred_train = regressor.predict(X_train)
y_pred = regressor.predict(data_test)

df_train_error = pd.DataFrame({'Actual':y_train , 'Predicted':y_pred_train})
df_test_error = pd.DataFrame({'Actual': y_test , 'Predicted': y_pred})

from sklearn import metrics  
print('Root Mean Squared Error (Train):', np.sqrt(metrics.mean_squared_error(y_train, y_pred_train)) ) 
print('Root Mean Squared Error (Test):', np.sqrt(metrics.mean_squared_error(y_test, y_pred))  )

Submit = pd.read_csv("E:\\Work\\AV_Compete\\BlackFriday\\Sample_Submission.csv")
Submit['User_ID'] = uids
Submit['Product_ID'] = pids
Submit['Purchase'] = y_pred
uids.shape
pids.shape
y_pred.shape
Submit.columns
Submit.to_csv('E:\\Work\\AV_Compete\\BlackFriday\\LinearReg.csv', index= False)


