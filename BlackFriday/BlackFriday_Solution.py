# -*- coding: utf-8 -*-
"""
Created on Wed May 23 13:03:36 2018

@author: Girish
"""

import pandas as pd
import numpy as np

data_train = pd.read_csv("E:\\Work\\AV_Compete\\BlackFriday\\train.csv")
data_test = pd.read_csv("E:\\Work\\AV_Compete\\BlackFriday\\test.csv")


data_train['isTrain'] = 1
data_test['isTrain'] = 0

uids = data_test['User_ID']
pids = data_test['Product_ID']

data = pd.concat([data_train,data_test])

### Some basic checks for overlap. Proves that all USER_ID's and some PRODUCT_ID's are repeated in the test data set
data_train['User_ID'].map(lambda x : True if x in data_test['User_ID'] else False).value_counts()
data_train['Product_ID'].map(lambda x : True if x in data_test['Product_ID'] else False).value_counts()


data_train['Product_ID'].isin(data_test['Product_ID']).value_counts()

data_train['User_ID'].head()
data_test[data_test.User_ID == 1000001]

data.describe()
data.isna().sum()
data = data.fillna(-999)

corr_mat = data.corr()
print(corr_mat)

import matplotlib.pyplot as plt
plt.matshow(corr_mat)
data.info()
data.columns

data['Age'].describe()
data['Age'].value_counts()
data['Age'] = data['Age'].astype('category')
data['Age_Categories'] = data['Age'].cat.codes

data.dtypes


data['City_Category'].describe()
data['City_Category'].unique()
data['City_Category'] = data['City_Category'].astype('category')
data['City_Categories'] = data['City_Category'].cat.codes

data['Gender_Categories'] = (data['Gender'].astype('category')).cat.codes

data.Marital_Status = data.Marital_Status.astype('category')

data.Occupation.value_counts()
data.Occupation = (data.Occupation.astype('category')).cat.codes

data.Product_Category_2 = data.Product_Category_2.astype('int64')
data.Product_Category_3 = data.Product_Category_3.astype('int64')

data['Product_ID_Categories'] = (data.Product_ID.astype('category')).cat.codes
data.Stay_In_Current_City_Years.unique()

# leveraging the variables already created above
mapper = {'0': 0, '1': 1, '2': 2 , '3': 3 ,'4+': 4}
data.Stay_In_Current_City_Years = data.Stay_In_Current_City_Years.replace(mapper)
data.Stay_In_Current_City_Years.unique()



data = data[['Age_Categories','City_Categories','Gender_Categories','Product_ID_Categories', 
                   'Marital_Status','Occupation','Product_Category_1','Product_Category_2','isTrain','Purchase']]

data_train = data[data.isTrain == 1]
data_test = data[data.isTrain == 0]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_train.drop('Purchase',axis=1), data_train['Purchase'] , test_size=0.2, random_state=0)

from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(data_train.drop('Purchase',axis=1))
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=1000,oob_score=True)
rf.fit(X_train,y_train)
print(rf)

plt.plot(rf.feature_importances_)
plt.xticks(np.arange(X_train.shape[1]), data_train.columns.tolist() , rotation=10)
plt.show()

## Predicting to calculate train and test error
y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)
y_pred = rf.predict(data_test.drop('Purchase',axis=1))

## Generating test and train errors RMSE
from sklearn import metrics  
print('Root Mean Squared Error (Train):', np.sqrt(metrics.mean_squared_error(y_train, y_pred_train)) ) 
print('Root Mean Squared Error (Test):', np.sqrt(metrics.mean_squared_error(y_test, y_pred_test))  )

##Submitting your work
Submit = pd.read_csv("E:\\Work\\AV_Compete\\BlackFriday\\Sample_Submission.csv")
Submit['User_ID'] = uids
Submit['Product_ID'] = pids
Submit['Purchase'] = y_pred
Submit.to_csv('E:\\Work\\AV_Compete\\BlackFriday\\RF_Basic.csv', index= False)

