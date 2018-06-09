# -*- coding: utf-8 -*-
"""
Created on Wed May 23 13:03:36 2018

@author: Girish
"""

import pandas as pd
import numpy as np

data_train = pd.read_csv("E:\\Work\\AV_Compete\\BlackFriday\\train.csv")
data_test = pd.read_csv("E:\\Work\\AV_Compete\\BlackFriday\\test.csv")

uids = data_test['User_ID']
pids = data_test['Product_ID']

data_train['is_Train'] = 1
data_test['is_Train'] = 0

data = pd.concat([data_train,data_test])
data.isnull().values.any()


y = data_train['Purchase']
data = pd.concat([data_train.drop(['Purchase'], axis = 1),data_test])

### Some basic checks for overlap. Proves that all USER_ID's and some PRODUCT_ID's are repeated in the test data set
data_train['User_ID'].map(lambda x : True if x in data_test['User_ID'] else False).value_counts()
data_train['Product_ID'].map(lambda x : True if x in data_test['Product_ID'] else False).value_counts()

## Filling na values with -999
data.isna().sum()
data.index

data.fillna(999 , inplace=True)

#Convert all the columns to string 
data = data.applymap(str)
data.dtypes

### Keep a copy of data
Xcopy = data.copy()

data = np.array(data)

from sklearn import preprocessing
for i in range(data.shape[1]):
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(data[:,i]))
    data[:, i] = lbl.transform(data[:, i])


data = pd.DataFrame( data , columns=Xcopy.columns)
data = data.applymap(int)

### Normalized x  =    (X - Xmin) / (Xmax - Xmin)
from sklearn.preprocessing import MinMaxScaler
min_max=MinMaxScaler()
data = min_max.fit_transform(data)

data = pd.DataFrame( data , columns=Xcopy.columns)

data_train = data[data.is_Train == 1] 
data_test = data[data.is_Train == 0]

X = data_train.drop(['is_Train'] , axis=1)


### RF Plain

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=200) ## Default is 10
rf.fit(X,y)
rf.score(X,y)

predicted= rf.predict(data_test.drop(['is_Train'] , axis=1))


##### RF With Paramters
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 5],
#    'min_samples_split': [8, 10, 12],
    'n_estimators': [500, 1000 ,1200]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(X,y)
print(rf)

import matplotlib.pyplot as plt
plt.plot(rf.feature_importances_)
plt.xticks(np.arange(X.shape[1]), data_train.columns.tolist() , rotation=10)
plt.show()

## Predicting to calculate train and test error
y_pred_train = rf.predict(X)
predicted = rf.predict(data_test.drop(['is_Train'] , axis=1))


##Submitting your work
Submit = pd.read_csv("E:\\Work\\AV_Compete\\BlackFriday\\Sample_Submission.csv")
Submit['User_ID'] = uids
Submit['Product_ID'] = pids
Submit['Purchase'] = predicted
Submit.to_csv('E:\\Work\\AV_Compete\\BlackFriday\\SVM_Latest.csv', index= False)

