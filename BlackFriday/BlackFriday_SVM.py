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



data = data.applymap(int)

### Normalized x  =    (X - Xmin) / (Xmax - Xmin)
from sklearn.preprocessing import MinMaxScaler
min_max=MinMaxScaler()
data = min_max.fit_transform(data)

data = pd.DataFrame( data , columns=Xcopy.columns)

data_train = data[data.is_Train == 1] 
data_test = data[data.is_Train == 0]

X = data_train.drop(['is_Train'] , axis=1)

X.dtypes
X.head()
X.shape
y.shape
#Import Library
from sklearn import svm
#svr_rbf = SVR(kernel='rbf')
#svr_lin = SVR(kernel='linear')
#svr_poly = SVR(kernel='poly')

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=200)
rf.fit(X,y)
rf.score(X,y)
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create SVM classification object 
model = svm.SVR(kernel='linear',C=1.0, epsilon=0.2) 
# there is various option associated with it, like changing kernel, gamma and C value. Will discuss more # about it in next section.Train the model using the training sets and check score
model.fit(X, y)
rf.score(X, y)
#Predict Output
data_test.columns
predicted= rf.predict(data_test.drop(['is_Train'] , axis=1))

##Submitting your work
Submit = pd.read_csv("E:\\Work\\AV_Compete\\BlackFriday\\Sample_Submission.csv")
Submit['User_ID'] = uids
Submit['Product_ID'] = pids
Submit['Purchase'] = predicted
Submit.to_csv('E:\\Work\\AV_Compete\\BlackFriday\\SVM_Latest.csv', index= False)

