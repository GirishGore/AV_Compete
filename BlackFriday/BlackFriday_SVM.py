# -*- coding: utf-8 -*-
"""
Created on Wed May 23 13:03:36 2018

@author: Girish
"""

import pandas as pd
import numpy as np

data_train = pd.read_csv("E:\\Work\\AV_Compete\\BlackFriday\\train.csv")
data_test = pd.read_csv("E:\\Work\\AV_Compete\\BlackFriday\\test.csv")


data_train['is_Train'] = 1
data_test['is_Train'] = 0

data = pd.concat([data_train,data_test])
data.isnull().values.any()

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

##data = data[['Age_Categories','City_Categories','Gender_Categories','Product_ID_Categories', 'Marital_Status','Occupation','Product_Category_1','Product_Category_2','is_Train','Purchase']]

data = data[['Age_Categories','City_Categories','Gender_Categories','is_Train','Purchase']]
data.describe(include='all')
data.isnull().values.any()

#data.fillna(data.mean() , inplace=True)


#for ctype in Xcopy.columns[Xcopy.dtypes == 'object']:
#	Xcopy[ctype] = Xcopy[ctype].factorize()[0];
    
for ctype in data.columns:
	data[ctype] = data[ctype].astype('category');

### Basic Label Encoding
Xcopy = data.copy()
data_train = Xcopy[Xcopy.is_Train == 1]
data_test = Xcopy[Xcopy.is_Train == 0]

y = data_train['Purchase']
X = data_train.drop(['Purchase','is_Train'] , axis=1)

del  Xcopy , corr_mat , data , data_train , mapper

X.info()
X.head()
y.head()

X = X['City_Categories']
X = X.astype('int64');
X = X.reshape(-1, 1)
y = y.astype('float64');
#Import Library
from sklearn import svm
#svr_rbf = SVR(kernel='rbf')
#svr_lin = SVR(kernel='linear')
#svr_poly = SVR(kernel='poly')

#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create SVM classification object 
model = svm.SVR(kernel='linear') 
# there is various option associated with it, like changing kernel, gamma and C value. Will discuss more # about it in next section.Train the model using the training sets and check score
model.fit(X, y)
model.score(X, y)
#Predict Output
predicted= model.predict(data_test.drop(['Purchase'] , axis=1))

##Submitting your work
Submit = pd.read_csv("E:\\Work\\AV_Compete\\BlackFriday\\Sample_Submission.csv")
Submit['User_ID'] = uids
Submit['Product_ID'] = pids
Submit['Purchase'] = y_pred
Submit.to_csv('E:\\Work\\AV_Compete\\BlackFriday\\SVM_Latest.csv', index= False)

