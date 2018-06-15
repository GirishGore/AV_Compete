# -*- coding: utf-8 -*-
"""
Created on Wed May 23 13:03:36 2018

@author: Girish
"""

## Importing core pandas and numpy libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Loading train and test data and assigning a new variable to differentiate train data
data_train = pd.read_csv("E:\\Work\\AV_Compete\\BlackFriday\\train.csv")
data_train['isTrain'] = 1
data_test = pd.read_csv("E:\\Work\\AV_Compete\\BlackFriday\\test.csv")
data_test['isTrain'] = 0

## Extracting the Dependent Variable
Y = data_train['Purchase']
del data_train['Purchase']

uids = data_test['User_ID']
pids = data_test['Product_ID']

## Combining the train and test data for feature engineering.
data = [ data_train , data_test]
data = pd.concat(data)

data.dtypes
data = data.applymap(str)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

for ctype in data.columns[data.dtypes == 'object']:
    le.fit(data[ctype].values)
    data[ctype]=le.transform(data[ctype])

data.describe()
data.info()
## Check for right data types or change them

## Split back to test and train
data_test = data.loc[data['isTrain'] == 0]
data_train = data.loc[data['isTrain'] == 1]



## Splitting data into test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_train, Y , test_size=0.2, random_state=0)  

## Applying linear regression
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train)  
print (regressor.coef_,regressor.intercept_)


## Predicting to calculate train and test error
y_pred_train = regressor.predict(X_train)
y_pred_test = regressor.predict(X_test)
y_pred = regressor.predict(data_test)


## Creating train_error and test_error data frames and plotting them
df_train_error = pd.DataFrame({'Actual':y_train , 'Predicted':y_pred_train})
df_train_error = df_train_error.sort_values(['Actual','Predicted'])
plt.rcParams['agg.path.chunksize'] = 10000
df_train_error.plot.scatter(x=['Predicted'] , y=['Actual'])

df_test_error = pd.DataFrame({'Actual': y_test , 'Predicted': y_pred_test})
df_test_error.plot.scatter(x=['Predicted'] , y=['Actual'])


## Generating test and train errors RMSE
from sklearn import metrics  
print('Root Mean Squared Error (Train):', np.sqrt(metrics.mean_squared_error(y_train, y_pred_train)) ) 
print('Root Mean Squared Error (Test):', np.sqrt(metrics.mean_squared_error(y_test, y_pred_test))  )



##Submitting your work
Submit = pd.read_csv("E:\\Work\\AV_Compete\\BlackFriday\\Sample_Submission.csv")
Submit['User_ID'] = uids
Submit['Product_ID'] = pids
Submit['Purchase'] = y_pred
Submit.to_csv('E:\\Work\\AV_Compete\\BlackFriday\\LinearReg1.csv', index= False)


