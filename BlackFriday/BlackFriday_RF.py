# -*- coding: utf-8 -*-
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

X = X_train
y = y_train
### RF Plain

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=5) ## Default is 10
rf.fit(X,y)
rf.score(X,y)  ### Bigger is better

predicted_Train = rf.predict(X_train)
predicted_Test = rf.predict(X_test)
predicted= rf.predict(data_test)

### Getting Variable Importance Scores
rf.feature_importances_
sorted(zip(rf.feature_importances_,data_train.columns) , reverse=True)

## Generating test and train errors RMSE
from sklearn import metrics  
print('Root Mean Squared Error (Train):', np.sqrt(metrics.mean_squared_error(y_train, predicted_Train)) ) 
print('Root Mean Squared Error (Test):', np.sqrt(metrics.mean_squared_error(y_test, predicted_Test))  )

##Submitting your work
Submit = pd.read_csv("E:\\Work\\AV_Compete\\BlackFriday\\Sample_Submission.csv")
Submit['User_ID'] = uids
Submit['Product_ID'] = pids
Submit['Purchase'] = predicted
Submit.to_csv('E:\\Work\\AV_Compete\\BlackFriday\\RF.csv', index= False)

#### Ways of going through different paramters
##### RF With Paramters
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    #'max_depth': [80, 90],
    'max_features': [7],
    'min_samples_leaf': [5],
#    'min_samples_split': [8, 10, 12],
    'n_estimators': [80]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(X,y)
print(grid_search.best_params_)
print(grid_search.best_estimator_)
print(grid_search.cv_results_)
## Predicting to calculate train and test error
predict_Train = grid_search.predict(X_train)
predict_Test = grid_search.predict(X_test)
predicted = grid_search.predict(data_test)

## Generating test and train errors RMSE
from sklearn import metrics  
print('Root Mean Squared Error (Train):', np.sqrt(metrics.mean_squared_error(y_train, predict_Train)) ) 
print('Root Mean Squared Error (Test):', np.sqrt(metrics.mean_squared_error(y_test, predict_Test))  )


##Submitting your work
Submit = pd.read_csv("E:\\Work\\AV_Compete\\BlackFriday\\Sample_Submission.csv")
Submit['User_ID'] = uids
Submit['Product_ID'] = pids
Submit['Purchase'] = predicted
Submit.to_csv('E:\\Work\\AV_Compete\\BlackFriday\\RFGRid_Latest.csv', index= False)

