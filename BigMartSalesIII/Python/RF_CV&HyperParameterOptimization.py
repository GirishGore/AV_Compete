# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

X = pd.read_csv("E:\\Work\\AV_Compete\\BigMartSalesIII\\Train.csv")
X['isTrain']= 1
Xt = pd.read_csv("E:\\Work\\AV_Compete\\BigMartSalesIII\\Test.csv")
Xt['isTrain'] = 0
y = X.Item_Outlet_Sales

X = [X , Xt]
X = pd.concat(X)


print(X.head())
print(X.dtypes)
X = X.fillna(-999)


X = X.drop(['Item_Outlet_Sales'], axis=1)

train = X.copy()

for ctype in train.columns[X.dtypes == 'object']: 	X[ctype] = X[ctype].factorize()[0];

print(X.columns)
print(X.head)
print(y.head)

Xt = X.loc[X['isTrain'] == 0]
X = X.loc[X['isTrain'] == 1]


#from sklearn.ensemble import RandomForestRegressor
#rf = RandomForestRegressor(n_estimators=1500,verbose=1)
#rf.fit(X,y)

from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
rf = RandomForestRegressor(verbose=1)
param_grid = {
                 'n_estimators': [1100,1200,1300],
                 'max_depth': [4,5,6]
             }
grid_clf = GridSearchCV(rf, param_grid, cv=10)
grid_clf.fit(X, y)


from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(y, grid_clf.predict(X))
print(rmse)
predicted = grid_clf.predict(Xt)

print(grid_clf.grid_scores_)
print(grid_clf.best_params_)



Submit = pd.read_csv("E:\\Work\\AV_Compete\\BigMartSalesIII\\SampleSubmission.csv")
Submit['Item_Outlet_Sales'] = predicted
Submit.to_csv('E:\\Work\\AV_Compete\\BigMartSalesIII\\Python\\RF_Submit1500CV.csv', index= False)

plt.plot(grid_clf.best_estimator_.feature_importances_)
plt.xticks(np.arange(X.shape[1]), X.columns.tolist() , rotation=80)
plt.show()
