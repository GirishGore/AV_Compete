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

for ctype in train.columns[X.dtypes == 'object']:
	X[ctype] = X[ctype].factorize()[0];

print(X.columns)
print(X.head)
print(y.head)

Xt = X.loc[X['isTrain'] == 0]
X = X.loc[X['isTrain'] == 1]


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X,y)

from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(y, rf.predict(X))
print(rmse)
predicted = rf.predict(Xt)

Submit = pd.read_csv("E:\\Work\\AV_Compete\\BigMartSalesIII\\SampleSubmission.csv")
Submit['Item_Outlet_Sales'] = predicted

pd.to_csv('RF_Submit.csv',Submit)

plt.plot(rf.feature_importances_)
plt.xticks(np.arange(X.shape[1]), X.columns.tolist() , rotation=80)
plt.show()
