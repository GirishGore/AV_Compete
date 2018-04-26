# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

X = pd.read_csv("E:\\Work\\AV_Compete\\BigMartSalesIII\\Train.csv")
print(X.head())
print(X.dtypes)
X = X.fillna(-999)

y = X.Item_Outlet_Sales
X = X.drop(['Item_Outlet_Sales'], axis=1)

train = X.copy()

for ctype in train.columns[X.dtypes == 'object']:
	if( ctype != 'Loan_ID'):
		X[ctype] = X[ctype].factorize()[0];

print(X.columns)
print(X.head)
print(y.head)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X,y)

plt.plot(rf.feature_importances_)
plt.xticks(np.arange(X.shape[1]), X.columns.tolist() , rotation=80)
plt.show()
