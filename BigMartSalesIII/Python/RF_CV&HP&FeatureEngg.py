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


## Combining the test and train data sets for Feature engineering
X = [X , Xt]
X = pd.concat(X)


X.describe(include='all')

#### Checking for NA Value columns
X.isna().any()

## Removing NA's by filling in Item_Weights
X["Item_Weight"].isnull().sum()
X["Item_Weight"].fillna(X.groupby("Item_Identifier")["Item_Weight"].transform("mean"), inplace=True)
##Still 4 Remain which would be filled by -999
X["Outlet_Size"].isnull().sum()
X["Outlet_Size"].describe()
X.Outlet_Type.unique()
X.loc[X.Outlet_Type == "Grocery Store", 'Outlet_Size'] = "Small"
X.loc[X.Outlet_Type == "Supermarket Type1", 'Outlet_Size'] = "Small"
X.loc[X.Outlet_Type == "Supermarket Type2", 'Outlet_Size'] = "Medium"
X.loc[X.Outlet_Type == "Supermarket Type3", 'Outlet_Size'] = "High"

## Removing the ambiguity in the fat content
X['Item_Fat_Content'].unique()
X['Item_Fat_New'] = np.where((X['Item_Fat_Content'] == 'Regular') | (X['Item_Fat_Content'] == 'reg') , 1 , 0)

## Binning the MRP Variable
X['Item_MRP_New'] = pd.qcut(X['Item_MRP'], 24 , labels=False)

X['Item_Weight_New'] = pd.qcut(X['Item_Weight'], 15, labels=False)

X['Outlet_Type_New'] = np.where(X['Outlet_Type'] == 'Grocery Store', 1 , 0)

X['Years_Old'] = (2013 - X['Outlet_Establishment_Year'])

### Still if NA remain fill it with -999
##X = X.fillna(-999)


X = X.drop(['Item_Outlet_Sales','Item_MRP','Item_Weight','Item_Fat_Content','Outlet_Establishment_Year'], axis=1)

train = X.copy()

print(X.dtypes)
### One lable encoding of categorical variables
for ctype in train.columns[X.dtypes == 'object']: 	X[ctype] = X[ctype].factorize()[0];


## A final looka at teh data
X.describe(include='all')

## Split back to test and train
Xt = X.loc[X['isTrain'] == 0]
X = X.loc[X['isTrain'] == 1]



#from sklearn.ensemble import RandomForestRegressor
#rf = RandomForestRegressor(n_estimators=1500,verbose=1)
#rf.fit(X,y)

from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
rf = RandomForestRegressor(verbose=1)
param_grid = {
                 'n_estimators': [1200],
                 'max_depth': [5],
                 "max_features"      : ["auto"],
                 "bootstrap" :  [True],
                 'min_samples_split': [3],
                 'min_samples_leaf': [4]
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
Submit.to_csv('E:\\Work\\AV_Compete\\BigMartSalesIII\\Python\\RF_Submit_FE.csv', index= False)

plt.plot(grid_clf.best_estimator_.feature_importances_)
plt.xticks(np.arange(X.shape[1]), X.columns.tolist() , rotation=80)
plt.show()


