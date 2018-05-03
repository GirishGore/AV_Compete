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

print(X.groupby(['Item_Type','Outlet_Type'])['Item_Outlet_Sales'].mean())
temp = X.groupby('Item_Identifier')['Item_Outlet_Sales'].mean()
plt.hist(temp)
print(temp.head)
temp['Item_Cat_New'] = pd.qcut(temp, 12, labels=False)


plt.hist(np.log(y.astype(float)))
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
X['Item_Fat_New'] = np.where((X['Item_Fat_Content'] == 'Regular') | (X['Item_Fat_Content'] == 'reg') , "Reg" , "Fat")

#Impute 0 values with mean visibility of that product:
visibility_avg = X.pivot_table(values='Item_Visibility', index='Item_Identifier')
print(visibility_avg.head)
missing = (X['Item_Visibility'] == 0)
print(missing.head)
print ('Number of 0 values initially: %d'%sum(missing))
X.loc[missing,'Item_Visibility'] = X.loc[missing,'Item_Identifier'].apply(lambda x: visibility_avg.loc[x])
print ('Number of 0 values after modification: %d'%sum(X['Item_Visibility'] == 0))
X['Item_Visibility'] = pd.qcut(X['Item_MRP'], 10 , labels=False)

## Binning the MRP Variable
X['Item_MRP_New'] = pd.qcut(X['Item_MRP'], 24 , labels=False)
X['Item_MRP_New'].value_counts()

X['Item_Weight_New'] = pd.qcut(X['Item_Weight'], 15, labels=False)
X['Item_Weight_New'].value_counts()

X['Outlet_Type_New'] = np.where(X['Outlet_Type'] == 'Grocery Store', 1 , 0)


X['Years_Old'] = (2013 - X['Outlet_Establishment_Year'])
plt.hist(X['Years_Old'])
X['Age_New'] = pd.qcut(X['Years_Old'], 4, labels=False)
X['Age_New'].value_counts()
### Still if NA remain fill it with -999
##New categories based on name conventions
#Get the first two characters of ID:
X['Item_Type_Combined'] = X['Item_Identifier'].apply(lambda x: x[0:2])
#Rename them to more intuitive categories:
X['Item_Type_Combined'] = X['Item_Type_Combined'].map({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'})
X['Item_Type_Combined'].value_counts()
X.loc[X['Item_Type_Combined']=="NC",'Item_Fat_New'] = "Not Valid"

X = pd.get_dummies(data=X, columns=['Outlet_Type'])
X = pd.get_dummies(data=X, columns=['Outlet_Size'])
X = pd.get_dummies(data=X, columns=['Item_Type_Combined'])
X = pd.get_dummies(data=X, columns=['Outlet_Location_Type'])



X = X.drop(['Item_Outlet_Sales','Item_MRP','Item_Weight','Item_Fat_Content','Outlet_Establishment_Year','Years_Old'], axis=1)

#X['Item_Visibility_MeanRatio'] = X.apply(lambda X: X['Item_Visibility']/visibility_avg[X['Item_Identifier']], axis=1)
#print (X['Item_Visibility_MeanRatio'].describe())


train = X.copy()
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

from xgboost import XGBRegressor
from sklearn.grid_search import GridSearchCV
xgb = XGBRegressor(verbose_eval=1)
param_grid = {
                 'n_estimators': [900,800],
                 'max_depth': [2],
                 'learning_rate': [0.01],
                 'min_child_weight': [3],
                 'subsample': [0.85],
                 'colsample_bytree' : [0.9]
             }
grid_clf = GridSearchCV(xgb, param_grid, cv=10)
grid_clf.fit(X, y)


from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(y, grid_clf.predict(X))
print(rmse)
predicted = grid_clf.predict(Xt)

print(grid_clf.grid_scores_)
print(grid_clf.best_params_)



Submit = pd.read_csv("E:\\Work\\AV_Compete\\BigMartSalesIII\\SampleSubmission.csv")
Submit['Item_Outlet_Sales'] = predicted
Submit.to_csv('E:\\Work\\AV_Compete\\BigMartSalesIII\\Python\\XGB_BasePlus.csv', index= False)

plt.plot(grid_clf.best_estimator_.feature_importances_)
plt.xticks(np.arange(X.shape[1]), X.columns.tolist() , rotation=80)
plt.show()


