# -*- coding: utf-8 -*-
"""
Created on Wed May 23 13:03:36 2018

@author: Girish
"""

## Importing core pandas and numpy libraries
import pandas as pd
import numpy as np

## Loading train and test data and assigning a new variable to differentiate train data
data_train = pd.read_csv("E:\\Work\\AV_Compete\\BlackFriday\\train.csv")
data_train['isTrain'] = 1
data_test = pd.read_csv("E:\\Work\\AV_Compete\\BlackFriday\\test.csv")
data_test['isTrain'] = 0



data_train.columns
data = [data_train , data_test]
data = pd.concat(data)
#data['Product_ID'] = data['Product_ID'].str[1:]
#data['Product_ID'] = data['Product_ID'].astype(int)

table = data.pivot_table(values='Purchase' , index=['User_ID','Product_ID'] , aggfunc=np.mean)
table = table.fillna(np.mean(data['Purchase']))
print(table)
table.columns
table.head()
table.loc[(1000004, 'P00128942'),'Purchase']
table.loc[(1001616, 'P00278642'),'Purchase']
#table.loc[(1000001,97142),'Purchase']

data_test.head()
data_test.info()
# data_test['Product_ID'] = data_test['Product_ID'].str[1:]  
#data_test['Product_ID'] = data_test['Product_ID'].astype(int)

#data_test['Purchase'] = [table.loc((row['User_ID'].astype(int),row['Product_ID'].astype(int)),'Purchase') for row in data_test]

data_test['Purchase'] = [ table.loc[(x,y),'Purchase'] for x, y in zip(data_test['User_ID'], data_test['Product_ID'])]
[ (x,y) for x, y in zip(data_test['User_ID'], data_test['Product_ID'])]

submit = pd.read_csv("E:\\Work\\AV_Compete\\BlackFriday\\Sample_Submission.csv")
submit = data_test[['User_ID','Product_ID','Purchase']]
submit.shape
submit.to_csv("E:\\Work\\AV_Compete\\BlackFriday\\Base.csv", index=False)

### Trying another variant

table = data.pivot_table(values='Purchase' , index=['Product_ID'] , aggfunc=np.mean)
table = table.fillna(np.mean(data['Purchase']))
print(table)
table.columns
table.head()
table.loc['P00128942','Purchase']
table.loc['P00278642','Purchase']
#table.loc[(1000001,97142),'Purchase']

data_test.head()
data_test.info()
# data_test['Product_ID'] = data_test['Product_ID'].str[1:]  
#data_test['Product_ID'] = data_test['Product_ID'].astype(int)

#data_test['Purchase'] = [table.loc((row['User_ID'].astype(int),row['Product_ID'].astype(int)),'Purchase') for row in data_test]

data_test['Purchase'] = [ table.loc[y,'Purchase'] for x, y in zip(data_test['User_ID'], data_test['Product_ID'])]

submit = pd.read_csv("E:\\Work\\AV_Compete\\BlackFriday\\Sample_Submission.csv")
submit = data_test[['User_ID','Product_ID','Purchase']]
submit.shape
submit.to_csv("E:\\Work\\AV_Compete\\BlackFriday\\Base.csv", index=False)
