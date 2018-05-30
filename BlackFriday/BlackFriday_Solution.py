# -*- coding: utf-8 -*-
"""
Created on Wed May 23 13:03:36 2018

@author: Girish
"""

import pandas as pd
import numpy as np

data_train = pd.read_csv("E:\\Work\\AV_Compete\\BlackFriday\\train.csv")
data_test = pd.read_csv("E:\\Work\\AV_Compete\\BlackFriday\\test.csv")


data_train['isTrain'] = 1
data_test['isTrain'] = 0

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





