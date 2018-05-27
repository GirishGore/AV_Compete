# -*- coding: utf-8 -*-
"""
Created on Sun May 27 09:39:22 2018

@author: Girish
"""

import pandas as pd
import numpy as np

data_train = pd.read_csv("E:\\Work\\AV_Compete\\Innoplexus\\train.csv")
data_test = pd.read_csv("E:\\Work\\AV_Compete\\Innoplexus\\test.csv")
submission = pd.read_csv("E:\\Work\\AV_Compete\\Innoplexus\\sample_submission.csv")

info_train = pd.read_csv("E:\\Work\\AV_Compete\\Innoplexus\\information_train.csv" , sep="\t")
info_test = pd.read_csv("E:\\Work\\AV_Compete\\Innoplexus\\information_test.csv", sep="\t")

#for index, text in input_train_data.iterrows():
#    print(text['abstract'])
#    tokens = [t for t in text['abstract'].split() if t not in stopwords.words('english')]
#    lemtokens = [lemmer.lemmatize(token) for token in tokens]
#    tokenList.append(lemtokens)
# input_train_data['tokenized'] = tokenList

import string 
from nltk import word_tokenize
from nltk import WordNetLemmatizer
lemmer = WordNetLemmatizer()
def LemTokens(tokens):
     return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
     return LemTokens(word_tokenize(text.lower().translate(remove_punct_dict)))
 
from sklearn.feature_extraction.text import TfidfVectorizer
TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
def cos_similarity(textlist):
    tfidf = TfidfVec.fit_transform(textlist)
    return (tfidf * tfidf.T).toarray()

documents = info_test['abstract']
a = cos_similarity(documents)

from collections import defaultdict
dicts = defaultdict(list)

for i in range(0,len(a)):
    for j in range(0,len(a)):
        if(a[i][j] > 0.12 and i != j):
            print(info_test.iloc[i]['pmid'] , "   " , info_test.iloc[j]['pmid'] , "   " , a[i][j])
            dicts[info_test.iloc[i]['pmid']].append(info_test.iloc[j]['pmid'])
            
print("Number of elements in the dictionary",len(dicts.keys()))

for i in range(0,len(data_train)):
    print(data_train['pmid'][i] , " value" ,dicts.get(data_train['pmid'][i]) , " expected ",data_train['ref_list'][i] )

submission.head()

print(dicts.get(6211173))
print(len(dicts.keys()))
print(submission.shape)

for index, row in submission.iterrows():
    #print (row["pmid"], row["ref_list"])
    #print (dicts.get(row["pmid"]))
    if (dicts.get(row["pmid"]) != None):
        val = dicts.get(row["pmid"])
    else :
        val = []
    submission.set_value(index,'ref_list', val )
    
submission.to_csv("E:\\Work\\AV_Compete\\Innoplexus\\base.csv" , index=False)
    
    
# submission.loc[index, "ref_list"] = dicts.get(row["pmid"])
    
    
      
#############################################################################

    
from sklearn.feature_extraction.text import CountVectorizer
LemVectorizer = CountVectorizer(tokenizer=LemNormalize, stop_words='english')
LemVectorizer.fit_transform(input_train_data['abstract'])
 
 
tf_matrix = LemVectorizer.transform(input_train_data['abstract']).toarray()
print(tf_matrix)

import math
def idf(n,df):
    result = math.log((n+1.0)/(df+1.0)) + 1
    return result
print("The idf for terms that appear in one document: " ,str(idf(4,1)))
print("The idf for terms that appear in two documents: " , str(idf(4,2)))

from sklearn.feature_extraction.text import TfidfTransformer
tfidfTran = TfidfTransformer(norm="l2")
tfidfTran.fit(tf_matrix)
print(tfidfTran)

tfidf_matrix = tfidfTran.transform(tf_matrix)
print(tfidf_matrix.toarray())


cos_similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
print(cos_similarity_matrix.shape)
print(input_train_data.shape)

from sklearn.feature_extraction.text import TfidfVectorizer
TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
def cos_similarity(textlist):
    tfidf = TfidfVec.fit_transform(textlist)
    return (tfidf * tfidf.T).toarray()
a = cos_similarity(input_train_data['abstract'])

print(a[1][1])
print(len(a))
for i in range(0,len(a)):
    for j in range(0,len(a)):
        print(a[i][j])