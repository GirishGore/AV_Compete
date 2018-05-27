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

info_train.head()

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

documents = info_test
a = cos_similarity(documents['abstract'])

from collections import defaultdict
dicts_abstract = defaultdict(list)

for i in range(0,len(a)):
    for j in range(0,len(a)):
        if(a[i][j] > 0.23 and i != j and documents.iloc[i]['set'] == documents.iloc[j]['set']):
            #print(documents.iloc[i]['pmid'] , "   " , documents.iloc[j]['pmid'] , "   " , a[i][j])
            print( documents.iloc[i]['set'], "==SET==", documents.iloc[j]['set'])
            dicts_abstract[documents.iloc[i]['pmid']].append(documents.iloc[j]['pmid'])
            
print("Number of elements in the dictionary",len(dicts_abstract.keys()))

documents = info_test
a = cos_similarity(documents['article_title'])

from collections import defaultdict
dicts_title = defaultdict(list)

for i in range(0,len(a)):
    for j in range(0,len(a)):
        if(a[i][j] > 0.30 and i != j and documents.iloc[i]['set'] == documents.iloc[j]['set']):
            #print(documents.iloc[i]['pmid'] , "   " , documents.iloc[j]['pmid'] , "   " , a[i][j])
            print( documents.iloc[i]['set'], "==SET==", documents.iloc[j]['set'])
            dicts_title[documents.iloc[i]['pmid']].append(documents.iloc[j]['pmid'])
            
print("Number of elements in the dictionary",len(dicts_title.keys()))


print(dicts_abstract)
print(dicts_title)
#### Final Preparation
from collections import defaultdict

dicts = defaultdict(list)

for tempdict in (dicts_abstract, dicts_title): # you can list as many input dicts as you want here
    for key, value in tempdict.items():
        dicts[key].append(value)

print(dicts.get(4171054))
from itertools import chain
for key, value in dicts.items():
    dicts[key] = list(chain.from_iterable(value))
    
print(dicts)

for index, row in submission.iterrows():
    #print (row["pmid"], row["ref_list"])
    #print (dicts.get(row["pmid"]))
    if (dicts.get(row["pmid"]) != None):
        val = dicts.get(row["pmid"])
    else :
        val = []
    submission.set_value(index,'ref_list', val )
    
    
submission.to_csv("E:\\Work\\AV_Compete\\Innoplexus\\HybridSet2030.csv" , index=False)
    