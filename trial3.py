#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 13:14:57 2019

@author: trvpo333
"""
import pandas as pd 

from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
import matplotlib.pyplot as plt
## Load dataset
url = "http://promise.site.uottawa.ca/SERepository/datasets/pc1.arff"
names = ['loc', 'v(g)', 'ev(g)', 'iv(g)', 'n', 'v', 'l', 'd', 'i', 'e',
         'b', 't', 'lOCode', 'lOComment', 'lOBlank', 'lOCodeAndComment', 'uniq_Op', 'uniq_Opnd', 'total_Op', 'total_Opnd'
         , 'branchCount', 'defects']
balance_data = pd.read_csv(url, names=names, skiprows=356)
    # Printing the dataswet shape 
print ("Dataset Length: ", len(balance_data)) 
print ("Dataset Shape: ", balance_data.shape) 
      
    # Printing the dataset obseravtions 
print ("Dataset:\n ",balance_data.head()) 

X = balance_data.values[:, 0:20]
Y = balance_data.values[:,21]
Y = Y.astype('int')
X_train, X_test, y_train, y_test = train_test_split( 
          X, Y, test_size = 0.3, random_state = 100)

model = GaussianNB()
model.fit(X_train, y_train)
