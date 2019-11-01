#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 17:22:10 2019

@author: trvpo333
"""
from scipy.io import arff

# Python version
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas as pd
print('pandas: {}'.format(pd.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))


# Load libraries
from sklearn.tree import export_graphviz
from sklearn import tree
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split  
from sklearn.externals.six import StringIO 
from sklearn.datasets import load_iris
from pydot import graph_from_dot_data
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


## Load dataset
url = "http://promise.site.uottawa.ca/SERepository/datasets/pc1.arff"
names = ['loc', 'v(g)', 'ev(g)', 'iv(g)', 'n', 'v', 'l', 'd', 'i', 'e',
         'b', 't', 'lOCode', 'lOComment', 'lOBlank', 'lOCodeAndComment', 'uniq_Op', 'uniq_Opnd', 'total_Op', 'total_Opnd'
         , 'branchCount', 'defects']
dataset = pd.read_csv(url, names=names, skiprows=356)
#print(dataset.shape)
#print(dataset.head(1451))
#print(dataset.describe())

# Printing the dataswet shape 
print ("Dataset Length: ", len(dataset)) 
print ("Dataset Shape: ", dataset.shape) 
X = dataset.values[:, 0:20]
Y = dataset.values[:,21]
Y = Y.astype('int')
X_train, X_test, y_train, y_test = train_test_split( 
          X, Y, test_size = 0.3, random_state = 100)
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

dot_data = StringIO()
export_graphviz(dt, out_file=dot_data, feature_names=names)
(graph, ) = graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())