#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 15:57:24 2019

@author: trvpo333
"""



# Run this program on your local python 
# interpreter, provided you have installed 
# the required libraries. 
  
# Importing the required packages 
import pandas as pd 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
#import matplotlib.pyplot as plt
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydotplus
import matplotlib.pyplot as plt



  
# Function importing Dataset 
def importdata(): 

    url = "http://promise.site.uottawa.ca/SERepository/datasets/kc1.arff"
    names = ['loc', 'v(g)', 'ev(g)', 'iv(g)', 'n', 'v', 'l', 'd', 'i', 'e',
         'b', 't', 'lOCode', 'lOComment', 'lOBlank', 'lOCodeAndComment', 'uniq_Op', 'uniq_Opnd', 'total_Op', 'total_Opnd'
         , 'branchCount', 'defects']
    #balance_data = pd.read_csv(url, names=names, skiprows=356)
    balance_data = pd.read_csv("jm1.arff", names=names, skiprows= 9)
    # Printing the dataswet shape 
    print ("Dataset Length: ", len(balance_data)) 
    print ("Dataset Shape: ", balance_data.shape) 
      
    # Printing the dataset obseravtions 
    print ("Dataset:\n ",balance_data.head()) 
    return balance_data 
  
## Function to split the dataset 
def splitdataset(balance_data): 
  
    # Seperating the target variable 
    X = balance_data.values[:, 0:20] 
    Y = balance_data.values[:, 21] 
    #Y = Y.astype('int')

  
    # Spliting the dataset into train and test 
    X_train, X_test, y_train, y_test = train_test_split(  
    X, Y, test_size = 0.3, random_state = 100) 
      
    return X, Y, X_train, X_test, y_train, y_test 
#      
## Function to perform training with giniIndex. 
def train_using_gini(X_train, X_test, y_train): 
  
    # Creating the classifier object 
    clf_gini = DecisionTreeClassifier(criterion = "gini", 
            random_state = 100,max_depth=7, min_samples_leaf=5) 
  
    # Performing training 
    clf_gini.fit(X_train, y_train) 
    return clf_gini 
#      
## Function to perform training with entropy. 
def train_using_entropy(X_train, X_test, y_train): 
  
    # Decision tree with entropy 
    clf_entropy = DecisionTreeClassifier( 
            criterion = "entropy", random_state = 100, 
            max_depth = 7, min_samples_leaf = 5) 
  
    # Performing training 
    clf_entropy.fit(X_train, y_train) 
    return clf_entropy 
#  
#  
# Function to make predictions 
def prediction(X_test, clf_object): 
  
    # Predicton on test with giniIndex 
    y_pred = clf_object.predict(X_test) 
    print("Predicted values:") 
    print(y_pred) 
    return y_pred 
      
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred): 
      
    print("Confusion Matrix:\n ", 
        confusion_matrix(y_test, y_pred)) 
      
    print ("Accuracy :\n ", 
    accuracy_score(y_test,y_pred)*100) 
      
    print("Report :\n ", 
    classification_report(y_test, y_pred)) 
  
    
def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    
# Driver code 
def main(): 
      
    # Building Phase 
    data = importdata() 
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data) 
    clf_gini = train_using_gini(X_train, X_test, y_train) 
    clf_entropy = train_using_entropy(X_train, X_test, y_train) 
#      
    # Operational Phase 
    print("Results Using Gini Index:") 
      
    # Prediction using gini 
    y_pred_gini = prediction(X_test, clf_gini) 
    cal_accuracy(y_test, y_pred_gini) 
      
    print("Results Using Entropy:") 
    # Prediction using entropy 
    y_pred_entropy = prediction(X_test, clf_entropy) 
    cal_accuracy(y_test, y_pred_entropy) 
    
    from subprocess import call
    class__names = ['Not Faulty', 'Faulty']
    featurenames = ['loc', 'v(g)', 'ev(g)', 'iv(g)', 'n', 'v', 'l', 'd', 'i', 'e',
         'b', 't', 'lOCode', 'lOComment', 'lOBlank',
         'lOCodeAndComment', 'uniq_Op', 'uniq_Opnd', 'total_Op', 'total_Opnd']
    dot_data = StringIO()
    export_graphviz(clf_gini, out_file='tree.dot', 
                feature_names = featurenames,
                #class_names = class__names,
                rounded = True, proportion = False, 
                precision = 2, filled = True)

    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
    #graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    plt.figure(figsize = (14, 18))
    plt.imshow(plt.imread('tree.png'))
    plt.axis('off');
    plt.show();
    #Plotting ROC
#    probs = clf_gini.predict_proba(X_test)
#    fpr, tpr, thresholds = roc_curve(y_test, probs)
#    plot_roc_curve(fpr,tpr)
##  
    
      
# Calling main function 
if __name__=="__main__": 
    main() 