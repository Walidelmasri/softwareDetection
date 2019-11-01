#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 18:59:15 2019
for jm1, i had to copy paste the file onto my laptop and get rid of rows that had unknown
@author: trvpo333
"""
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
import sklearn.metrics as metrics

# Function importing Dataset 
def importdata(): 

    url = "http://promise.site.uottawa.ca/SERepository/datasets/jm1.arff"
    names = ['loc', 'v(g)', 'ev(g)', 'iv(g)', 'n', 'v', 'l', 'd', 'i', 'e',
         'b', 't', 'lOCode', 'lOComment', 'lOBlank', 'lOCodeAndComment', 'uniq_Op', 'uniq_Opnd', 'total_Op', 'total_Opnd'
         , 'branchCount', 'defects']
    #kc2 skip 358 and no yasint
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

# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred): 
      
    print("Confusion Matrix:\n ", 
        confusion_matrix(y_test, y_pred)) 
      
    print ("Accuracy :\n ", 
    accuracy_score(y_test,y_pred)*100) 
      
    print("Report (Prediction using Naive-Bayes):\n ", 
    classification_report(y_test, y_pred)) 
  
    
# Driver code 
def main(): 
      
    # Building Phase 
    data = importdata() 
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data) 
    #class__names = ['Not Faulty', 'Faulty']
    #Create a Gaussian Classifier
    gnb = GaussianNB()
    #Train the model using the training sets
    gnb.fit(X_train, y_train)
    #Predict the response for test dataset
    y_pred = gnb.predict(X_test)
    cal_accuracy(y_test, y_pred) 
    print("Predicted Values:\n")
    print(y_pred)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    
    true = "true\\"

    for x in range(3264):
        if y_test[x] == true:
            y_test[x] = 1
        else: 
            y_test[x] = 0 
  
    # calculate the fpr and tpr for all thresholds of the classification
    probs = gnb.predict_proba(X_test)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)
    
    # method I: plt
    import matplotlib.pyplot as plt
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

# Calling main function 
if __name__=="__main__": 
    main() 