import numpy as np
import csv
from numpy import choose
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import neighbors
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def classify(x_train,y_train,choose,argument):
    if(choose==0):
        Classifier = svm.SVC(C=argument,kernel='sigmoid')
    elif(choose==1):
        Classifier = neighbors.KNeighborsClassifier(n_neighbors=argument)
    else:
        print(choose,' not available choice')
        return
    Classifier.fit(x_train,y_train)
    return Classifier


#return accuracy of Classifier
def test(Classifier,X_test,y_test):
    y_pred = Classifier.predict(X_test)
    #print(y_pred)
    #print('Train Accuracy: {:.2f} %'.format(np.mean(y_pred == y_test) * 100))
    return np.mean(y_pred == y_test)*100


#predict the output given dataset print it
def predict(Classifier,X_test):
    y_pred = Classifier.predict(X_test)
    return(y_pred)

#read file into training, cross validation and test data
def read_data(file_name):
    
    #read file
    file = open(file_name)
    csvreader = csv.reader(file)
    rows = []
    for row in csvreader:
        rows.append(row)
    rows=np.array(rows)
    x=rows[1:,3:90].astype(float)
    y=rows[1:,1].astype(float)
    
    # 60% training 20% cross-validation 20% test
    X_train, x_valid, y_train,y_valid = train_test_split(x, y, test_size=0.4,random_state=109)
    return X_train, x_valid,y_train,y_valid





def get_Classification(X_train,y_train,estimator):

    regressor = RandomForestClassifier(n_estimators=estimator, random_state=0)
    regressor.fit(X_train, y_train)
    return regressor 

def randomForestPredict(regressor,x_test):
    y_pred = regressor.predict(x_test)
            
    return y_pred

def get_best_parameter(X_train,y_train,x_cross_validation,y_cross_validation):
# cross validation set(to determine the parameters)
    Accuracy = 0
    estimator = 0
    for i in range(1,300):
        regressor = RandomForestClassifier(n_estimators=i, random_state=0)
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(x_cross_validation)
        new_accuracy = round(accuracy_score(y_cross_validation, y_pred) * 100,2)
        if Accuracy < new_accuracy :
            Accuracy = new_accuracy
            estimator = i
    return estimator


def readData(file_name):
    
    #read file
    file = open(file_name)
    csvreader = csv.reader(file)
    rows = []
    for row in csvreader:
        rows.append(row)
    rows=np.array(rows)
    x=rows[1:,3:].astype(float)
    y=rows[1:,1].astype(float)
    
    # 60% training 20% cross-validation 20% test
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4,random_state=0)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.01,random_state=0)
    return X_train, X_test,X_valid,y_valid, y_train, y_test