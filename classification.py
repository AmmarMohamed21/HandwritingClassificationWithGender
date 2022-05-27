import numpy as np
import csv
from numpy import choose
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import neighbors

#choose: 0 for svm 1 for knn
#argument: if svm then c as integer if knn then choose k as an integer
#return Classifier with fit data to predict new ones
def classify(x_train,y_train,choose,argument):
    if(choose==0):
        Classifier = svm.SVC(C=argument,kernel='linear')
    elif(choose==1):
        Classifier = neighbors.KNeighborsClassifier(n_neighbors=argument)
    else:
        print(choose,' not available choice')
        return
    Classifier.fit(x_train,y_train)
    return Classifier

def test(Classifier,X_test,y_test):
    y_pred = Classifier.predict(X_test)
    print(y_pred)
    print('Train Accuracy: {:.2f} %'.format(np.mean(y_pred == y_test) * 100))

def predict(Classifier,X_test):
    y_pred = Classifier.predict(X_test)
    print(y_pred)

def read_data(file_name):
    
    #read file
    file = open(file_name)
    csvreader = csv.reader(file)
    rows = []
    for row in csvreader:
        rows.append(row)
    rows=np.array(rows)
    x=rows[1:,2:-1].astype(float)
    y=rows[1:,-1].astype(int)
    
    # 70% training and 30% test
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=109)
    
    return X_train, X_test, y_train, y_test

print('output with svm with linear kernel and c=1')
x_train, x_test, y_train, y_test=read_data('dataset.csv')
Classifier=classify(x_train,y_train,0,1)
test(Classifier,x_test,y_test)
print()
print('output with knn with k=20')
x_train, x_test, y_train, y_test=read_data('dataset.csv')
Classifier=classify(x_train,y_train,1,20)
test(Classifier,x_test,y_test)