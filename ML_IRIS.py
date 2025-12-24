import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import zero_one_loss

def load_data():
    dataset=pd.read_csv("D:\iris.data",header=None,
                        names=['septal lenght','septal width','petal lenght','petal width'])
    data=dataset.iloc[:,:4]
    label=dataset.iloc[:,-1]
    xtrain,xtest,ytrain,ytest=train_test_split(data,label,test_size=0.2)
    return xtrain,xtest,ytrain,ytest
def training():
    neigh=KNeighborsClassifier(n_neighbors=5)
    neigh.fit(xtrain,ytrain)
    return neigh
def result():
    ypred=neig.predict(xtest)
    acc=accuracy_score(ytest,ypred)
    print("accuracy:{:.2f}".format(acc*100))
    loss=zero_one_loss(ytest,ypred)
    print("loss:{:.2f}".format(loss*100))

xtrain,xtest,ytrain,ytest=load_data()   
neig=training() 
result()



