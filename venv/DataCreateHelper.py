from random import randint
from random import random
from random import uniform
from math import pi
from math import sin
from math import exp
from numpy import array
from numpy import ndarray
from numpy import shape
from numpy import argmax
from numpy import empty
from matplotlib import pyplot

class DataCreate:
    def printdatainfo(self,object):
        if type(object)==list or type(object)==tuple:
            print("type: list","len:",len(object))
        elif type(object)== ndarray:
            print("type: array",shape(object))
        else:
            print(type(object))
    def plotonesequence(self,showsequence,labelname=None):
        pyplot.plot(showsequence,'-o',label=labelname )
        pyplot.legend()
        pyplot.show()
    def preparedata(self,traincount,evaluatecount,func,**kwargs):
        X_train=list()
        y_train=list()
        for _ in range(traincount):
            X,y=func(**kwargs)
            X_train.append(X)
            y_train.append(y)
        kwargs['n_patterns']=evaluatecount
        X_predict,y_preidct= func(**kwargs)
        return X_train,y_train,X_predict,y_preidct

    def decode_vanillaLSTM(self,onedimensionarray,encodetuple):
        returnlist= list()
        for i in range(len(onedimensionarray)):
            j=argmax(onedimensionarray[i])
            returnlist.append(encodetuple[j])
        return returnlist
    def generateexample_vanillaLSTM(self,n_patterns,length,n_features,out_index):
        sequence=[randint(0,n_features-1) for _ in range(length)]
        returnX=empty([n_patterns,length,n_features],int)
        returnY=empty([n_patterns,n_features] , int)
        for j in range(n_patterns):
            onehotsequence = list()
            for tmpfeature in sequence:
                vector = [0 for _ in range(n_features)]
                vector[tmpfeature] = 1
                onehotsequence.append(vector)
            onehotsequence = array(onehotsequence)
            X = onehotsequence.reshape(1, length, n_features)
            y = onehotsequence[out_index].reshape(1, n_features)
            returnX[j]=X[0]
            returnY[j]=y[0]
        return returnX , returnY
    def generateexample_cnnlstm(self,n_patterns,timesteps,width,height,channels):
        X,y=list(),list()

        return X,y
    def generateexample_stacklstm(self,n_patterns,length,n_features,out_count):
        X,y=list(),list()
        for _ in range(n_patterns):
            p=randint(10,20)
            d=uniform(0.01,0.1)
            sequence=[0.5+0.5*sin(2*pi*i/p)*exp(-d*i) for i in range(length+out_count)]
            X.append(sequence[:-out_count])
            y.append(sequence[-out_count:])
        X=array(X).reshape(n_patterns,length,n_features)
        y=array(y).reshape(n_patterns,out_count)
        return X,y

if __name__=="__main__":
    mydatacreate=DataCreate()
    X_train, y_train = generateexample_cnnlstm()
