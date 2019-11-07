# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 22:13:24 2019

3 input neurons which take in normalized values of rgb value of color input. 
hidden layer has 4 neurons (no logic behind having a 2nd layer )
output layer has 2 nurons. If index 0 is high network suggests white otherwise black [1,0] white ; [0,1] black

@author: Abhishek
"""
import numpy as np
import time
import matplotlib.pyplot as plt

np.random.seed(1)

def sig(x):
    return 1/(1+(2.7**(-1*x)))

def sigd(x):
    return sig(x)*sig(1-x)

def feed(w,d,b):
    return d.dot(w)+b

def cost(eo,x3):
    return np.average((x3-eo)**2)

def backprop(eo,x3,x2,x1,w2):
    db2=2*(x3-eo)*x3*(1-x3)
    dw2=db2.T.dot(x2).T
    db1=db2.dot(w2.T)*x2*(1-x2)
    dw1=db1.T.dot(x1).T
    return dw1, dw2 , db1, db2

'''
c=(x3-eox)**2
x3=sig(z3)
z3=wx2+b2
x2=sig(z2)
z2=wx1+b1


'''

def neuralnet(traind,eo,it=1000,stp=0.2): #does not overfit at 1270 iterations
    w1=np.random.normal(0,1,(3,4))
    w2=np.random.normal(0,1,(4,2))
    b1=np.zeros((1,4))
    b2=np.zeros((1,2))
    crap=np.zeros(it)
    st=time.time()
    for i in range(it):
        c=np.zeros(np.shape(traind)[0])
        dw1=np.zeros((np.shape(traind)[0],3,4))
        dw2=np.zeros((np.shape(traind)[0],4,2))
        db1=np.zeros((np.shape(traind)[0],1,4))
        db2=np.zeros((np.shape(traind)[0],1,2))
        count=0
        for x1 ,eox in zip(traind,eo):
            eox=np.reshape(eox,(1,2))
            x1=np.reshape(x1,(1,3))
            x2=sig(feed(w1,x1,b1))
            x3=sig(feed(w2,x2,b2))
            c[count]=cost(eox,x3)
            dw1[count], dw2[count], db1[count], db2[count] = backprop(eox,x3,x2,x1,w2)
            count+=1
        w1-=np.sum(dw1,axis=0)*stp
        w2-=np.sum(dw2,axis=0)*stp
        b1-=np.sum(db1,axis=0)*stp
        b2-=np.sum(db2,axis=0)*stp
        crap[i]=np.average(c)
    plt.grid()
    plt.plot(crap)
    #plt.ylim(0,0.05)
    plt.show()
    print(time.time()-st, "s", end=' ')
    print('')
    return w1, w2, b1, b2

def accu(w1, w2, b1, b2,  test, teo):
    accur=0
    for x1,y1 in zip(test,teo):
        y1=np.reshape(y1,(1,2))
        x1=np.reshape(x1,(1,3))
        x2=sig(feed(w1,x1,b1))
        x3=sig(feed(w2,x2,b2))
        print(x3,y1)
        if x3[0,0] > x3[0,1] and y1[0,0] > y1[0,1]:   #super inaccurate accuracy funstion improve on this pls
            accur+=1
        if x3[0,0] < x3[0,1] and y1[0,0] < y1[0,1]:
            accur+=1
    print(accur/np.shape(test)[0])
        
            

traind=np.array([[255,128,128],
        [255,255,128],
        [128,255,128],
        [128,255,255],
        [0,128,255],
        [225,128,192],
        [255,128,255],
        [255,0,0],
        [255,255,0],
        [0,255,128],
        [128,255,0],
        [64,0,64],
        [192,192,192],
        [255,0,128],
        [128,64,0],
        [64,0,0],
        [64,128,128],
        [128,128,0],
        [0,0,0]])
        
eo=np.array([[1,0],
             [0,1],
             [0,1],
             [0,1],
             [1,0],
             [1,0],
             [1,0],
             [0,1],
             [0,1],
             [1,0],
             [1,0],
             [1,0],
             [0,1],
             [1,0],
             [1,0],
             [1,0],
             [0,1],
             [1,0],
             [1,0]])
    
test=np.array([[255,128,64],
               [128,128,255],
               [0,0,255],
               [128,0,64],
               [255,255,255],
               [0,0,0],
               [142,142,142],
               [150,150,85]])
    
teo=np.array([[0,1],
              [1,0],
              [1,0],
              [1,0],
              [0,1],
              [1,0],
              [1,0],
              [0,1]])
    
w1, w2, b1, b2 = neuralnet(traind/255,eo)
accu(w1, w2, b1, b2, test/255, teo)

#print(np.shape(traind))