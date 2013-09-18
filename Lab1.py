# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

# Task 1.1
def gen_sinusoidal(N):
    x = np.linspace(0,2 * np.pi, N)   
    for i in np.linspace (0, 2*np.pi, N):
        plt.plot(i, np.sin(i) + np.random.normal(-0.1, 0.1), 'o')
    plt.plot(x, np.sin(x))
    plt.show()
#gen_sinusoidal(10)

# Task 1.2
def fit_polynomial(x,t,M):
    w=[]
    y=[]
    for i in range(0,M+1):
        w.append([])
        if(i==0):
            y.append([])
        for j in range (0,M+1):
            w[i].append(j)
            if(i==0):
                y[i].append(j)
            w[i][j]=0
            if(i==0):
                y[0][j]=0
            for k in range(0,len(x)):
                w[i][j]+=pow(x[k],i+j)
                if(i==0):
                    y[i][j] += t[k]*pow(x[k],j)
    a=np.linalg.inv(w)
    b=np.transpose(y)
    result = np.dot(a,b)
    return np.transpose(result)
#x = [0.0,1.0,2,3,4,5]
#y = [0,0.8,0.9,0.1,-0.8,-1]
#print (fit_polynomial(x,y,3))

#Task 1.3
def generateSampleDataSet(N):
    x = np.linspace(0,2*np.pi, N) 
    retx = []
    rety = []
    for i in x:
        retx.append(i)
        rety.append(np.sin(i) + np.random.normal(-0.1, 0.1))
    retx = np.array([retx])
    rety = np.array([rety])
    return np.concatenate((retx, rety), axis = 0)

def fit_polynominals() :
    x = np.linspace(0,2 * np.pi)
    
    M = [0, 1, 3, 8]
    sampleData = generateSampleDataSet(9)
    sampleX = sampleData[0]
    sampleY = sampleData[1]
    
    fig = plt.figure()
    subplot = 1
    for m in M:    
        fig.add_subplot(2,2,subplot)
        plt.plot(x, np.sin(x))
        for i in range (0, np.size(sampleX)):
            plt.plot(sampleX[i], sampleY[i], 'o')
        poly = fit_polynomial(sampleX, sampleY, m)
        pol = np.polynomial.Polynomial(poly[0])
        plt.plot(x, pol(x), 'r')
        subplot = subplot + 1
    plt.show()
fit_polynominals()





