# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
#import pylab as pl

# Task 1.1
def gen_sinusoidal(N):
    x = np.linspace (0,2 * np.pi)
    
    for i in np.linspace (0, 2*np.pi, N/2 * np.pi):
        plt.plot(i, np.sin(i) + np.random.normal(-0.1, 0.1), 'o')
    plt.plot(x, np.sin(x))
    
gen_sinusoidal(10)

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
    #print w
    #print y
    a=np.linalg.inv(w)
    #print a
    b=np.transpose(y)
    result = np.dot(a,b)
    return np.transpose(result)
    
    
# data to fit
#x = [0.0,1.0,2,3,4,5]
#y = [0,0.8,0.9,0.1,-0.8,-1]
#print (fit_polynomial(x,y,3))