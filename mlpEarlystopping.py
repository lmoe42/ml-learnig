#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 16:55:56 2019

@author: moratz
"""
import pylab as pl

def mlpEarlystopping():
    import numpy as np
    import mlp
    pima = np.loadtxt('pima.csv',delimiter=',')
    targets = pima[:,8:9] 
    nData = np.shape(targets)[0]
    print ("nData ",nData)
    # inputs = pima[:,:8]
    best_rate = 0
    i_param = 0
    j_param = 1

    for i in range(8):
        for j in range(i+1,8):
            inputs = pima[:,[i,j]]
            inputs = (inputs - inputs.mean(axis=0))/inputs.std(axis=0)
            trainIn = inputs[::2,:]
            trainTgt = targets[::2,:]
            testIn = inputs[1::2,:]
            testTgt = targets[1::2,:]
            net = mlp.mlp(trainIn, trainTgt, 2)
            net.earlystopping(trainIn, trainTgt,testIn,testTgt, 0.015)
            net.confmat(trainIn,trainTgt)
            current_percentage = net.confmat(testIn,testTgt)
            print("current rate", current_percentage)
            print("current indices", i, j)
            if current_percentage > best_rate:
                i_param = i
                j_param = j
                best_rate = current_percentage

    print ("best_rate: ", best_rate)
    print("decisive parameters: ", i_param, j_param) 
    indices0 = np.where(pima[:,8]==0)
    indices1 = np.where(pima[:,8]==1)
    
    #pl.ion()
    pl.plot(pima[indices0,j_param],pima[indices0,i_param],'go')
    pl.plot(pima[indices1,j_param],pima[indices1,i_param],'rx') 
    pl.show()    
    print(pl.show())

mlpEarlystopping()