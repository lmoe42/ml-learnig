#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 16:55:56 2019

@author: moratz
"""

def mlpEarlystoppingPrunedPlot():
    import numpy as np
    import matplotlib.pyplot as pl
    import mlp
    
    gridResolution = 1000
    
    pima = np.loadtxt('pima.csv',delimiter=',')
    targets = pima[:,8:9] 
    nData = np.shape(targets)[0]
    print ("nData ",nData)
    inputs = pima[:,:8]
    inputs = (inputs - inputs.mean(axis=0))/inputs.std(axis=0)
    
    x = []
    y = []
    xDiabetes = []
    yDiabetes = []
    
    for i in range(nData):
        inputs[i,0] = 0  
        inputs[i,2] = 0
        inputs[i,3] = 0
        inputs[i,4] = 0
        inputs[i,6] = 0
        inputs[i,7] = 0
        if targets[i] == 0:
            x.append(inputs[i, 1])
            y.append(inputs[i, 5])
        else:
            xDiabetes.append(inputs[i, 1])
            yDiabetes.append(inputs[i, 5])
        
    
    trainIn = inputs[::2,:]
    trainTgt = targets[::2,:]
    testIn = inputs[1::2,:]
    testTgt = targets[1::2,:]
    
    net = mlp.mlp(trainIn, trainTgt, 2)
    net.earlystopping(trainIn, trainTgt,testIn,testTgt, 0.01)
    net.confmat(trainIn,trainTgt)
    net.confmat(testIn,testTgt)
    
    rasterInputs = np.zeros((gridResolution*gridResolution,8))
    for i in range(gridResolution):
        for j in range(gridResolution):
            rasterInputs[i*gridResolution+j, 1] = -3.0 + 6.0 * i / gridResolution
            rasterInputs[i*gridResolution+j, 5] = -3.0 + 6.0 * j / gridResolution
    
    #print(rasterInputs)
    
    confinputs = rasterInputs
    confinputs = np.concatenate((confinputs,-np.ones((np.shape(confinputs)[0],1))),axis=1)
    outputs = net.mlpfwd(confinputs)
    
    #print(outputs)
 
    xBorder = []
    yBorder = []
    for i in range(gridResolution):
        for j in range(gridResolution):
            testValue = outputs[i*gridResolution+j]
            if 0.48 < testValue and 0.52 > testValue:
                  xBorder.append(rasterInputs[i*gridResolution+j, 1])
                  yBorder.append(rasterInputs[i*gridResolution+j, 5])
    
    pl.ion()
    pl.figure()
    #x = [1,2,3]
    #y = [1,2,3]
    pl.plot(xBorder, yBorder,'.')
    pl.plot(x, y,'gx')
    pl.plot(xDiabetes, yDiabetes,'ro')
    pl.show()
    

    
        