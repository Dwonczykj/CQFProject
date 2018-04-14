import math
import numpy as np
from Returns import var

def RunningAverage(Arr: np.array):
    res = np.zeros(shape=Arr.shape)
    res[0] = Arr[0]
    for i in range(1,res.shape[0]):
        res[i] = ((res[i-1] * i) + Arr[i]) / (i+1)
    return res.transpose()

def RunningVarianceOfRunningAverage(Arr:np.array,sampleSize):
    res = np.zeros(shape=(Arr.shape[0],math.ceil(Arr.shape[1]/sampleSize)),dtype=np.float)
    i=0
    k = Arr.shape[0]
    iterationNo = 0
    while i < Arr.shape[1]:
        ub = min(i+sampleSize,Arr.shape[1])
        lb = max(0,ub - sampleSize)
        for j in range(0,k):
            res[j,iterationNo] = var(Arr[j,lb:ub])
        iterationNo += 1
        i += sampleSize
    return res