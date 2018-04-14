import math
import numpy as np
import pandas as pd
from Returns import *
from Sorting import *
import collections

def Empirical_StepWise_CDF(Ordrdx: pd.Series):
    #mu = mean(unOrdrdx)
    #S = sd(unOrdrdx)
    #order the data and then set the prob by how many values 
    #OrderedX = pd.Series(x).sort_values()
    #x = quickSort(x)
    #def resFn(p):
    #    for i in range(0,len(x)-1):
    #        if p >= OrderedX[i] and p <= OrderedX[i+1]:
    #            return (i+1)/len(x)
    #    return -1
    def resFn(a):
        def innerFn(a,xpr,lb,ub):
            i = len(xpr) / 2 if len(xpr) % 2 == 0 else (len(xpr)-1) / 2
            i = int(i)
            if xpr[i-1] < a and xpr[i] < a:
                return innerFn(a,xpr[i:],i,ub) if i != len(Ordrdx) else 1
            elif xpr[i] > a and xpr[i-1] > a:
                return innerFn(a,xpr[0:i],lb,i-1) if i != 1 else 0
            elif xpr[i-1] == a:
                return (lb + i - 1) / len(Ordrdx) 
            else:
                return (lb + i) / len(Ordrdx)
        return innerFn(a,Ordrdx,0,len(Ordrdx)-1)
    return resFn
    #return the EmpiricalCDF function from this function.
def ApproxPWCDFDicFromHazardRates(l,step = 0.01):
    def P(T):
        ls = pd.Series(l)
        keys = ls.index #!change this if ever change the time periods from years os cant use the index of the array to count periods unless 1 period = 1 year
        s = keys[1] - keys[0]
        kT = max(next( (key for key in keys if T >= (key-s) and T <= (key)), max(keys)) - 1,0)
        p = math.exp(-(sum(ls[0:kT]) + ls[kT]*(T - kT)))
        return p
    res = dict()
    for i in np.arange(0,1,step/100):
        res[P(i)] = i
    for i in np.arange(1,10,step):
        res[P(i)] = i
    for i in np.arange(10,100,step*100):
        res[P(i)] = i
    #return res
    #def InnerFn(u):
    #    T = res.values()
    #    def fn(a,U):
    #        i = len(U) / 2 if len(U) % 2 == 0 else (len(U)-1) / 2
    #        i = int(i)
    #        if U[i-1] < a and U[i] < a:
    #            return innerFn(a,U[i:]) if len(U) != 2 else max(T)
    #        elif U[i] > a and U[i-1] > a:
    #            return innerFn(a,U[0:i]) if len(U) != 2 else 0
    #        elif U[i-1] == a:
    #            return res[U[i-1]]
    #        else
    #            return res[U[i]]
    #    return fn(u,res.keys())
    #return InnerFn
    return FindClosestKeyInDicAndReturnValueAlgorithm(res)



def fnVals(a,U,res):
    i = len(U) / 2 if len(U) % 2 == 0 else (len(U)-1) / 2
    i = int(i)
    if U[i-1] < a and U[i] < a:
        return fnVals(a,U[i:],res) if len(U) - i > 1 else max(res.values())
    elif U[i] > a and U[i-1] > a:
        return fnVals(a,U[0:i],res) if i > 1 else min(res.values())
    elif U[i-1] == a:
        return res[U[i-1]]
    else:
        return res[U[i]]

def fKeys(a,U):
    i = len(U) / 2 if len(U) % 2 == 0 else (len(U)-1) / 2
    i = int(i)
    if U[i-1] < a and U[i] < a:
        return fKeys(a,U[i:]) if len(U) - i > 1 else max(U)
    elif U[i] > a and U[i-1] > a:
        return fKeys(a,U[0:i]) if i > 1 else min(U)
    elif U[i-1] == a or U[i] == a:
        return [a]
    else:
        return [U[i-1],U[i]]

def FindClosestKeyInDicAndReturnValueAlgorithm(Res):
    ORes = collections.OrderedDict(sorted(Res.items()))
    def InnerFn(u):
        return fnVals(u,list(ORes.keys()),ORes)
    return InnerFn

def FindClosestKeyInDicAndReturnKeyBoundsAlgorithm(Res):
    ORes = collections.OrderedDict(sorted(Res.items()))
    def FN(u):
        return fKeys(u,list(ORes.keys()))
    return FN

