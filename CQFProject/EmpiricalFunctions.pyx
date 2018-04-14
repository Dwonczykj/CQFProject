import math
import numpy as np
from Returns import *
from Sorting import *
import collections
cimport numpy as np
DTYPE = np.int
FDTYPE = np.float

def check(a, i, step, xpr):
    if len(xpr) > i+step and xpr[i+step] == a:
        return check(a, i+step, step, xpr)
    else:
        return i

def Empirical_StepWise_CDF(Ordrdx):
    def resFn(a):
        if a > max(Ordrdx):
            return 1.0 
        if a < min(Ordrdx):
            return 0.0
        def innerFn(a,xpr,lb,ub):
            i = int((len(xpr) / 2))
            if xpr[i-1] < a and xpr[i] < a:
                return innerFn(a,xpr[i:],lb + i,ub) if i != len(Ordrdx) else 1
            elif xpr[i] > a and xpr[i-1] > a:
                return innerFn(a,xpr[0:i],lb,lb + i-1) if i != 1 else 0
            elif xpr[i-1] == a:
                return (lb + i) / len(Ordrdx) if xpr[i] > a else (lb + check(a, i, 1, xpr) + 1) / len(Ordrdx)
            else:
                return (lb + check(a,i,1,xpr) + 1) / len(Ordrdx)
        return innerFn(a,Ordrdx,0,len(Ordrdx)-1)
    return resFn
    #return the EmpiricalCDF function from this function.
def ApproxPWCDFDicFromHazardRates(l,step = 0.01):
    cdef float i
    def P(T):
        ls = np.asarray(l,dtype=FDTYPE)
        keys = np.arange(0,len(ls),1,dtype=DTYPE) #!change this if ever change the time periods from years os cant use the index of the array to count periods unless 1 period = 1 year
        s = keys[1] - keys[0]
        kT = max(next( (key for key in keys if T >= (key-s) and T <= (key)), max(keys)) - 1,0)
        p = 1 - math.exp(-(sum(ls[0:kT]) + ls[kT]*(T - kT)))
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



cdef float fnVals(float a,np.ndarray U,res):
    cdef int i
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

cdef np.ndarray fKeys(a,np.ndarray U):
    cdef int i
    i = len(U) / 2 if len(U) % 2 == 0 else (len(U)-1) / 2
    i = int(i)
    if U[i-1] < a and U[i] < a:
        return fKeys(a,U[i:]) if len(U) - i > 1 else np.asarray([max(U)])
    elif U[i] > a and U[i-1] > a:
        return fKeys(a,U[0:i]) if i > 1 else np.asarray([min(U)])
    elif U[i-1] == a or U[i] == a:
        return np.asarray([a])
    else:
        return np.asarray([U[i-1],U[i]])

def FindClosestKeyInDicAndReturnValueAlgorithm(Res):
    ORes = collections.OrderedDict(sorted(Res.items()))
    def InnerFn(float u):
        return fnVals(u,np.asarray(list(ORes.keys())),ORes)
    return InnerFn

def FindClosestKeyInDicAndReturnKeyBoundsAlgorithm(Res):
    ORes = collections.OrderedDict(sorted(Res.items()))
    def FN(u):
        return fKeys(u,np.asarray(list(ORes.keys())))
    return FN

