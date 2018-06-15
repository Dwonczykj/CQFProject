import numpy as np

def quickSort(arr):
    less = []
    pivotList = []
    more = []
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[0]
        for i in arr:
            if i < pivot:
                less.append(i)
            elif i > pivot:
                more.append(i)
            else:
                pivotList.append(i)
        less = quickSort(less)
        more = quickSort(more)
        return less + pivotList + more

def Tweak(arr, index, tweak):
    '''
    Tweak a correlation in arr at index by tweak
    '''
    res = np.array(arr)
    res[index[0],index[1]] += tweak
    res[index[1],index[0]] += tweak
    return res

def FisherTransform(p):
    return 0.5*np.log((1+p)/(1-p))

def InvFisherTransform(Z):
    return (np.exp(2*Z)-1)/(np.exp(2*Z)+1)

def TweakByPercent(arr, index, percentTweak):
    '''
    Tweak a correlation in arr at index by mutiplying by percentTweak, i.e. percentTweak = 0.5 for 50% tweak
    '''
    res = np.array(arr)
    FisherP = FisherTransform(res[index[0],index[1]])
    FisherP *= (1+np.min([np.max([percentTweak,-1]),1]))
    res[index[0],index[1]] = InvFisherTransform(FisherP)
    res[index[1],index[0]] = res[index[0],index[1]]
    return res

def TweakWhole2DMatrixByPercent(arr,percentTweak):
    '''
    Tweak all correlations in arr by mutiplying by percentTweak, i.e. percentTweak = 0.5 for 50% tweak
    '''
    res = np.array(arr)
    for i in range(0,arr.shape[0]):
        for j in range(i+1,arr.shape[1]):
            res[i][j] = TweakByPercent(arr,(i,j),percentTweak)[i][j]
            res[j][i] = res[i][j]
    return res

def SetArbitrarily(arr, index, value):
    res = np.array(arr)
    res[index[0],index[1]] = value
    res[index[1],index[0]] = value
    return res

def SetWhole2DMatrixArbitrarily(arr, value):
    res = np.array(arr)
    for i in range(0,len(res[:])):
        for j in range(0, len(res[0][:])):
            res[i,j] = value
    return res