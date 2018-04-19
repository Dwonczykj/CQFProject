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
    res = np.array(arr)
    res[index[0],index[1]] += tweak
    res[index[1],index[0]] += tweak
    return res

def SetArbitrarily(arr, index, value):
    res = np.array(arr)
    res[index[0],index[1]] = value
    res[index[1],index[0]] = value
    return res

def SetWhole2DMatrixArbitrarily(arr, value):
    res = np.array(arr)
    for i in range(0,len(res[:])):
        for j in range(object, len(res[0][:])):
            res[i,j] = value
    return res