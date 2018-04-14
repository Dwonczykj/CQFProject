import numpy as np
import pandas as pd
import math
from scipy import linalg as LA
from scipy.optimize.nonlin import KrylovJacobian

def LogReturns(values: pd.Series, lag: np.int = 1, jump: np.int = 1, averageJumpPeriod = False):
    '''
    Pass in a series of values and calculate log returns by taking the return of every jump^th value 
    with the value lag periods before it and then taking the log of that return.
    '''
    quotRtn = Returns(values,lag,jump,averageJumpPeriod)
    logRtns = quotRtn.apply(func=math.log)
    return logRtns

def Returns(values: pd.Series, lag: np.int = 1, jump: np.int = 1, averageJumpPeriod = False):
    '''
    Pass in a series of values and calculate returns by taking the return of every jump^th value 
    with the value lag periods before it.
    '''
    if len(values) / jump <= lag:
        return None
    else:
        #absRtns = pd.Series(pd.Series(values[lag:].values,index=values.index[:-lag]) - values[:-lag].values) / values[:-lag].values
        #if lag == 1:
        #    subvalues = values[::jump].values
        #    quotRtn = pd.Series(values[lag::jump].values,index=values.index[:-lag:jump]) / values[:-lag:jump].values
        #else:
        #    quotRtn = pd.Series(values[lag:].values,index=values.index[:-lag]) / values[:-lag].values
        if averageJumpPeriod:
            av = np.fromiter(map( lambda ar: mean(ar), np.array_split(values[lag:].values,int((len(values)-lag)/jump))),dtype=np.float)
            av2 = np.fromiter(map( lambda ar: mean(ar), np.array_split(values[:-lag].values,int((len(values)-lag)/jump))),dtype=np.float)
            quotRtn = pd.Series(av, index=values.index[:-lag:jump][:len(av)]) / av2
        else:
            quotRtn = pd.Series(values[lag::jump].values,index=values.index[:-lag:jump]) / values[:-lag:jump].values
        return quotRtn

def AbsoluteReturns(values: pd.Series, lag: np.int = 1, jump: np.int = 1, averageJumpPeriod = False):
    '''
    Pass in a series of values and calculate Absolute returns by taking the return of every jump^th value 
    less the value lag periods before it all divided by the value lag periods before it.
    '''
    if len(values) / jump <= lag:
        return None
    else:
        if averageJumpPeriod:
            av = np.fromiter(map( lambda ar: mean(ar), np.array_split(values[lag:].values,int((len(values)-lag)/jump))),dtype=np.float)
            av2 = np.fromiter(map( lambda ar: mean(ar), np.array_split(values[:-lag].values,int((len(values)-lag)/jump))),dtype=np.float)
            absRtns = pd.Series(pd.Series(av, index=values.index[:-lag:jump][:len(av)]) - av2) / av2
        else:
            absRtns = pd.Series(pd.Series(values[lag::jump].values,index=values.index[:-lag:jump]) - values[:-lag:jump].values) / values[:-lag:jump].values
        return absRtns

def AbsoluteDifferences(values: pd.Series, lag: np.int = 1, jump: np.int = 1, averageJumpPeriod = False):
    if len(values) / jump <= lag:
        return None
    else:
        if averageJumpPeriod:
            av = np.fromiter(map( lambda ar: mean(ar), np.array_split(values[lag:].values,int((len(values)-lag)/jump))),dtype=np.float)
            av2 = np.fromiter(map( lambda ar: mean(ar), np.array_split(values[:-lag].values,int((len(values)-lag)/jump))),dtype=np.float)
            absRtns = pd.Series(pd.Series(av, index=values.index[:-lag:jump][:len(av)]) - av2)
        else:
            absRtns = pd.Series(pd.Series(values[lag::jump].values,index=values.index[:-lag:jump]) - values[:-lag:jump].values)
        #absRtns -= mean(absRtns)
        return absRtns

def mean(x):
    return float(sum(x)) / len(x)

def pairWiseCov(x,y):
 # Assume len(x) == len(y)
  n = len(x)
  sum_x = float(sum(x))
  sum_y = float(sum(y))
  #sum_x_sq = sum(map(lambda x: pow(x, 2), x))
  #sum_y_sq = sum(map(lambda x: pow(x, 2), y))
  psum = sum(map(lambda x, y: x * y, x, y)) #dotproduct
  num = (psum - (sum_x * sum_y)/n)/n
  return num

def var(x):
    # Assume len(x) == len(y)
  n = len(x)
  sum_x = float(sum(x))
  sum_x_sq = sum(map(lambda x: pow(x, 2), x))
  #return (sum_x_sq - pow(sum_x/n, 2)) / n
  return np.var(x)

def sd(x):
    return pow(var(x),0.5)

def StandardisedResiduals(values:pd.Series):
    mu = mean(values)
    s = sd(values)
    residuals = (values - mu) / s
    return residuals


def pearsonr(x, y):
  # Assume len(x) == len(y)
  #n = len(x)
  #sum_x = float(sum(x))
  #sum_y = float(sum(y))
  #sum_x_sq = sum(map(lambda x: pow(x, 2), x))
  #sum_y_sq = sum(map(lambda x: pow(x, 2), y))
  #psum = sum(map(lambda x, y: x * y, x, y)) #dotproduct
  #num = psum - (sum_x * sum_y/n)
  #den = pow((sum_x_sq - pow(sum_x, 2) / n) * (sum_y_sq - pow(sum_y, 2) / n), 0.5)
  num = pairWiseCov(x,y)
  den = sd(x) * sd(y)

  if den == 0: return 0
  return num / den

def Cov(M):
   nC = M.shape[1]
   nR = M.shape[0]
   cv = np.zeros(shape=(nC,nC))
   for i in range (0,nC):
       for j in range(0,nC):
           if i > j:
               cv[i,j] = cv[j,i]
           #elif i == j:
           #    cv[i,i] = 1
           else:
               cv[i,j] = pairWiseCov(M[:,i],M[:,j])

   return cv

def SquaredUpperTriangSumOfElements(M):
    res = 0
    nC = M.shape[1]
    nR = M.shape[0]
    for i in range (0,nR):
       for j in range(i+1,nC):
           res = res + math.pow(M[i,j],2)
    return res

#def Jacobian(A,b,N = 25, x=None):
#    """Solves the equation Ax=B via the Jacobi iterative method"""

def PCA(Cov,noOfDesiredFactors = 0):
    m = Cov.shape[0]
    # use 'eigh' rather than 'eig' since R is symmetric, 
    # the performance gain is substantial
    hey = LA.eigh(Cov)
    he3 = np.linalg.eig(Cov)
    M = Cov
    n = M.shape[0]
    tolerance= math.pow(10,-20)
    sumsq = SquaredUpperTriangSumOfElements(M)
    V_prime = np.identity(n)
    while sumsq > tolerance:
        A_prime = JacobiMat(n,M)
        V_prime = JacobiVMat(n,M,V_prime)
        sumsq = SquaredUpperTriangSumOfElements(A_prime)
        M = A_prime
    evals = np.zeros(shape=(n),dtype=np.float)
    for i in range(0,n):
        evals[i] = M[i,i]
    evecs = V_prime
    


    #evals, evecs = LA.eigh(Cov)
    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    #calculate CumR^2 for Goodness of fit of model.
    
    lenVals = len(evals) if noOfDesiredFactors == 0 else noOfDesiredFactors
    CumR2 = np.zeros(shape=(lenVals),dtype=np.float)
    for i in range(0,lenVals):
        CumR2[i] = sum(evals[0:i])/sum(evals)

    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    if noOfDesiredFactors > 0:
        evecs = evecs[:, :noOfDesiredFactors]
        evals = evals[:noOfDesiredFactors]

    return evals, evecs.transpose(), CumR2, idx

def JacobiVMat(n,A,V):
    rot = JacobiRVec(n,A)
    P = JacobiPmat(n,rot[0],rot[1], rot[2])
    return np.matmul(np.array(V),np.array(P))

def JacobiRVec(n,A):
    """Returns  a vector contianing row number, col number and rotation angle"""
    maxV = -1
    col = -1 
    row = -1
    jrad = 0
    nCols = A.shape[1]
    nRows = A.shape[0]
    for i in range(0,nRows):
        for j in range(i+1,nCols):
            if abs(A[i,j]) > maxV:
                maxV = abs(A[i,j])
                col = j
                row = i
    if A[row,row] == A[col,col]:
        jrad = 0.25 * math.pi * (-1 if A[row,col] < 0 else 1 )
    else :
        jrad = 0.5 * math.atan(2 * A[row,col] / (A[row,row] - A[col,col]))
    return row, col, jrad

def JacobiPmat(n,row,col,rotAngleRad):
    """Returns the rotation matrix for the angle rotAngleRad"""
    P = np.identity(n, dtype=np.float)
    P[row,row] = math.cos(rotAngleRad)
    P[col,row] = math.sin(rotAngleRad)
    P[row,col] = -math.sin(rotAngleRad)
    P[col,col] = math.cos(rotAngleRad)
    return np.array(P)

def JacobiMat(n,A):
    rot = JacobiRVec(n,A)
    P = JacobiPmat(n,rot[0],rot[1], rot[2])
    return np.matmul(np.array(P.transpose()),np.matmul(np.array(A),np.array(P)))

def VolFromPCA(evals, evecs):
    volV = np.zeros(shape=evecs.shape)
    for i in range(0,len(evals)):
        volV[i] = math.sqrt(evals[i]) * evecs[i]
    return volV

def CorP(df:dict):
    keys = list(df.keys())
    M = len(keys)
    CorMat = np.zeros(shape=[M,M])
    for i in range(0,M):
        for j in range(0,M):
            IndKeyI = keys[i]
            IndKeyJ = keys[j]
            p = 0.0
            if i == j:
                CorMat[i,i] = 1
            elif CorMat[j,i] == 0:
                CorMat[i,j] = pearsonr(df[IndKeyI].values, df[IndKeyJ].values)
            else:
                CorMat[i,j] = CorMat[j,i]

    return CorMat


def CholeskyDecomp(Sigma: np.matrix):
    A = np.zeros(Sigma.shape)
    if A.shape[0] != A.shape[1]:
        print("Correlation Matrix passed to Cholesky distn is not square.")
    for i in range(0,A.shape[0]):
        for j in range(0,A.shape[1]):
            if j == i:
                A[i,i] = math.sqrt(Sigma[i,i] - sum(pow(A[i,0:(i)], 2)))
            elif j < i:
                A[j,i] = 0
            else:
                A[j,i] = (1/A[i,i]) * (Sigma[i,j] - sum(map(lambda x, y: x * y, A[i,0:(i)], A[j,0:(i)])))
    return A