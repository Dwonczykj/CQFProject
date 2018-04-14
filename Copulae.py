import math
import numpy as np
import pandas as pd
from scipy.stats import norm, t
from Returns import CholeskyDecomp
from Distributions import InvStdCumNormal
from LowDiscrepancyNumberGenerators import SobolNumbers

def SharedCopulaAlgoWork(P:np.matrix, LowDiscNumbers: SobolNumbers):
    A = CholeskyDecomp(P)
    #A1 = np.linalg.cholesky(P)
    d = A.shape[0]
    #Xtilda = np.random.uniform(size=A.shape[0])
    #Xtilda = norm.ppf(Xtilda)
    #todo: Use the random number generator to generate the normal vals here. Read Jaeckel
    LowDiscU = LowDiscNumbers.Generate()
    IndptXtilda = np.fromiter(map(lambda u: InvStdCumNormal(u),LowDiscU),dtype=float)
    #IndependentXtilda = np.random.randn(A.shape[0])
    CorrlelatedX = np.matmul(A.transpose(),IndptXtilda)
    return CorrlelatedX

def MultVarGaussianCopula(P:np.matrix, LowDiscNumbers: SobolNumbers):
    CorrlelatedX = SharedCopulaAlgoWork(P,LowDiscNumbers)
    U = norm.cdf(CorrlelatedX)
    return U

def MultVarTDistnCopula(P:np.matrix,df: np.int, LowDiscNumbers: SobolNumbers):
    CorrlelatedX = SharedCopulaAlgoWork(P,LowDiscNumbers)
    epsilon = sum(np.random.uniform(size=df))
    U = t.cdf(CorrlelatedX/math.sqrt(epsilon/df),df)
    return U