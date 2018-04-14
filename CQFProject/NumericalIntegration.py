from scipy import integrate
import math
from scipy.stats import norm

def estimateDefProb(k):
    def func(W):
        Z = norm.ppf(W)
        return nCr(125,k) * math.pow(F1z(Z),k) * math.pow(1 - F1z(Z),125 - k)
    return integrate.quad(func,0,1)



def testThy(Z):
    w = 0.3
    val = (-1.88 - w * Z)/(math.sqrt(1-math.pow(w,2)))
    return norm.cdf(val)

def nCr(N,k):
    return math.factorial(N) / (math.factorial(k) * math.factorial(N - k))

def F1z(Z):
    return norm.cdf((-1.88 - 0.3*Z)/(math.sqrt(1 - math.pow(0.3,2))))
