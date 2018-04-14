import math
import scipy as scp
from scipy.optimize import newton_krylov, newton
from scipy.stats import norm

def Merton1974GetD0(V0,sigmaV,D,r,T):
    d1 = (math.log(V0/D) + (r + 0.5 * math.pow(sigmaV,2)) * T)/(sigmaV * math.sqrt(T))
    d2 = d1 - (sigmaV * math.sqrt(T))
    return D * math.exp(-1 * r * T) - (D * math.exp(-1 * r * T) * norm.cdf(-1 * d2) - V0 *  norm.cdf(-1 * d1))

def implSigmaV(V0,D,E0,r,T):
    def func(sigmaV): 
        d1 = (math.log(V0/D) + (r + 0.5 * math.pow(sigmaV,2)) * T)/(sigmaV * math.sqrt(T))
        d2 = d1 - (sigmaV * math.sqrt(T))
        return V0 * norm.cdf(d1) - D * math.exp(-1 * r * T) * norm.cdf(d2) - E0
    sol = newton(func,0.15)
    return sol


def implV0SigmaV(D,E0,r,T,sigmaE):

    def func1(x):
        if len(x) != 2:
            print('Length of args must be 2')
        d1 = (math.log(x[0]/D) + (r + 0.5 * math.pow(x[1],2)) * T)/(x[1] * math.sqrt(T))
        d2 = d1 - (x[1] * math.sqrt(T))
        y = [0,0]
        y[0] = x[0] * norm.cdf(d1) - D * math.exp(-1 * r * T) * norm.cdf(d2) - E0
        y[1] = sigmaE * E0 - norm.cdf(d1) * x[1] * x[0]
        return y

    sol = newton_krylov(func1,[E0 + D,0.4])
    return sol


