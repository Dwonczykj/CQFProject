import math
from scipy.stats import norm

def DefaultProbBlackCox(V0,sigmaV,r,T,K):
    h1 = (math.log(K / (V0 * math.exp(r*T))) + 0.5 * math.pow(sigmaV,2) * T ) / (sigmaV * math.sqrt(T))
    h2 = h1 - sigmaV * math.sqrt(T)
    return norm.cdf(h1) + math.exp(math.log(K/V0) * (1/math.pow(sigmaV,2)) * 2 * (r - math.pow(sigmaV,2) / 2) ) * norm.cdf(h2) 

def DefaultProbMerton(V0,sigmaV,r,T,D):
    d1 = (math.log(V0/D) + (r + 0.5 * math.pow(sigmaV,2)) * T)/(sigmaV * math.sqrt(T))
    d2 = d1 - (sigmaV * math.sqrt(T))
    return norm.cdf(-1 * d2)