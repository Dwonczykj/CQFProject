import pandas as pd
import math
import numpy as np
from scipy.optimize import newton

from dateutil.relativedelta import relativedelta
from EmpiricalFunctions import FindClosestKeyInDicAndReturnKeyBoundsAlgorithm
#create a datatable using pandas to contain the hazard rates and then another to contain the interpolated rates


def GetDFs(RequiredTenors, DiscountToDate, Dates, Rates):
    
    if not Dates[0] == DiscountToDate:
        Dates = pd.to_datetime([np.datetime64(DiscountToDate)] + list(Dates.values))
        Rates = pd.Series([0.0]).append(Rates,ignore_index=True)
    PosRates = (1 + Rates)
    Ts = np.fromiter(map(lambda end,start: relativedelta(end, start).years / 1.0 + relativedelta(end, start).months / 12.0 + relativedelta(end, start).days / 365.0 , Dates , np.full(fill_value=DiscountToDate,shape=len(Dates))), dtype=float)
    Dfs = np.fromiter(map(lambda x,y: pow(x,y), PosRates, -1.0 * Ts), dtype=np.float)
    DFTS = pd.DataFrame(data={'DF':Dfs},index=Ts)
    res = dict()
    res = InterpolateDFLogLinear(DFTS.to_dict()['DF'],RequiredTenors)
    return res

def InterpolateDFLogLinear(TenoredSeries, Tenors):
    i = 0
    fnd = FindClosestKeyInDicAndReturnKeyBoundsAlgorithm(TenoredSeries)

    res = dict()
    res[0] = 1
    for t in Tenors: 
        res[t] = fnd(t)
        if len(res[t]) > 1:
            tiplus1 = res[t][1]
            ti = res[t][0]
            lnDiscFac = ((t - ti)/(tiplus1 - ti)*math.log(TenoredSeries[tiplus1])) + ((tiplus1 - t)/(tiplus1 - ti)*math.log(TenoredSeries[ti]))
            res[t] = math.exp(lnDiscFac)
        else:
            res[t] = res[t][0]
            
    return res

def InterpolateHazrateLinear(TenoredSeries,Tenors):
    i = 0
    fnd2 = FindClosestKeyInDicAndReturnKeyBoundsAlgorithm(TenoredSeries)

    res = dict()
    res[0] = 1
    for t in Tenors: 
        res[t] = fnd2(t)
        if len(res[t]) > 1:
            tiplus1 = res[t][1]
            ti = res[t][0]
            intRate = (t - ti)/(tiplus1 - ti) * (TenoredSeries[tiplus1] - TenoredSeries[ti]) + TenoredSeries[ti]
            res[t] = intRate
        else:
            res[t] = TenoredSeries[res[t]].values[0]
            
    return res

#def DiscountFactorFn(DF: dict):
#    fnd = FindClosestKeyInDicAndReturnKeyBoundsAlgorithm(DF)
#    cache = dict()
#    def f(val):
#        if val in cache.keys():
#            return cache[val]
#        else:
#            res = fnd(val)
#            if len(res) > 1:
#                cache[val] = math.exp((val - res[0])/(res[1] - res[0]) * math.log(DF[res[1]]) + (res[1] - val)/(res[1] - res[0]) * math.log(DF[res[0]]))
#                return cache[val]
#            else:
#                cache[val] = DF[res[0]]
#                return cache[val]
#    return f

class DiscountFactorFn(object):
    def __init__(self, DF: dict):
        self.fnd = FindClosestKeyInDicAndReturnKeyBoundsAlgorithm(DF)
        self.cache = dict()
        self.DF = DF
    def __call__(self, val):
        if val in self.cache.keys():
            return self.cache[val]
        else:
            res = self.fnd(val)
            if len(res) > 1:
                self.cache[val] = math.exp((val - res[0])/(res[1] - res[0]) * math.log(self.DF[res[1]]) + (res[1] - val)/(res[1] - res[0]) * math.log(self.DF[res[0]]))
                return self.cache[val]
            else:
                self.cache[val] = self.DF[res[0]]
                return self.cache[val]

#def ImpProbFn(DF: dict):
#    fnd = FindClosestKeyInDicAndReturnKeyBoundsAlgorithm(DF)
#    cache = dict()
#    def f(val):
#        if val in cache.keys():
#            return cache[val]
#        else:
#            res = fnd(val)
#            if len(res) > 1:
#                cache[val] = math.exp((val - res[0])/(res[1] - res[0]) * math.log(DF[res[1]]) + (res[1] - val)/(res[1] - res[0]) * math.log(DF[res[0]]))
#                return cache[val]
#            else:
#                cache[val] = DF[res[0]]
#                return cache[val]
#    return f

class ImpProbFn(object):
    def __init__(self, DF: dict):
        self.fnd = FindClosestKeyInDicAndReturnKeyBoundsAlgorithm(DF)
        self.cache = dict()
        self.DF = DF
    def __call__(self, val):
        if val in self.cache.keys():
            return self.cache[val]
        else:
            res = self.fnd(val)
            if len(res) > 1:
                self.cache[val] = math.exp((val - res[0])/(res[1] - res[0]) * math.log(self.DF[res[1]]) + (res[1] - val)/(res[1] - res[0]) * math.log(self.DF[res[0]]))
                return self.cache[val]
            else:
                self.cache[val] = self.DF[res[0]]
                return self.cache[val]

def LogLinearInterpolatorForDiscFac(qDataHazards: pd.DataFrame, delta = 1):
    #check if params are to young
    i = 0
    sData = pd.DataFrame({'IntHazards-NonCum': [0],'IntDF0_T': [1]},index=[0])
    while i < (len(qDataHazards.index) - 1):
        ti = qDataHazards.index[i]
        tiplus1 = qDataHazards.index[i+1]
        j = 1
        while j <= ((qDataHazards.index[i+1] - qDataHazards.index[i])/delta):
            tau = ti + j * delta
            lnDiscFac = ((tau - ti)/(tiplus1 - ti)*math.log(qDataHazards['DF0_T'][tiplus1])) + ((tiplus1 - tau)/(tiplus1 - ti)*math.log(qDataHazards['DF0_T'][ti]))
            DF = math.exp(lnDiscFac)
            intRate = (tau - ti)/(tiplus1 - ti) * (qDataHazards['Hazards-NonCum'][tiplus1] - qDataHazards['Hazards-NonCum'][ti]) + qDataHazards['Hazards-NonCum'][ti]
            newFrame = pd.DataFrame({'IntHazards-NonCum': [intRate],
                                     'IntDF0_T': [DF]},index=[tau])
            sData = pd.concat([sData,newFrame])
            j+=1
        i+=1
    return sData

def PrSurv(l,R,T):
        if l == 0:
            return 1
        deltaArray = [T[k+1] - T[k] for k in range(l)]
        test = np.dot(R[T[1]:T[l]].values, deltaArray)
        sumTest = test.sum()
        res = math.exp(-1 * (sumTest))
        return res

def CreateCDSAccrualPVLegs(InterpolatedDataFrame: pd.DataFrame, Spread, RecvR):
    i = 1
    res = pd.DataFrame() #pd.concat([InterpolatedDataFrame])
    while i < (len(InterpolatedDataFrame.index)):
        #ti = InterpolatedDataFrame.index[i]
        #tiless1 = InterpolatedDataFrame.index[i-1]
        Rates = InterpolatedDataFrame['Hazards-NonCum']
        DFs = InterpolatedDataFrame['DF0_T']
        Tenors = InterpolatedDataFrame.index
        #t1 = InterpolatedDataFrame.index[1]

        PL = Spread * (DFs[Tenors[i]] * PrSurv(i,Rates,Tenors) * (Tenors[i] - Tenors[i-1]) + 
                       DFs[Tenors[i]] * (PrSurv(i-1,Rates,Tenors) - PrSurv(i,Rates,Tenors)) * (Tenors[i] - Tenors[i-1])/2 )
        #testPrSurv = (PrSurv(i-1,Rates,Tenors) - PrSurv(i,Rates,Tenors))
        CL = (1 - RecvR) * DFs[Tenors[i]] * (PrSurv(i-1,Rates,Tenors) - PrSurv(i,Rates,Tenors))

        newFrame = pd.DataFrame({'PremiumLeg': [PL],
                                 'CompensationLeg': [CL]},index=[InterpolatedDataFrame.index[i]])
        res = pd.concat([res,newFrame])
        i += 1
    return res

def CreateCDSPVLegs(InterpolatedDataFrame: pd.DataFrame, Spread, RecvR):
    i = 1
    res = pd.DataFrame()
    while i < (len(InterpolatedDataFrame.index)):
        Rates = InterpolatedDataFrame['Hazards-NonCum']
        DFs = InterpolatedDataFrame['DF0_T']
        Tenors = InterpolatedDataFrame.index

        PL = Spread * (DFs[Tenors[i]] * PrSurv(i,Rates,Tenors) * (Tenors[i] - Tenors[i-1])) 
        
        CL = (1 - RecvR) * DFs[Tenors[i]] * (PrSurv(i-1,Rates,Tenors) - PrSurv(i,Rates,Tenors))

        newFrame = pd.DataFrame({'PremiumLeg': [PL],
                                 'CompensationLeg': [CL]},index=[InterpolatedDataFrame.index[i]])
        res = pd.concat([res,newFrame])
        i += 1
    return res

def CreateCDSPVLegsForExactDefault(DefaultTime, PaymntDt, Rates, DFs, Spread, RecvR):
    i = 0
    res = dict()
    PremiumLeg = np.zeros(shape=(len(PaymntDt)),dtype=float)
    CompensationLeg = np.zeros(shape=(len(PaymntDt)),dtype=float)
    #todo: The legs have premium payemnts every 3 months, not once a year
    dft_occured_last_period = False
    while i < (len(PaymntDt)):
        dflt_occured_this_period = DefaultTime < PaymntDt[i]
        indP = 1 if not dft_occured_last_period else 0
        PL = Spread * indP * ((DFs(PaymntDt[i]) * (PaymntDt[i] - (PaymntDt[i-1] if i > 0 else 0))) if not dflt_occured_this_period 
                              else (DFs(DefaultTime) * (DefaultTime - (PaymntDt[i-1] if i > 0 else 0))) )
        indC = 1 if dflt_occured_this_period and not dft_occured_last_period else 0
        CL = (1 - RecvR) * DFs(PaymntDt[i]) * indC
        #indP = 1 if PaymntDt[i] <= DefaultTime else 0
        #indC = 1 if PaymntDt[i] > DefaultTime and (i == 0 or (i > 0 and PaymntDt[i-1] <= DefaultTime )) else 0
        #PL = Spread * (DFs(PaymntDt[i]) * indP * (PaymntDt[i] - (PaymntDt[i-1] if i > 0 else 0)))  #* PrSurv(i,Rates,Tenors)
        #CL = (1 - RecvR) * DFs(PaymntDt[i]) * indC #* (PrSurv(i-1,Rates,Tenors) - PrSurv(i,Rates,Tenors))
        #newFrame = pd.DataFrame({'PremiumLeg': [PL],
        #                         'CompensationLeg': [CL]},index=[InterpolatedDataFrame.index[i]])
        PremiumLeg[i] = PL;
        CompensationLeg[i] = CL;
        #res = pd.concat([res,newFrame])
        i += 1
        dft_occured_last_period = dflt_occured_this_period
    res["CompensationLeg"] = CompensationLeg
    res["PremiumLeg"] = PremiumLeg
    return res


def PriceCDS(PricedDataFrame: pd.DataFrame):
    PremLeg = PricedDataFrame['PremiumLeg'].sum()
    CompLeg = PricedDataFrame['CompensationLeg'].sum()
    MTM = CompLeg - PremLeg
    return MTM

def GetImpliedSpread(InterpolatedDataFrame: pd.DataFrame, RecvR):
    def func(s):
        Legs = CreateCDSAccrualPVLegs(InterpolatedDataFrame,s,RecvR)
        return PriceCDS(Legs)

    sol = newton(func,0.01)
    return sol


def Sum(start,stop,step,func,dtype=np.int):
    range = np.arange(start,stop + step,step,dtype)
    maped= map(func,range)
    res = np.fromiter(maped,dtype=np.float).sum()
    return res





def BootstrapImpliedProbalities(RR, spreads: pd.Series, DF):
    '''
    Function assumes annual term spreads, ie terms of 1,2,3,4,5,6,...
    RR: Recovery Rate,
    spreads: Credit spreads,
    DF: A function that takes time t as an argument and returns DF from that time to present.
    Returns Implied probabilities.
    '''
    sprSr = [np.nan] + spreads
    
    qDataDBCDS = pd.DataFrame({'Spread': sprSr},
                 index = range(0,len(sprSr)))
    
    #def DF(T):
    #    return math.exp(-1 * 0.008 * T)
    
    Tenors = qDataDBCDS.index
    Spreads =  qDataDBCDS['Spread']
    iP = []
    def initP(i):
        return (1 - RR) / (1 - RR + (Tenors[i] - Tenors[i-1])*Spreads[i])
    iP.append(1)
    iP.append(initP(1))

    i = 2
    while i < len(Tenors):
        def numFunc(j):
            return DF(Tenors[j]) * (((1 - RR) * iP[j-1]) - ((1 - RR + (Tenors[j] - Tenors[j-1]) * Spreads[i])*iP[j]))
        sm = Sum(start=1,stop=i-1,step=1,func=numFunc) 
        qtnt = (DF(Tenors[i])*(1 - RR + (Tenors[i] - Tenors[i-1]) * Spreads[i])) 
        extra = (iP[i-1]*initP(i))
        PTi = sm / qtnt + extra
        iP.append(PTi)
        i += 1
    qDataDBCDS['ImpliedPrSurv'] = iP
    return qDataDBCDS


def GetHazardsFromP(P,T):
    i = 1
    #rt = [math.log(P[T[1]]) / (-1 * T[1])]
    rt = []
    while i < len(T):
        nxt = math.log(P[T[i]]/P[T[i-1]]) / (T[i-1] - T[i])
        rt.append(nxt)
        i += 1
    return rt

