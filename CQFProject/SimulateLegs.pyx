from Copulae import MultVarGaussianCopula, MultVarTDistnCopula
from HazardRates import CreateCDSPVLegsForExactDefault
import operator
import numpy as np
cimport numpy as np
from LowDiscrepancyNumberGenerators cimport SobolNumbers
FDTYPE = np.float

cpdef np.ndarray UnifFromGaussCopula(np.ndarray LogRtnCorP, SobolNumbers NumbGen, int noIterations):
    cdef int i,j
    cdef np.ndarray[dtype=FDTYPE_t,ndim=2] u = np.zeros(shape=(LogRtnCorP.shape[0],noIterations))
    for i in range(0,noIterations):
        x = MultVarGaussianCopula(LogRtnCorP,NumbGen)
        for j in range(0,len(x)):
            u[j,i] = x[j]
    return u

cpdef np.ndarray UnifFromTCopula(np.ndarray RankCorP, SobolNumbers NumbGen, int SeriesLength, int noIterations):
    cdef int i,j
    cdef np.ndarray[dtype=FDTYPE_t,ndim=2] u = np.zeros(shape=(RankCorP.shape[0],noIterations))
    for i in range(0,noIterations):
        x = MultVarTDistnCopula(RankCorP, SeriesLength, NumbGen)
        for j in range(0,len(x)):
            u[j,i] = x[j]
    return u

def SimulateLegPricesFromCorrelationNormal(HistCreditSpreads,TenorCreditSpreads,TenorCDSPayments,InvPWCDF,DiscountFactors,ImpHazdRts,DataTenorDic,U_correlatedNorm,CDSMaturity,FairSpreads,R=0.4):
    cdef int i,i_TenorData,i_HistData
    ExactDefaultTimesGauss = dict()
    CDSLegsN = dict()
    CDSLegsSumN = dict()

    for i in range(0,5):
        i_TenorData = 5*i
        i_HistData = i + 1
        IndKey_Hist = HistCreditSpreads.columns[i_HistData]
        IndKey_Tenor = TenorCreditSpreads['Ticker'][i_TenorData]
        ExactDefaultTimesGauss[IndKey_Tenor] = InvPWCDF[IndKey_Tenor](U_correlatedNorm[i])
    
    OrderedExactDefaultTimesGauss = sorted(ExactDefaultTimesGauss.items(), key=operator.itemgetter(1)) #quickSort(list(ExactDefaultTimesGauss.values()))
    for i in range(0,5):
        IndKey_Tenor = OrderedExactDefaultTimesGauss[i][0]
        CDSLegsN[i+1] = CreateCDSPVLegsForExactDefault(OrderedExactDefaultTimesGauss[i][1],TenorCDSPayments,ImpHazdRts[IndKey_Tenor],DiscountFactors["Sonia"],FairSpreads[i],R) 
        #todo: how can we use this spread unless we already know a SIMULATED defualt time which then gives us the order of defaults and hence which of the kth to default spreads to use.
        # CDSLegsN[i+1] = CreateCDSPVLegsForExactDefault(OrderedExactDefaultTimesGauss[i][1],TenorCDSPayments,ImpHazdRts[IndKey_Tenor],DiscountFactors["Sonia"],DataTenorDic[IndKey_Tenor][CDSMaturity-1],0.4) #Use quoted credit spread instead
        CDSLegsSumN[i+1] = [sum(CDSLegsN[i+1]["CompensationLeg"]), sum(CDSLegsN[i+1]["PremiumLeg"])]
    return CDSLegsSumN

def SimulateLegPricesFromCorrelationT(HistCreditSpreads,TenorCreditSpreads,TenorCDSPayments,InvPWCDF,DiscountFactors,ImpHazdRts,DataTenorDic,U_correlatedT,CDSMaturity,FairSpreads,R=0.4):
    cdef int i,i_TenorData,i_HistData
    ExactDefaultTimesT = dict()
    CDSLegsT = dict()
    CDSLegsSumT = dict()

    for i in range(0,5):
        i_TenorData = 5*i
        i_HistData = i + 1
        IndKey_Hist = HistCreditSpreads.columns[i_HistData]
        IndKey_Tenor = TenorCreditSpreads['Ticker'][i_TenorData]
        ExactDefaultTimesT[IndKey_Tenor] = InvPWCDF[IndKey_Tenor](U_correlatedT[i])
    OrderedExactDefaultTimesT= sorted(ExactDefaultTimesT.items(), key=operator.itemgetter(1))
    for i in range(0,5):
        IndKey_Tenor = OrderedExactDefaultTimesT[i][0]
        CDSLegsT[i+1] = CreateCDSPVLegsForExactDefault(OrderedExactDefaultTimesT[i][1],TenorCDSPayments,ImpHazdRts[IndKey_Tenor],DiscountFactors["Sonia"],FairSpreads[i],R)
        # CDSLegsT[i+1] = CreateCDSPVLegsForExactDefault(OrderedExactDefaultTimesT[i][1],TenorCDSPayments,ImpHazdRts[IndKey_Tenor],DiscountFactors["Sonia"],DataTenorDic[IndKey_Tenor][CDSMaturity-1],0.4) #Use quoted credit spread instead
        CDSLegsSumT[i+1] = [sum(CDSLegsT[i+1]["CompensationLeg"]), sum(CDSLegsT[i+1]["PremiumLeg"])]
    return CDSLegsSumT
