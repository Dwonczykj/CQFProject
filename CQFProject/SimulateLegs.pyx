from Copulae import MultVarGaussianCopula, MultVarTDistnCopula, TCopulaDensity
from HazardRates import CreateCDSPVLegsForExactDefault
import operator
from plotting import return_lineChart
import numpy as np
cimport numpy as np
from LowDiscrepancyNumberGenerators cimport SobolNumbers
FDTYPE = np.float
DTYPE = np.int

cpdef np.ndarray UnifFromGaussCopula(np.ndarray LogRtnCorP, SobolNumbers NumbGen, int noIterations):
    cdef int i,j
    cdef np.ndarray[dtype=FDTYPE_t,ndim=2] u = np.zeros(shape=(LogRtnCorP.shape[0],noIterations))
    for i in range(0,noIterations):
        x = MultVarGaussianCopula(LogRtnCorP,NumbGen)
        for j in range(0,len(x)):
            u[j,i] = x[j]
    return u

cpdef np.ndarray UnifFromTCopula(np.ndarray RankCorP, SobolNumbers NumbGen, int T_df, int noIterations):
    cdef int i,j, vMLE
    cdef np.ndarray[dtype=FDTYPE_t,ndim=2] u = np.zeros(shape=(RankCorP.shape[0],noIterations))
    # cdef np.ndarray[dtype=FDTYPE_t,ndim=1] vE = np.zeros(shape=(24))
    # vE = TCopula_DF_MLE(U_hist, RankCorP)
    # return_lineChart(np.arange(1,24),[vE[1:]],name="MLE procedure for T copula degrees of freedom",xlabel="Degrees of Freedom",ylabel="Log-likelihood")
    for i in range(0,noIterations):
        x = MultVarTDistnCopula(RankCorP, T_df, NumbGen)
        for j in range(0,len(x)):
            u[j,i] = x[j]
    return u

cpdef np.ndarray TCopula_DF_MLE(np.ndarray U_hist_t, np.ndarray corM):
    return _TCopula_DF_MLE(U_hist_t,corM)

cdef np.ndarray _TCopula_DF_MLE(np.ndarray U_hist_t, np.ndarray corM):
    cdef int i,j,v, vStart, vCounter, nVars, maxV
    cdef float sumMLE, maxSumMLE
    cdef np.ndarray[dtype=DTYPE_t,ndim=1] corInds = np.zeros(shape=(0),dtype=DTYPE)
    cdef np.ndarray[dtype=FDTYPE_t,ndim=1] vE = np.zeros(shape=(24))
    nVars = corM.shape[0]
    for i in range(0, corM.shape[0]):
        for j in range(i+1, corM.shape[1]):
            if np.abs(corM[i,j]) > 0.8:
                if i not in corInds: corInds = np.append(corInds,i)
                if j not in corInds: corInds = np.append(corInds,j)
                if i not in corInds and j not in corInds:
                    nVars = nVars - 1
    v = np.max([1,np.min([23, (nVars *nVars - nVars)/2-1])])
    vStart = v
    vCounter = 0
    while v > 0 and v < 24:
        sumMLE = 0.0
        maxSumMLE = 0.0
        for i in range(0,U_hist_t.shape[0]):
            ct = TCopulaDensity(U_hist_t[i],v)
            print(ct)
            sumMLE += np.log(ct)
        vE[v] = sumMLE 
        if sumMLE > maxSumMLE:
            maxSumMLE = sumMLE
            maxV = v
        else: vCounter += 1

        if v < 23:
            if v > (vStart-1): v = v+1
            else: v = v-1
        else:   
            v = vStart - 1
            vCounter = 0

        v = ( (v+1) if v > (vStart-1) else (v-1) ) if v < 23 else (vStart - 1)
        if vCounter > 5:
            if v > (vStart-1): 
                v = vStart - 1
                vCounter = 0
            else: 
                v = 0
    vE[0] = maxV
    return vE



def SimulateLegPricesFromCorrelationNormal(HistCreditSpreads,TenorCreditSpreads,TenorCDSPayments,InvPWCDF,DiscountFactors,ImpHazdRts,DataTenorDic,U_correlatedNorm,CDSMaturity,FairSpreads,R=0.4,InitialNoDefaults_PeriodLengthYrs=0.0):
    cdef int i,i_TenorData,i_HistData
    ExactDefaultTimesGauss = dict()
    CDSLegsN = dict()
    CDSLegsSumN = dict()

    for i in range(0,5):
        i_TenorData = 5*i
        i_HistData = i + 1
        IndKey_Hist = HistCreditSpreads.columns[i_HistData]
        IndKey_Tenor = TenorCreditSpreads['Ticker'][i_TenorData]
        defaultTime = InvPWCDF[IndKey_Tenor](U_correlatedNorm[i])
        ExactDefaultTimesGauss[IndKey_Tenor] = defaultTime
    
    OrderedExactDefaultTimesGauss = sorted(ExactDefaultTimesGauss.items(), key=operator.itemgetter(1)) #quickSort(list(ExactDefaultTimesGauss.values()))
    for i in range(0,5):
        IndKey_Tenor = OrderedExactDefaultTimesGauss[i][0]
        CDSLegsN[i+1] = CreateCDSPVLegsForExactDefault(OrderedExactDefaultTimesGauss[i][1]+ InitialNoDefaults_PeriodLengthYrs,TenorCDSPayments,ImpHazdRts[IndKey_Tenor],DiscountFactors["Sonia"],FairSpreads[i],R) 
        #todo: how can we use this spread unless we already know a SIMULATED defualt time which then gives us the order of defaults and hence which of the kth to default spreads to use.
        # CDSLegsN[i+1] = CreateCDSPVLegsForExactDefault(OrderedExactDefaultTimesGauss[i][1],TenorCDSPayments,ImpHazdRts[IndKey_Tenor],DiscountFactors["Sonia"],DataTenorDic[IndKey_Tenor][CDSMaturity-1],0.4) #Use quoted credit spread instead
        CDSLegsSumN[i+1] = [sum(CDSLegsN[i+1]["CompensationLeg"]), sum(CDSLegsN[i+1]["PremiumLeg"])]
    return CDSLegsSumN

def SimulateLegPricesFromCorrelationT(HistCreditSpreads,TenorCreditSpreads,TenorCDSPayments,InvPWCDF,DiscountFactors,ImpHazdRts,DataTenorDic,U_correlatedT,CDSMaturity,FairSpreads,R=0.4,InitialNoDefaults_PeriodLengthYrs=0.0):
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
        CDSLegsT[i+1] = CreateCDSPVLegsForExactDefault(OrderedExactDefaultTimesT[i][1]+ InitialNoDefaults_PeriodLengthYrs,TenorCDSPayments,ImpHazdRts[IndKey_Tenor],DiscountFactors["Sonia"],FairSpreads[i],R)
        # CDSLegsT[i+1] = CreateCDSPVLegsForExactDefault(OrderedExactDefaultTimesT[i][1],TenorCDSPayments,ImpHazdRts[IndKey_Tenor],DiscountFactors["Sonia"],DataTenorDic[IndKey_Tenor][CDSMaturity-1],0.4) #Use quoted credit spread instead
        CDSLegsSumT[i+1] = [sum(CDSLegsT[i+1]["CompensationLeg"]), sum(CDSLegsT[i+1]["PremiumLeg"])]
    return CDSLegsSumT
