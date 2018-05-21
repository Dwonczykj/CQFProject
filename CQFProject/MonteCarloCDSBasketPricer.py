import math
import numpy as np
from LowDiscrepancyNumberGenerators import SobolNumbers
import time

#from HazardRates import *
#from Returns import *
#from EmpiricalFunctions import *
from plotting import plot_histogram_array, plot_codependence_scatters, return_lineChart
#from Sorting import *
from SimulateLegs import SimulateLegPricesFromCorrelationNormal, SimulateLegPricesFromCorrelationT, UnifFromGaussCopula, UnifFromTCopula
from RunningMoments import RunningAverage, RunningVarianceOfRunningAverage
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!MONTE CARLO SIMULATION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def SimulateCDSBasketDefaultsAndValueLegsT(TimeAtStart,CorP,M,HistCreditSpreads,TransformedHistDataDic,TenorCreditSpreads,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,SpreadArr=[],R=0.4):
    if len(SpreadArr) == 0:
        SpreadArr = np.ones(shape=(5))
    CompensationLegSumsForEachRefNameT = np.zeros(shape=(M,5),dtype=np.float)
    PremiumLegSumsForEachRefNameT = np.zeros(shape=(M,5),dtype=np.float)
    NumbGen = SobolNumbers()
    NumbGen.initialise(LogRtnCorP.shape[0])
    UT = UnifFromTCopula(CorP,NumbGen,len(TransformedHistDataDic[HistCreditSpreads.columns[1]]) - 1,M)
    UniformityDic = dict()
    for i in range(2,4):
        UniformityDic["T Copula %d" % (i+1)] = UT[i,:]
    #!plot_histogram_array(UniformityDic,"Simulated Ui T Copula")
    plot_codependence_scatters(UniformityDic,"Simulated Ui T Copula", "Simulated Uj T Copula")
    M_Min = 50
    Tolerance = 0.000001
    CompLegRunningAv = np.zeros(shape=(M,5))
    PremLegRunningAv = np.zeros(shape=(M,5))
    for m in range(1,M+1):
        TLegsSum = SimulateLegPricesFromCorrelationT(HistCreditSpreads,TenorCreditSpreads,CDSPaymentTenors,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,UT[:,m-1],int(CDSBasketMaturity),SpreadArr,R)
        CompensationLegSumsForEachRefNameT[m-1] = [TLegsSum[1][0],TLegsSum[2][0],TLegsSum[3][0],TLegsSum[4][0],TLegsSum[5][0]]
        PremiumLegSumsForEachRefNameT[m-1] = [TLegsSum[1][1],TLegsSum[2][1],TLegsSum[3][1],TLegsSum[4][1],TLegsSum[5][1]]
        CompLegRunningAv[m-1,:] = ((CompLegRunningAv[m-2,:] * (m-1)) + CompensationLegSumsForEachRefNameT[m-1]) / (m)
        PremLegRunningAv[m-1,:] = ((PremLegRunningAv[m-2,:] * (m-1)) + PremiumLegSumsForEachRefNameT[m-1]) / (m)
        if (m-1) % M_Min == 0 and m > M_Min:
            runningVarCompLeg = RunningVarianceOfRunningAverage(np.array(CompLegRunningAv[0:(m-1),:]).transpose(),M_Min)
            runningVarPremLeg = RunningVarianceOfRunningAverage(np.array(PremLegRunningAv[0:(m-1),:]).transpose(),M_Min)
            supRunningVarsCompLeg = max(runningVarCompLeg.transpose()[runningVarCompLeg.shape[1]-1])
            supRunningVarsPremLeg = max(runningVarPremLeg.transpose()[runningVarPremLeg.shape[1]-1])
            sup = max(supRunningVarsCompLeg,supRunningVarsPremLeg)
            if(sup < Tolerance):
                break
    #!return_lineChart(np.arange(0,M),CompLegRunningAv.transpose(),"Compensation Leg Running Average (Student's T Copula)",xlabel="Iteration",ylabel="Running Average", legend=["1st to Default","2nd to Default","3rd to Default","4th to Default","5th to Default"])
    #!return_lineChart(np.arange(0,M),PremLegRunningAv.transpose(),"Premium Leg Running Average (Student's T Copula)",xlabel="Iteration",ylabel="Running Average", legend=["1st to Default","2nd to Default","3rd to Default","4th to Default","5th to Default"])
    #!return_lineChart(np.arange(0,len(runningVarCompLeg[0])),runningVarCompLeg,"Running Variance of Compensation Leg (Student's T Copula)",legend=["1st to Default","2nd to Default","3rd to Default","4th to Default","5th to Default"])
    #!return_lineChart(np.arange(0,len(runningVarCompLeg[0])),runningVarPremLeg,"Running Variance of Premium Leg (Student's T Copula)",legend=["1st to Default","2nd to Default","3rd to Default","4th to Default","5th to Default"])
    tEnd = time.time()
    print("Took %.10f seconds to simulate %d iterations of T copulae and calculate legs from them." % (tEnd - TimeAtStart,M))

    return CompensationLegSumsForEachRefNameT,PremiumLegSumsForEachRefNameT,tEnd

def SimulateCDSBasketDefaultsAndValueLegsGauss(TimeAtStart,CorP,M,HistCreditSpreads,TenorCreditSpreads,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,SpreadArr=[],R=0.4):
    if len(SpreadArr) == 0:
        SpreadArr = np.ones(shape=(5))
    CompensationLegSumsForEachRefNameGauss = np.zeros(shape=(M,5),dtype=np.float)
    PremiumLegSumsForEachRefNameGauss = np.zeros(shape=(M,5),dtype=np.float)
    NumbGen = SobolNumbers()
    NumbGen.initialise(LogRtnCorP.shape[0])
    UNorm = UnifFromGaussCopula(CorP,NumbGen,M)
    UniformityDic = dict()
    for i in range(0,2):
        UniformityDic["Gaussian Copula %d" % (i+1)] = UNorm[i,:]
    #!plot_histogram_array(UniformityDic,"Simulated Ui Gaussian Copula")
    #!plot_codependence_scatters(UniformityDic,"Simulated Ui Gaussian Copula","Simulated Uj Gaussian Copula")
    M_Min = 50
    Tolerance = 0.000001
    CompLegRunningAv = np.zeros(shape=(M,5))
    PremLegRunningAv = np.zeros(shape=(M,5))
    for m in range(1,M):
        GaussLegsSum = SimulateLegPricesFromCorrelationNormal(HistCreditSpreads,TenorCreditSpreads,CDSPaymentTenors,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,UNorm[:,m-1],int(CDSBasketMaturity),SpreadArr,R)
        CompensationLegSumsForEachRefNameGauss[m-1] = [GaussLegsSum[1][0],GaussLegsSum[2][0],GaussLegsSum[3][0],GaussLegsSum[4][0],GaussLegsSum[5][0]]
        PremiumLegSumsForEachRefNameGauss[m-1] = [GaussLegsSum[1][1],GaussLegsSum[2][1],GaussLegsSum[3][1],GaussLegsSum[4][1],GaussLegsSum[5][1]]
        CompLegRunningAv[m-1,:] = ((CompLegRunningAv[m-2,:] * (m-1)) + CompensationLegSumsForEachRefNameGauss[m-1]) / (m)
        PremLegRunningAv[m-1,:] = ((PremLegRunningAv[m-2,:] * (m-1)) + PremiumLegSumsForEachRefNameGauss[m-1]) / (m)
        if (m-1) % M_Min == 0 and m > M_Min:
            runningVarCompLeg = RunningVarianceOfRunningAverage(np.array(CompLegRunningAv[0:(m-1),:]).transpose(),M_Min)
            runningVarPremLeg = RunningVarianceOfRunningAverage(np.array(PremLegRunningAv[0:(m-1),:]).transpose(),M_Min)
            supRunningVarsCompLeg = max(runningVarCompLeg.transpose()[runningVarCompLeg.shape[1]-1])
            supRunningVarsPremLeg = max(runningVarPremLeg.transpose()[runningVarPremLeg.shape[1]-1])
            sup = max(supRunningVarsCompLeg,supRunningVarsPremLeg)
            if(sup < Tolerance):
                break
    #!return_lineChart(np.arange(0,M),CompLegRunningAv.transpose(),"Compensation Leg Running Average (Gaussian Copula)",xlabel="Iteration",ylabel="Running Average", legend=["1st to Default","2nd to Default","3rd to Default","4th to Default","5th to Default"])
    #!return_lineChart(np.arange(0,M),PremLegRunningAv.transpose(),"Premium Leg Running Average (Gaussian Copula)",xlabel="Iteration",ylabel="Running Average", legend=["1st to Default","2nd to Default","3rd to Default","4th to Default","5th to Default"])
    #!return_lineChart(np.arange(0,len(runningVarCompLeg[0])),runningVarCompLeg,"Running Variance of Compensation Leg (Gaussian Copula)",legend=["1st to Default","2nd to Default","3rd to Default","4th to Default","5th to Default"])
    #!return_lineChart(np.arange(0,len(runningVarCompLeg[0])),runningVarPremLeg,"Running Variance of Premium Leg (Gaussian Copula)",legend=["1st to Default","2nd to Default","3rd to Default","4th to Default","5th to Default"])
    tEnd = time.time()
    print("Took %.10f seconds to simulate %d iterations of Gaussian copulae and calculate legs from them." % (tEnd - TimeAtStart,M))

    return CompensationLegSumsForEachRefNameGauss,PremiumLegSumsForEachRefNameGauss,tEnd

def CalculateFairSpreadFromLegs(CompensationLegSumsForEachRefName,PremiumLegSumsForEachRefName,M,t8,CopulaClass=""):
    CompAv = dict()
    PremAv = dict()
    CompRunningAv = np.zeros(shape=(5+1,M),dtype=np.float)
    PremRunningAv = np.zeros(shape=(5+1,M),dtype=np.float)

    for i in range(1,6):
        CompAv[i] = sum([item[i-1] for item in CompensationLegSumsForEachRefName]) / len(CompensationLegSumsForEachRefName)
        PremAv[i] = sum([item[i-1] for item in PremiumLegSumsForEachRefName]) / len(PremiumLegSumsForEachRefName)

    #CompRunningAv[1:6] = RunningAverage(CompensationLegSumsForEachRefName)
    #PremRunningAv[1:6] = RunningAverage(PremiumLegSumsForEachRefName)

    #MCIterations = np.arange(1,M+1,1,dtype=np.int)
    #return_lineChart(MCIterations,CompRunningAv[1:6], "Compensation Leg (%s Copula)" % CopulaClass,legend=["1st to default","2nd to default","3rd to default","4th to default","5th to default"])
    #return_lineChart(MCIterations,PremRunningAv[1:6], "Premium Leg (%s Copula)" % CopulaClass,legend=["1st to default","2nd to default","3rd to default","4th to default","5th to default"])

    #varWindow = 50 #no of iterations of MC algo to take for running Variance
    #CompRunningVar = np.zeros(shape=(5+1,math.ceil(M/varWindow)),dtype=np.float)
    #PremRunningVar = np.zeros(shape=(5+1,math.ceil(M/varWindow)),dtype=np.float)

    #CompRunningVar[1:6] = RunningVarianceOfRunningAverage(CompRunningAv[1:6],varWindow)
    #PremRunningVar[1:6] = RunningVarianceOfRunningAverage(PremRunningAv[1:6],varWindow)

    #MCIterations = np.arange(0,M,varWindow,dtype=np.int)
    #return_lineChart(MCIterations,CompRunningVar[1:6], "Running Variance of Compensation Leg (%s Copula)" % CopulaClass, xlabel="Per 100 MC Iterations", legend=["1st to default","2nd to default","3rd to default","4th to default","5th to default"])
    #return_lineChart(MCIterations,PremRunningVar[1:6], "Running Variance of Premium Leg (%s Copula)" % CopulaClass, xlabel="Per 100 MC Iterations", legend=["1st to default","2nd to default","3rd to default","4th to default","5th to default"])

    FairSpreads = np.zeros(shape=(5),dtype=np.float)

    for i in range(0,5):
        FairSpreads[i] = CompAv[i+1] / PremAv[i+1]

    #!return_lineChart(range(1,6),[FairSpreads], "Fair spread for Kth to default (%s Copula)" % CopulaClass, xlabel="Kth to default")

    t9 = time.time()
    print("Took %.10f seconds to print running statistics and calculate fair spreads from legs for %s Copula." % (t9 - t8,CopulaClass))

    return FairSpreads, t9

def FullMCFairSpreadValuation(startTime,LogRtnCorP,RankCorP,M,HistCreditSpreads,TransformedHistDataDic,TenorCreditSpreads,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,TFairSpread=[],GaussFairSpread=[],R=0.4):

    TLegs = SimulateCDSBasketDefaultsAndValueLegsT(startTime,RankCorP,M,HistCreditSpreads,TransformedHistDataDic,TenorCreditSpreads,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,TFairSpread,R)
    GaussLegs = SimulateCDSBasketDefaultsAndValueLegsGauss(TLegs[2],LogRtnCorP,M,HistCreditSpreads,TenorCreditSpreads,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,GaussFairSpread,R)


    GaussFairSpread,t9 = CalculateFairSpreadFromLegs(GaussLegs[0],GaussLegs[1],M,GaussLegs[2],"Gauss")
    TFairSpread,t10 = CalculateFairSpreadFromLegs(TLegs[0],TLegs[1],M,t9,"T")

    return GaussFairSpread, TFairSpread, t10