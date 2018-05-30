#! TO BUILD CYTHON EXTENSIONS, RUN "python setup.py build_ext --inplace" IN A SHELL AT THE PROJECT ROOT. 
#!Ensure that the version of C Compiler you have matches the version used to compile the python environment that you are using.

import pandas as pd
import os
#import datetime
#import collections
import time

from HazardRates import *
from Returns import *
from EmpiricalFunctions import *
from plotting import plot_histogram_array, showAllPlots, plot_codependence_scatters, Plot_Converging_Averages, return_lineChart, return_scatter_multdependencies, return_barchart, return_lineChart_dates, save_all_figs
#from Copulae import MultVarGaussianCopula, MultVarTDistnCopula
from Sorting import *
from LowDiscrepancyNumberGenerators import SobolNumbers
from Logger import convertToLaTeX, printf
from MonteCarloCDSBasketPricer import CalculateFairSpreadFromLegs, SimulateCDSBasketDefaultsAndValueLegsGauss, SimulateCDSBasketDefaultsAndValueLegsT, FullMCFairSpreadValuation
from ProbabilityIntegralTransform import *
#from SimulateLegs import SimulateLegPricesFromCorrelationNormal, SimulateLegPricesFromCorrelationT, UnifFromGaussCopula, UnifFromTCopula
#from RunningMoments import RunningAverage, RunningVariance
#from sklearn.grid_search import GridSearchCV
#grid = GridSearchCV(KernelDensity(),
#                    {'bandwidth': np.linspace(0.1, 1.0, 30)},
#                    cv=20) # 20-fold cross-validation
#grid.fit(x[:, None])
##print grid.best_params_
#kde = grid.best_estimator_
#pdf = np.exp(kde.score_samples(x_grid[:, None]))

#fig, ax = plt.subplots()
#ax.plot(x_grid, pdf, linewidth=3, alpha=0.5, label='bw=%.2f' % kde.bandwidth)
#ax.hist(x, 30, fc='gray', histtype='stepfilled', alpha=0.3, normed=True)
#ax.legend(loc='upper left')
#ax.set_xlim(-4.5, 3.5);
#test()
#GenerateEVTKernelSmoothing()
from sobol_seq import i4_sobol
#SobolInitDataFp = os.getcwd() + '/Sobol_Initialisation_Numbers.csv'
#SobolInitData = pd.read_csv(SobolInitDataFp)
#def ProcessDirections(s):
#    st = s.split(' ')
#    return np.append(np.array([int(float(a)) for a in st]),np.zeros(shape=(30-len(st))))
#SobolInitData['v_i'] = SobolInitData['m_i'].map(ProcessDirections)
#Z = SobolInitData['Degree'][40-1]
#tv = SobolInitData['v_i'].values[0:40]
#tvalt = np.array(list(map(lambda x: np.array(list(x)), zip(*tv))))
#initV = np.append([0],tvalt[0:Z])
#mdeg=SobolInitData['Degree'][0:40].values
#initP=SobolInitData['iP'][0:40].values

#class ImportSobolNumbGen:
#    def __init__(self, dim):
#        quasi, s = i4_sobol(dim,self.seed)
#        self.seed = s
#        self.dim = dim
        
#    seed = 0
#    dim = 0

#    def Generate(self):
#        quasi, s = i4_sobol(self.dim,self.seed)
#        self.seed = s
#        return quasi

#NumbGen = ImportSobolNumbGen(5)
#l = 500
#DummyRUs = np.zeros(shape=(l,5))
#for j in range(0,l):
#    DummyRUs[j] = np.array(NumbGen.Generate(),np.float)
#plot_codependence_scatters(dict(enumerate(DummyRUs.transpose())),"D%","D%")
#NumbGen = SobolNumbers()
#NumbGen.initialise(5)
#DummyRUs = np.zeros(shape=(l,5))
#for j in range(0,l):
#    DummyRUs[j] = np.array(NumbGen.Generate(),np.float)
#plot_codependence_scatters(dict(enumerate(DummyRUs.transpose())),"D%","D%")


cwd = os.getcwd()
Datafp = cwd + '/FinalProjData.xlsx'
TenorCreditSpreads = pd.read_excel(Datafp,'TenorCreditSpreads',0)
HistCreditSpreads = pd.read_excel(Datafp,'HistoricalCreditSpreads',0)
DiscountCurves = pd.read_excel(Datafp,'DiscountCurves',0)

        
t1 = time.time()
DataTenorDic = dict()
ImpProbDic = dict()
ImpHazdRts = dict()
InvPWCDF = dict()
hazardLines = [[] for i in range(0,5)]
hazardLegend = ["" for i in range(0,5)]
for i in range(0,5*5,5):
    IndKey = TenorCreditSpreads['Ticker'][i]
    DataTenorDic[IndKey] = list(TenorCreditSpreads['DataSR'][i:(i+5)] / 1000)
    ImpProbDic[IndKey] = BootstrapImpliedProbalities(0.4,DataTenorDic[IndKey],TenorCreditSpreads.index)
    Tenors = ImpProbDic[IndKey].index
    BootstrappedSurvProbs = ImpProbDic[IndKey]['ImpliedPrSurv']
    ImpHazdRts[IndKey] = GetHazardsFromP(BootstrappedSurvProbs,Tenors)
    hazardLegend[int(i/5)] = IndKey
    hazardLines[int(i/5)] = [0] + ImpHazdRts[IndKey]
    InvPWCDF[IndKey] = ApproxPWCDFDicFromHazardRates(ImpHazdRts[IndKey],0.01)
t2 = time.time()
return_lineChart(np.arange(0,6,1,dtype=np.int),np.array(hazardLines),"Hazard Rates",xlabel="Time/years",ylabel="Hazard Rate", legend=hazardLegend)
print("Took %.10f seconds to Grab Tenor Data and Init Inverse Empirical CDF functions." % (t2 - t1))
#TenorCreditSpreads["ToDate"] = pd.to_datetime(TenorCreditSpreads["ToDate"])

DiscountFactors = dict()
DiscountFactorCurve = dict()
for i in range(0,4):
    IndKey = DiscountCurves.columns[2*i + 1]
    IndKeyDate = IndKey + "_Date"
    DiscountFactors[IndKey] = GetDFs(
        TenorCreditSpreads['Tenor'][0:5], 
        TenorCreditSpreads["ToDate"][0], 
        pd.to_datetime(DiscountCurves[IndKeyDate])[DiscountCurves[IndKeyDate].notnull()], 
        DiscountCurves[IndKey][DiscountCurves[IndKey].notnull()]
    )
    DiscountFactorCurve[IndKey] = DiscountFactorFn(DiscountFactors[IndKey])
return_lineChart(np.arange(0,5.01,0.01,dtype=np.float),[pd.Series(np.arange(0,5.01,0.01,dtype=np.float)).apply(DiscountFactorCurve["Sonia"])],"Sonia Discount Curve",xlabel="Time/years",ylabel="Discount Factor")
t3 = time.time()
print("Took %.10f seconds to Grab Discount Factors." % (t3 - t2))

#calc log returns from historical data and then calc corrs on it.
HistDataDic = dict()
CanonicalMLETransformedHistDataDic = dict()
SemiParamTransformedCDFHistDataDic = dict()
SemiParamTransformedPDFHistDataDic = dict()
EmpFnForHistSpread = dict()
SemiParamametric = dict()
LogReturnsDic = dict()
HistDefaults = dict()
DifferencesDic = dict()
#ResidualsLogReturnsDic = dict()
def Bootstrap5yrDP(spread):
    return BootstrapImpliedProbalities(0.4,pd.Series([spread]),5)['ImpliedPrSurv'][1]
for i in range(1,6):
    IndKey = HistCreditSpreads.columns[i]
    HistDataDic[IndKey] = pd.DataFrame({ 'Spreads': HistCreditSpreads[IndKey].values}, dtype=np.float, index = HistCreditSpreads['Date'].values)
    #todo: Measure the autocorrelation of the non touched returns to give an argument for the need for standardised residuals.
    #From http://uk.mathworks.com/help/econ/examples/using-extreme-value-theory-and-copulas-to-evaluate-market-risk.html#d119e9608:
    #"Comparing the ACFs of the standardized residuals to the corresponding ACFs of the raw returns reveals that the standardized residuals are now approximately i.i.d., thereby far more amenable to subsequent tail estimation."
    LogReturnsDic[IndKey] = LogReturns(HistDataDic[IndKey]['Spreads'],lag=5,jump=5,averageJumpPeriod=True)
    DifferencesDic[IndKey] = AbsoluteDifferences(HistDataDic[IndKey]['Spreads'],jump=5,averageJumpPeriod=True)
    #ResidualsLogReturnsDic[IndKey] = StandardisedResiduals(LogReturnsDic[IndKey])
    
    EmpFnForHistSpread[IndKey] = Empirical_StepWise_CDF(quickSort(DifferencesDic[IndKey].values))
    u_gpdthreashold = np.percentile(HistDataDic[IndKey]['Spreads'],95)
    SemiParamametric[IndKey] = SemiParametricCDFFit(list(HistDataDic[IndKey]['Spreads']),u_gpdthreashold, True, "SemiParametricFit_%s"%(IndKey), "Historical Spreads", "Distribution")
    CanonicalMLETransformedHistDataDic[IndKey] = pd.Series(DifferencesDic[IndKey].values).apply(EmpFnForHistSpread[IndKey])
    SemiParamTransformedCDFHistDataDic[IndKey] =  pd.Series(SemiParamametric[IndKey]['%.10f'%(u_gpdthreashold)][0])
    SemiParamTransformedPDFHistDataDic[IndKey] =  pd.Series(SemiParamametric[IndKey]['%.10f'%(u_gpdthreashold)][1])
    
    
return_lineChart_dates(HistCreditSpreads['Date'].values,[
    list(HistCreditSpreads[HistCreditSpreads.columns[1]]),
    list(HistCreditSpreads[HistCreditSpreads.columns[2]]), 
    list(HistCreditSpreads[HistCreditSpreads.columns[3]]), 
    list(HistCreditSpreads[HistCreditSpreads.columns[4]]), 
    list(HistCreditSpreads[HistCreditSpreads.columns[5]])
    ],name="Historical Credit Spreads Data", xlabel="Historical Date", ylabel="Spread", legend=list(HistCreditSpreads.columns[1:]))
    
t4 = time.time()
print("Took %.10f seconds to grab Historical Spreads and Transform the data by its Empirical CDF." % (t4 - t3))  
#ResidualCorP = CorP(ResidualsLogReturnsDic) 
plot_histogram_array(LogReturnsDic, "Weekly Log Returns")
plot_histogram_array(DifferencesDic, "Weekly Differences")
plot_histogram_array(CanonicalMLETransformedHistDataDic,"Inverse ECDF (Rank)")
plot_histogram_array(SemiParamTransformedCDFHistDataDic,"Inverse Semi-Parametric CDF (Rank)")
plot_histogram_array(SemiParamTransformedPDFHistDataDic,"Inverse Semi-Parametric PDF (Rank)")
#!showAllPlots()
t4a = time.time()
print("Took %.10f seconds to print Transformed Spreads Histograms" % (t4a - t4))
#DefaultCorP = CorP(HistDefaults)

LogRtnCorP = CorP(LogReturnsDic)    #near correlation without kernel smoothing for gaussian copula 
pdCorLogRtnP = convertToLaTeX(pd.DataFrame(LogRtnCorP, dtype=np.float),name="Pearson Correlation")
#Transform HistCreditSpreads by its own empirical distn and then calc corrln on U to get Rank Corr. This is the defn of Spearmans Rho  
#!Consider also using KENDALLS TAU AS AN ALTERNATIVE FORMULATION FOR THE RANK CORRELATION
RankCorP = CorP(CanonicalMLETransformedHistDataDic)
pdCorRankP = convertToLaTeX(pd.DataFrame(RankCorP, dtype=np.float),name="Rank Correlation")
diffCor = RankCorP - LogRtnCorP
pdCorDiffs = convertToLaTeX(pd.DataFrame(diffCor, dtype=np.float),name="Rank Pearson Correlation Diffs")

#!SemiParametric Correlation Matrices
SemiParametricRankCorP = CorP(SemiParamTransformedCDFHistDataDic)
pdSemiParametricCorRankP = convertToLaTeX(pd.DataFrame(SemiParametricRankCorP, dtype=np.float),name="Semi Parametric Correlation")
diffCorSemi = SemiParametricRankCorP - LogRtnCorP
pdSemiCorDiffs = convertToLaTeX(pd.DataFrame(diffCorSemi, dtype=np.float),name="Semi Parametric Correlation Diffs")
pdRevissedCorDiffs = convertToLaTeX(pd.DataFrame(SemiParametricRankCorP - RankCorP, dtype=np.float),name="Semi Parametric Correlation Diffs From Revision")

#UseSemi:
RankCorP = SemiParametricRankCorP

t5 = time.time()
print("Took %.10f seconds to calculate Correlation Matrices." % (t5 - t4a))

#class ImportSobolNumbGen:
#    def __init__(self, dim):
#        quasi, s = i4_sobol(dim,self.seed)
#        self.seed = s
#        self.dim = dim
        
#    seed = 0
#    dim = 0

#    def Generate(self):
#        quasi, s = i4_sobol(self.dim,self.seed)
#        self.seed = s
#        return quasi

#NumbGen = ImportSobolNumbGen(LogRtnCorP.shape[0])
NumbGen = SobolNumbers()
NumbGen.initialise(LogRtnCorP.shape[0])
l = 500
DummyRUs = np.zeros(shape=(l,LogRtnCorP.shape[0]))
for j in range(0,l):
    DummyRUs[j] = np.array(NumbGen.Generate(),np.float)
plot_codependence_scatters(dict(enumerate(DummyRUs.transpose())),"D%","D%")
#DummyRUs = np.zeros(shape=(l,LogRtnCorP.shape[0]))
#for j in range(0,l):
#    DummyRUs[j] = np.array(NumbGen.Generate(),np.float)
#plot_codependence_scatters(dict(enumerate(DummyRUs.transpose())),"D%","D%")
#for i in range(0,5000):
#    NumbGen.Generate()    
t6 = time.time()
print("Took %.10f seconds to init and run 5000x%d iterations of sobol numbers." % ((t6 - t5) , LogRtnCorP.shape[0]))



#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!MONTE CARLO SIMULATION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

CDSBasketMaturity = 5.0
CDSPaymentTenor = 0.25
CDSPaymentTenors = np.arange(CDSPaymentTenor,CDSBasketMaturity+CDSPaymentTenor,CDSPaymentTenor,dtype=np.float)
M = 10000

#TLegs = SimulateCDSBasketDefaultsAndValueLegsT(t6,RankCorP,M,HistCreditSpreads,TransformedHistDataDic,TenorCreditSpreads,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity)
#GaussLegs = SimulateCDSBasketDefaultsAndValueLegsGauss(TLegs[2],LogRtnCorP,M,HistCreditSpreads,TenorCreditSpreads,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity)

#GaussFairSpread,t9 = CalculateFairSpreadFromLegs(GaussLegs[0],GaussLegs[1],M,GaussLegs[2],"Gauss")
#TFairSpread,t10 = CalculateFairSpreadFromLegs(TLegs[0],TLegs[1],M,TLegs[2],"T")


GaussFairSpread,TFairSpread,t10 = FullMCFairSpreadValuation(t6,LogRtnCorP,RankCorP,M,HistCreditSpreads,CanonicalMLETransformedHistDataDic,TenorCreditSpreads,InvPWCDF,
                                                            DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,name="Initial_Fair_Spread_Calculation")
latexFairSpreads = convertToLaTeX(pd.DataFrame(data=np.array([GaussFairSpread,TFairSpread],dtype=np.float), index = ["Gaussian", "Student's T"], columns=["1st to default","2nd to default","3rd to default","4th to default","5th to default"], dtype=np.float),"Fair Spreads")



#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!CHECK OF FAIR SPREADS BY MONTE CARLO SIMULATION USING ESTIMATED FAIR SPREAD!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

TLegsWFairSpread = SimulateCDSBasketDefaultsAndValueLegsT(t10,RankCorP,M,HistCreditSpreads,CanonicalMLETransformedHistDataDic,TenorCreditSpreads,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,TFairSpread,name="Check_plausability_of_fair_spread")
GaussLegsWFairSpread = SimulateCDSBasketDefaultsAndValueLegsGauss(TLegsWFairSpread[2],LogRtnCorP,M,HistCreditSpreads,TenorCreditSpreads,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,GaussFairSpread,name="Check_plausability_of_fair_spread")
t11 = GaussLegsWFairSpread[2]
#GaussLegsWFairSpread,TLegsWFairSpread,t11 = FullMCFairSpreadValuation(t10,LogRtnCorP,RankCorP,M,HistCreditSpreads,TransformedHistDataDic,TenorCreditSpreads,InvPWCDF,
#                                                                      DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,TFairSpread,GaussFairSpread)
TFairDiff = TLegsWFairSpread[0] - TLegsWFairSpread[1] 
GaussFairDiff = GaussLegsWFairSpread[0] - GaussLegsWFairSpread[1]

gaussCheckFairSpreadDic = dict()
tCheckFairSpreadDic = dict()
for iten in range(0,TFairDiff.shape[1]):
    gaussCheckFairSpreadDic["%d-th to default basket CDS" % (iten+1)] = GaussFairDiff[:,iten]
    tCheckFairSpreadDic["%d-th to default basket CDS" % (iten+1)] = TFairDiff[:,iten]
plot_histogram_array(gaussCheckFairSpreadDic,"CDS Basket Spreads",name="CDS Basket Outcome Using Gaussian-Copula to check calculated fair spreads")
plot_histogram_array(tCheckFairSpreadDic, "CDS Basket Spreads", name="CDS Basket Outcome Using T-Copula to check calculated fair spreads")
#For Kendalls Tau, we have X1 and X2 from the data with empirical cdf, we then also simulate X3 and X4 from the emp distributions of X1 and x2 resp. We then defn pTau := E[sign((x1-x3)*(x2-x4))] 
#Now Consider altering the tail dependence of the copulas upper and lower separately.

#Step 1: Calculate the empirical cdf for each hist Ref name, both using kernel smoothing(pdf -> cdf) and just empirical stewpise constant cdf.
#Step 2: Transform historical spreads, X, by their CDF F(X) = U
#Step 3: Obtain Normally, T distributed R.v.s by applying the respective inverse CDF to the U.
#Step 4: Calculate Correlations from Standardised Credit Spread returns / Default Probs and use these near and rank correlation matrices to simulate Normal or T distributed variables respectively.
#Step 5: Compare the difference between 1-3 and 4.
#showAllPlots()



#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  RISK AND SENSITIVITY ANALYSIS   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

itera = 0
Rrng = np.arange(0.2,0.9,0.1)
GaussFairSpreadTweakR = np.zeros(shape=(len(Rrng),5),dtype=np.float)
TFairSpreadTweakR = np.zeros(shape=(len(Rrng),5),dtype=np.float)
for RAlt in Rrng:
    GaussFairSpreadTweakR[itera], TFairSpreadTweakR[itera], t16 = FullMCFairSpreadValuation(time.time(),LogRtnCorP,RankCorP,M,HistCreditSpreads,CanonicalMLETransformedHistDataDic,TenorCreditSpreads,InvPWCDF,
                                                                      DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,RAlt,name="Alternative recovery rate: ({0})".format(RAlt))
    itera += 1

return_scatter_multdependencies(Rrng,GaussFairSpreadTweakR.transpose(),"Sensitivity of FairSpread to changing Recovery Rate (Gauss)", 
                 xlabel="Recovery Rate",ylabel="FairSpread",legend=["1st to default","2nd to default","3rd to default","4th to default","5th to default"])
return_scatter_multdependencies(Rrng,TFairSpreadTweakR.transpose(),"Sensitivity of FairSpread to changing Recovery Rate (T)", 
                 xlabel="Recovery Rate",ylabel="FairSpread",legend=["1st to default","2nd to default","3rd to default","4th to default","5th to default"])


#todo In report, make a note of the Companies that were used for this example, being blue chip with liquid CDS... Discuss the influence 
#todo of the chosen reference names on the result.


def TweakCDSSpreads(TweakIndKey,TweakAmountInBps):
    TweakedDataTenorDic = dict()
    TweakedImpProbDic = dict()
    TweakedImpHazdRts = dict()
    TweakedInvPWCDF = dict()
    for i in range(0,5*5,5):
        IndKey = TenorCreditSpreads['Ticker'][i]
        TweakedDataTenorDic[IndKey] = TenorCreditSpreads['DataSR'][i:(i+5)] / 1000
        if IndKey == TweakIndKey:
            TweakedDataTenorDic[TweakIndKey] += TweakAmountInBps
        TweakedDataTenorDic[IndKey] = list(TweakedDataTenorDic[IndKey])
        TweakedImpProbDic[IndKey] = BootstrapImpliedProbalities(0.4,TweakedDataTenorDic[IndKey],TenorCreditSpreads.index)
        Tenors = TweakedImpProbDic[IndKey].index
        BootstrappedSurvProbs = TweakedImpProbDic[IndKey]['ImpliedPrSurv']
        TweakedImpHazdRts[IndKey] = GetHazardsFromP(BootstrappedSurvProbs,Tenors)
        TweakedInvPWCDF[IndKey] = ApproxPWCDFDicFromHazardRates(TweakedImpHazdRts[IndKey],0.01)
    return TweakedDataTenorDic, TweakedImpProbDic, TweakedImpHazdRts, TweakedInvPWCDF

GaussFairSpreadTweakCDS = np.zeros(shape=(5,5),dtype=np.float)
TFairSpreadTweakCDS = np.zeros(shape=(5,5),dtype=np.float)
DeltaGaussFairSpreadTweakCDS = dict()
DeltaTFairSpreadTweakCDS = dict()
CreditTenorTweakAmount = 0.015
for i in range(0,5*5,5):
    IndKey = TenorCreditSpreads['Ticker'][i]
    TweakedDataTenorDic, TweakedImpProbDic, TweakedImpHazdRts, TweakedInvPWCDF = TweakCDSSpreads(IndKey,CreditTenorTweakAmount)
    print("Tweaking the credit spreads for %s and rerunning analysis"%(IndKey))
    GaussFairSpreadTweakCDS[int(i/5)], TFairSpreadTweakCDS[int(i/5)], t17 = FullMCFairSpreadValuation(time.time(),LogRtnCorP,RankCorP,M,HistCreditSpreads,CanonicalMLETransformedHistDataDic,TenorCreditSpreads,TweakedInvPWCDF,
                                                                      DiscountFactorCurve,TweakedImpHazdRts,TweakedDataTenorDic,CDSPaymentTenors,CDSBasketMaturity,name="Tweak Spread for {0} by {1}".format(IndKey,CreditTenorTweakAmount))
    DeltaGaussFairSpreadTweakCDS[IndKey] = GaussFairSpreadTweakCDS[int(i/5)] - GaussFairSpread
    DeltaTFairSpreadTweakCDS[IndKey] = TFairSpreadTweakCDS[int(i/5)] - TFairSpread
CDSRefNamesArr = TenorCreditSpreads['Ticker'][0:25:5]

return_barchart(CDSRefNamesArr,dataDic=DeltaGaussFairSpreadTweakCDS, name="Sensitivity of fair spread to a 150 bps increase in individual reference name CDS spreads (Gauss)",
                     xlabel="Altered reference name",ylabel="Change in Fair Spread")
return_barchart(CDSRefNamesArr,dataDic=DeltaTFairSpreadTweakCDS,name="Sensitivity of fair spread to 150 bps increase in individual reference name CDS spreads (T)",
                     xlabel="Altered reference name",ylabel="Change in Fair Spread")

return_barchart(CDSRefNamesArr,dataDic=DeltaGaussFairSpreadTweakCDS, name="Credit Delta at 150 bps increase for individual reference name CDS spreads (Gauss)",
                     xlabel="Altered reference name",ylabel="Credit delta",ScalingAmount=1/CreditTenorTweakAmount)
return_barchart(CDSRefNamesArr,dataDic=DeltaTFairSpreadTweakCDS,name="Credit Delta at 150 bps increase for individual reference name CDS spreads (T)",
                     xlabel="Altered reference name",ylabel="Credit delta",ScalingAmount=1/CreditTenorTweakAmount)


#!When tweaking the Cor Matrix, remember to keep the matrix symettric M[i,j] = M[j,i] = C
#todo Tweak selected points by +- n bps. And then plot n vs Fair Spread...
#todo Check which of the pairwise correlations the spread is most sensitive to, then change this one over a range of values and plot tweak vs delta fair spread.
TweakedRankCorP = Tweak(RankCorP,(1,2),np.min([0.1, 1-RankCorP[1,2]]))
TweakedLogRtnCorP = Tweak(LogRtnCorP,(1,2),np.min([0.1, 1-LogRtnCorP[1,2]]))
FairSpreadGaussTweakCor,FairSpreadTTweakCor,t12 = FullMCFairSpreadValuation(t11,TweakedLogRtnCorP,TweakedRankCorP,M,HistCreditSpreads,CanonicalMLETransformedHistDataDic,TenorCreditSpreads,InvPWCDF,
                                                                      DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,name="Tweaked Correlation between Barclays & JPMorgan by 0,1")

return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([GaussFairSpread,FairSpreadGaussTweakCor]),"Tweaked Correlation between Barclays & JPMorgan by 0,1 (Gaussian)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])
return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([TFairSpread,FairSpreadTTweakCor]),"Tweaked Correlation between Barclays & JPMorgan by 0,1 (Students T)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])

TweakedRankCorP = Tweak(RankCorP,(2,4),np.min([0.1, 1-RankCorP[2,4]]))
TweakedLogRtnCorP = Tweak(LogRtnCorP,(2,4),np.min([0.1, 1-LogRtnCorP[2,4]]))
FairSpreadGaussTweakCor,FairSpreadTTweakCor,t13 = FullMCFairSpreadValuation(t12,TweakedLogRtnCorP,TweakedRankCorP,M,HistCreditSpreads,CanonicalMLETransformedHistDataDic,TenorCreditSpreads,InvPWCDF,
                                                                      DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,name="Tweaked Correlation between JPMorgan & RBS by 0,1")
return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([GaussFairSpread,FairSpreadGaussTweakCor]),"Tweaked Correlation between JPMorgan & RBS by 0,1 (Gaussian)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])
return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([TFairSpread,FairSpreadTTweakCor]),"Tweaked Correlation between JPMorgan & RBS by 0,1 (Students T)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])

TweakedRankCorP = Tweak(RankCorP,(0,3),np.max([-0.1, -1+RankCorP[0,3]]))
TweakedLogRtnCorP = Tweak(LogRtnCorP,(0,3),np.max([-0.1, -1+LogRtnCorP[0,3]]))
FairSpreadGaussTweakCor,FairSpreadTTweakCor,t14 = FullMCFairSpreadValuation(t13,TweakedLogRtnCorP,TweakedRankCorP,M,HistCreditSpreads,CanonicalMLETransformedHistDataDic,TenorCreditSpreads,InvPWCDF,
                                                                      DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,name="Tweaked Correlation between Deutsche Bank & Goldman Sachs by -0,1")
return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([GaussFairSpread,FairSpreadGaussTweakCor]),"Tweaked Correlation between Deutsche Bank & Goldman Sachs by -0,1 (Gaussian)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])
return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([TFairSpread,FairSpreadTTweakCor]),"Tweaked Correlation between Deutsche Bank & Goldman Sachs by -0,1 (Students T)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])

TweakedRankCorP = Tweak(RankCorP,(1,2),np.min([0.1, 1-RankCorP[1,2]]))
TweakedLogRtnCorP = Tweak(LogRtnCorP,(1,2),np.min([0.1, 1-LogRtnCorP[1,2]]))
TweakedRankCorP = Tweak(RankCorP,(0,4),np.min([0.1, 1-RankCorP[0,4]]))
TweakedLogRtnCorP = Tweak(LogRtnCorP,(0,4),np.min([0.1, 1-LogRtnCorP[0,4]]))
TweakedRankCorP = Tweak(RankCorP,(2,3),np.min([0.1, 1-RankCorP[2,3]]))
TweakedLogRtnCorP = Tweak(LogRtnCorP,(2,3),np.min([0.1, 1-LogRtnCorP[2,3]]))
FairSpreadGaussTweakCor,FairSpreadTTweakCor,t15 = FullMCFairSpreadValuation(t14,TweakedLogRtnCorP,TweakedRankCorP,M,HistCreditSpreads,CanonicalMLETransformedHistDataDic,TenorCreditSpreads,InvPWCDF,
                                                                      DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,name="Tweaked Correlation between Barclays & JPMorgan, Deutsche Bank & RBS and JPMorgan & Goldman Sachs by 0,1")
return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([GaussFairSpread,FairSpreadGaussTweakCor]),"Tweaked Correlation between Barclays & JPMorgan, Deutsche Bank & RBS and JPMorgan & Goldman Sachs by 0,1 (Gaussian)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])
return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([TFairSpread,FairSpreadTTweakCor]),"Tweaked Correlation between Barclays & JPMorgan, Deutsche Bank & RBS and JPMorgan & Goldman Sachs by 0,1 (Students T)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])
#TweakedTLegs = SimulateCDSBasketDefaultsAndValueLegsT(t6,TweakedRankCorP,M,HistCreditSpreads,TransformedHistDataDic,TenorCreditSpreads,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity)
#TweakedGaussLegs = SimulateCDSBasketDefaultsAndValueLegsGauss(TweakedTLegs[2],TweakedLogRtnCorP,M,HistCreditSpreads,TenorCreditSpreads,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity)

#TweakedGaussFairSpread,t9 = CalculateFairSpreadFromLegs(TweakedGaussLegs[0],TweakedGaussLegs[1],M,TweakedGaussLegs[2],"Gauss")
#TweakedTFairSpread,t10 = CalculateFairSpreadFromLegs(TweakedTLegs[0],TweakedTLegs[1],M,TweakedTLegs[2],"T")

TweakedRankCorP = SetArbitrarily(RankCorP,(1,2),0.9)
TweakedLogRtnCorP = SetArbitrarily(LogRtnCorP,(1,2),0.9)
FairSpreadGaussTweakCor,FairSpreadTTweakCor,t15 = FullMCFairSpreadValuation(t14,TweakedLogRtnCorP,TweakedRankCorP,M,HistCreditSpreads,CanonicalMLETransformedHistDataDic,TenorCreditSpreads,InvPWCDF,
                                                                      DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,name="Correlation between Barclays & JPMorgan set to 0.9")
return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([GaussFairSpread,FairSpreadGaussTweakCor]),"Correlation between Barclays & JPMorgan set to 0.9 (Gaussian)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])
return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([TFairSpread,FairSpreadTTweakCor]),"Correlation between Barclays & JPMorgan set to 0.9 (Students T)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])


TweakedRankCorP = SetArbitrarily(RankCorP,(0,3),0.95)
TweakedLogRtnCorP = SetArbitrarily(LogRtnCorP,(0,3),0.95)
FairSpreadGaussTweakCor,FairSpreadTTweakCor,t15 = FullMCFairSpreadValuation(t14,TweakedLogRtnCorP,TweakedRankCorP,M,HistCreditSpreads,CanonicalMLETransformedHistDataDic,TenorCreditSpreads,InvPWCDF,
                                                                      DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,name="Correlation between Deutsche Bank & Goldman Sachs set to 0.95 ")
return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([GaussFairSpread,FairSpreadGaussTweakCor]),"Correlation between Deutsche Bank & Goldman Sachs set to 0.95 (Gaussian)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])
return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([TFairSpread,FairSpreadTTweakCor]),"Correlation between Deutsche Bank & Goldman Sachs set to 0.95 (Students T)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])


TweakedRankCorP = SetArbitrarily(RankCorP,(3,4),0.05)
TweakedLogRtnCorP = SetArbitrarily(LogRtnCorP,(3,4),0.05)
FairSpreadGaussTweakCor,FairSpreadTTweakCor,t15 = FullMCFairSpreadValuation(t14,TweakedLogRtnCorP,TweakedRankCorP,M,HistCreditSpreads,CanonicalMLETransformedHistDataDic,TenorCreditSpreads,InvPWCDF,
                                                                      DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,name="Correlation between Goldman Sachs & RBS set to 0.05")
return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([GaussFairSpread,FairSpreadGaussTweakCor]),"Correlation between Goldman Sachs & RBS set to 0.05 (Gaussian)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])
return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([TFairSpread,FairSpreadTTweakCor]),"Correlation between Goldman Sachs & RBS set to 0.05 (Students T)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])

TweakedRankCorP = SetArbitrarily(RankCorP,(0,3),-0.95)
TweakedLogRtnCorP = SetArbitrarily(LogRtnCorP,(0,3),-0.95)
FairSpreadGaussTweakCor,FairSpreadTTweakCor,t15 = FullMCFairSpreadValuation(t14,TweakedLogRtnCorP,TweakedRankCorP,M,HistCreditSpreads,CanonicalMLETransformedHistDataDic,TenorCreditSpreads,InvPWCDF,
                                                                      DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,name="Correlation between Deutsche Bank & Goldman Sachs set to -0.95")
return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([GaussFairSpread,FairSpreadGaussTweakCor]),"Correlation between Deutsche Bank & Goldman Sachs set to -0.95 (Gaussian)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])
return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([TFairSpread,FairSpreadTTweakCor]),"Correlation between Deutsche Bank & Goldman Sachs set to -0.95 (Students T)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])

TweakedRankCorP = SetWhole2DMatrixArbitrarily(RankCorP,-0.01)
TweakedLogRtnCorP = SetWhole2DMatrixArbitrarily(LogRtnCorP,-0.01)
FairSpreadGaussTweakCor,FairSpreadTTweakCor,t15 = FullMCFairSpreadValuation(t14,TweakedLogRtnCorP,TweakedRankCorP,M,HistCreditSpreads,CanonicalMLETransformedHistDataDic,TenorCreditSpreads,InvPWCDF,
                                                                      DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,name="Correlation between all reference names set to -0.01")
return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([GaussFairSpread,FairSpreadGaussTweakCor]),"Correlation between all reference names set to -0.01 (Gaussian)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])
return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([TFairSpread,FairSpreadTTweakCor]),"Correlation between all reference names set to -0.01 (Students T)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])

TweakedRankCorP = SetWhole2DMatrixArbitrarily(RankCorP,-0.99)
TweakedLogRtnCorP = SetWhole2DMatrixArbitrarily(LogRtnCorP,-0.99)
FairSpreadGaussTweakCor,FairSpreadTTweakCor,t15 = FullMCFairSpreadValuation(t14,TweakedLogRtnCorP,TweakedRankCorP,M,HistCreditSpreads,CanonicalMLETransformedHistDataDic,TenorCreditSpreads,InvPWCDF,
                                                                      DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,name="Correlation between all reference names set to -0.99")
return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([GaussFairSpread,FairSpreadGaussTweakCor]),"Correlation between all reference names set to -0.99 (Gaussian)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])
return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([TFairSpread,FairSpreadTTweakCor]),"Correlation between all reference names set to -0.99 (Students T)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])

TweakedRankCorP = SetWhole2DMatrixArbitrarily(RankCorP,0.99)
TweakedLogRtnCorP = SetWhole2DMatrixArbitrarily(LogRtnCorP,0.99)
FairSpreadGaussTweakCor,FairSpreadTTweakCor,t15 = FullMCFairSpreadValuation(t14,TweakedLogRtnCorP,TweakedRankCorP,M,HistCreditSpreads,CanonicalMLETransformedHistDataDic,TenorCreditSpreads,InvPWCDF,
                                                                      DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,name="Correlation between all reference names set to 0.99")
return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([GaussFairSpread,FairSpreadGaussTweakCor]),"Correlation between all reference names set to 0.99 (Gaussian)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])
return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([TFairSpread,FairSpreadTTweakCor]),"Correlation between all reference names set to 0.99 (Students T)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])

print("Enter any key to finish. (Debug Point)")

save_all_figs()
userIn = input()

debug = True
