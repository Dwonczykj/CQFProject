#! TO BUILD CYTHON EXTENSIONS, RUN "python setup.py build_ext --inplace --line-directives" IN A SHELL AT THE PROJECT ROOT. 
#!Ensure that the version of C Compiler you have matches the version used to compile the python environment that you are using.

import pandas as pd
import os
#import datetime
#import collections
import time

from HazardRates import *
from Returns import *
from EmpiricalFunctions import *
from plotting import plot_histogram_array, showAllPlots, plot_codependence_scatters, Plot_Converging_Averages, return_lineChart, return_barchart, return_lineChart_dates
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
#return_lineChart(np.arange(0,5.01,0.01,dtype=np.float),[pd.Series(np.arange(0,5.01,0.01,dtype=np.float)).apply(DiscountFactorCurve["Sonia"])],"Sonia Discount Curve",xlabel="Time/years",ylabel="Discount Factor")
t3 = time.time()
print("Took %.10f seconds to Grab Discount Factors." % (t3 - t2))

#calc log returns from historical data and then calc corrs on it.
HistDataDic = dict()
TransformedHistDataDic = dict()
CanonicallyTransformedHistDataDic = dict()
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
    #!Take 5 day averages of the data and then look at differences between them to reduce the influence of outliers.
    #!Then for pearson, we look at the correlation of the raw data
    #! For spearman, we rank the data using the Empirical CDF and then look at the correlation of the ranks.
    
    EmpFnForHistSpread[IndKey] = Empirical_StepWise_CDF(quickSort(DifferencesDic[IndKey].values))
    SemiParamametric[IndKey] = SemiParametricCDFFit(list(HistDataDic[IndKey]['Spreads']),np.percentile(HistDataDic[IndKey]['Spreads'],95),quickSort(DifferencesDic[IndKey].values),EmpFnForHistSpread[IndKey])
    TransformedHistDataDic[IndKey] = pd.Series(DifferencesDic[IndKey].values).apply(EmpFnForHistSpread[IndKey])#TODO: Add extreme value theory for tails of ECDF by replacing EmpFin with Kernel Smoothed function.
    CanonicallyTransformedHistDataDic[IndKey] = HistDataDic[IndKey]['Spreads'].apply(TransformedHistDataDic[IndKey]['%.10f'%(np.percentile(HistDataDic[IndKey]['Spreads'],95))][0]) #todo compare to empcdf in plot
    #TransformedHistDataLengthDic[IndKey] = len(TransformedHistDataDic[IndKey])
    #HistDefaults[IndKey] = pd.Series(HistCreditSpreads[IndKey].values).apply(Bootstrap5yrDP)
    
#!return_lineChart_dates(HistCreditSpreads['Date'].values,[
#    list(HistCreditSpreads[HistCreditSpreads.columns[1]]),
#    list(HistCreditSpreads[HistCreditSpreads.columns[2]]), 
#    list(HistCreditSpreads[HistCreditSpreads.columns[3]]), 
#    list(HistCreditSpreads[HistCreditSpreads.columns[4]]), 
#    list(HistCreditSpreads[HistCreditSpreads.columns[5]])
#    ],name="Historical Credit Spreads Data", xlabel="Historical Date", ylabel="Spread", legend=list(HistCreditSpreads.columns[1:]))
    
t4 = time.time()
print("Took %.10f seconds to grab Historical Spreads and Transform the data by its Empirical CDF." % (t4 - t3))  
#ResidualCorP = CorP(ResidualsLogReturnsDic) 
#!plot_histogram_array(LogReturnsDic, "Weekly Log Returns")
#!plot_histogram_array(DifferencesDic, "Weekly Differences")
#!plot_histogram_array(TransformedHistDataDic,"Inverse ECDF (Rank)")
#!showAllPlots()
t4a = time.time()
print("Took %.10f seconds to print Transformed Spreads Histograms" % (t4a - t4))
#DefaultCorP = CorP(HistDefaults)

LogRtnCorP = CorP(LogReturnsDic)    #near correlation without kernel smoothing for gaussian copula 
pdCorLogRtnP = convertToLaTeX(pd.DataFrame(LogRtnCorP, dtype=np.float))
#Transform HistCreditSpreads by its own empirical distn and then calc corrln on U to get Rank Corr. This is the defn of Spearmans Rho  
#!Consider also using KENDALLS TAU AS AN ALTERNATIVE FORMULATION FOR THE RANK CORRELATION
RankCorP = CorP(TransformedHistDataDic) #todo: Need to have a lag or take a subset of this data
pdCorRankP = convertToLaTeX(pd.DataFrame(RankCorP, dtype=np.float))
diffCor = RankCorP - LogRtnCorP
pdCorDiffs = convertToLaTeX(pd.DataFrame(diffCor, dtype=np.float))
t5 = time.time()
print("Took %.10f seconds to calculate Correlation Matrices." % (t5 - t4a))
#TODO ADD MonteCarlo Sim here to keep sampling U and then plugging into following which should all be one function that takes the above as params and outputs the legs dictionaries for each copula.
NumbGen = SobolNumbers()
NumbGen.initialise(LogRtnCorP.shape[0])
for i in range(0,5000):
    NumbGen.Generate()    
t6 = time.time()
print("Took %.10f seconds to init and run 5000x%d iterations of sobol numbers." % ((t6 - t5) , LogRtnCorP.shape[0]))



#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!MONTE CARLO SIMULATION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

CDSBasketMaturity = 5.0
CDSPaymentTenor = 0.25
CDSPaymentTenors = np.arange(CDSPaymentTenor,CDSBasketMaturity+CDSPaymentTenor,CDSPaymentTenor,dtype=np.float)
M = 5000

#TLegs = SimulateCDSBasketDefaultsAndValueLegsT(t6,RankCorP,NumbGen,M,HistCreditSpreads,TransformedHistDataDic,TenorCreditSpreads,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity)
#GaussLegs = SimulateCDSBasketDefaultsAndValueLegsGauss(TLegs[2],LogRtnCorP,NumbGen,M,HistCreditSpreads,TenorCreditSpreads,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity)

#GaussFairSpread,t9 = CalculateFairSpreadFromLegs(GaussLegs[0],GaussLegs[1],M,GaussLegs[2],"Gauss")
#TFairSpread,t10 = CalculateFairSpreadFromLegs(TLegs[0],TLegs[1],M,TLegs[2],"T")
#todo: Add Variance conversion to Simulation


GaussFairSpread,TFairSpread,t10 = FullMCFairSpreadValuation(t6,LogRtnCorP,RankCorP,NumbGen,M,HistCreditSpreads,TransformedHistDataDic,TenorCreditSpreads,InvPWCDF,
                                                            DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity)
latexFairSpreads = convertToLaTeX(pd.DataFrame(data=np.array([GaussFairSpread,TFairSpread],dtype=np.float), index = ["Gaussian", "Student's T"], dtype=np.float))

#todo Compare fair spreads with current spreads from ref names in plot...

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!CHECK OF FAIR SPREADS BY MONTE CARLO SIMULATION USING ESTIMATED FAIR SPREAD!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

TLegsWFairSpread = SimulateCDSBasketDefaultsAndValueLegsT(t10,RankCorP,NumbGen,M,HistCreditSpreads,TransformedHistDataDic,TenorCreditSpreads,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,TFairSpread)
GaussLegsWFairSpread = SimulateCDSBasketDefaultsAndValueLegsGauss(TLegsWFairSpread[2],LogRtnCorP,NumbGen,M,HistCreditSpreads,TenorCreditSpreads,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,GaussFairSpread)
t11 = GaussLegsWFairSpread[2]
#GaussLegsWFairSpread,TLegsWFairSpread,t11 = FullMCFairSpreadValuation(t10,LogRtnCorP,RankCorP,NumbGen,M,HistCreditSpreads,TransformedHistDataDic,TenorCreditSpreads,InvPWCDF,
#                                                                      DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,TFairSpread,GaussFairSpread)
TFairDiff = TLegsWFairSpread[0] - TLegsWFairSpread[1] #!Compensation Legs - Premium Legs
#todo: Average of Premium Legs should be similar to average of compensation legs
GaussFairDiff = GaussLegsWFairSpread[0] - GaussLegsWFairSpread[1] #!Compensation Legs - Premium Legs
#todo Plot the distribution of each of TFairDiff[:1]
gaussCheckFairSpreadDic = dict()
tCheckFairSpreadDic = dict()
for iten in range(0,TFairDiff.shape[1]):
    gaussCheckFairSpreadDic["%d-th to default basket CDS" % (iten+1)] = GaussFairDiff[:,iten]
    tCheckFairSpreadDic["%d-th to default basket CDS" % (iten+1)] = TFairDiff[:,iten]
plot_histogram_array(gaussCheckFairSpreadDic,"CDS Basket Outcome Using Gaussian-Copula")
plot_histogram_array(tCheckFairSpreadDic,"CDS Basket Outcome Using T-Copula")
#For Kendalls Tau, we have X1 and X2 from the data with empirical cdf, we then also simulate X3 and X4 from the emp distributions of X1 and x2 resp. We then defn pTau := E[sign((x1-x3)*(x2-x4))] 
#Now Consider altering the tail dependence of the copulas upper and lower separately.

#Step 1: Calculate the empirical cdf for each hist Ref name, both using kernel smoothing(pdf -> cdf) and just empirical stewpise constant cdf.
#todo: Kernel smoothing
#Step 2: Transform historical spreads, X, by their CDF F(X) = U
#Step 3: Obtain Normally, T distributed R.v.s by applying the respective inverse CDF to the U.
#Step 4: Calculate Correlations from Standardised Credit Spread returns / Default Probs and use these near and rank correlation matrices to simulate Normal or T distributed variables respectively.
#Step 5: Compare the difference between 1-3 and 4.
#showAllPlots()



#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  RISK AND SENSITIVITY ANALYSIS   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#todo Tweak the Recovery rate of the cds to see how this influences the fair spread. WE should see the spread changing. Note: The fact that we want to 
#todo change the recovery rate suggests that the initial simulation of the legs should not include a spread as this would be linked to a
#todo certain market accepted recovery rate of most likely 40%.

itera = 0
Rrng = np.arange(0.25,0.85,0.1)
GaussFairSpreadTweakR = np.zeros(shape=(len(Rrng),5),dtype=np.float)
TFairSpreadTweakR = np.zeros(shape=(len(Rrng),5),dtype=np.float)
for RAlt in Rrng:
    GaussFairSpreadTweakR[itera], TFairSpreadTweakR[itera], t16 = FullMCFairSpreadValuation(time.time(),LogRtnCorP,RankCorP,NumbGen,M,HistCreditSpreads,TransformedHistDataDic,TenorCreditSpreads,InvPWCDF,
                                                                      DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,TFairSpread,GaussFairSpread,RAlt)
    itera += 1

return_lineChart(Rrng,GaussFairSpreadTweakR.transpose(),"Sensitivity of FairSpread to changing Recovery Rate (Gauss)", 
                 xlabel="Recovery Rate",ylabel="FairSpread",legend=["1st to default","2nd to default","3rd to default","4th to default","5th to default"])
return_lineChart(Rrng,TFairSpreadTweakR.transpose(),"Sensitivity of FairSpread to changing Recovery Rate (T)", 
                 xlabel="Recovery Rate",ylabel="FairSpread",legend=["1st to default","2nd to default","3rd to default","4th to default","5th to default"])

#todo In report, make a note of the Companies that were used for this example, being blue chip with liquid CDS... Discuss the influence 
#todo of the chosen reference names on the result.

#todo Tweak the CDS spreads for Reference names in INDIVIDUAL TESTS to see the effect that those spreads for that ref name has on the FAIR SPREAD. 
#todo ie tweak all DB spreads for each tenor by +20bps.
#todo Do same as with correlation and find out which reference name the fair spread is most sensitive to and then tweak this ref name in comparison


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
    #todo: Fix issue here where Tweaked Fair spreads are completely out
    GaussFairSpreadTweakCDS[int(i/5)], TFairSpreadTweakCDS[int(i/5)], t17 = FullMCFairSpreadValuation(time.time(),LogRtnCorP,RankCorP,NumbGen,M,HistCreditSpreads,TransformedHistDataDic,TenorCreditSpreads,TweakedInvPWCDF,
                                                                      DiscountFactorCurve,TweakedImpHazdRts,TweakedDataTenorDic,CDSPaymentTenors,CDSBasketMaturity)
    DeltaGaussFairSpreadTweakCDS[IndKey] = GaussFairSpreadTweakCDS[int(i/5)] - GaussFairSpread
    DeltaTFairSpreadTweakCDS[IndKey] = TFairSpreadTweakCDS[int(i/5)] - TFairSpread
CDSRefNamesArr = TenorCreditSpreads['Ticker'][0:25:5]

#todo return a barchart, refname on the xaxis with each ref name having 5 bars for 1st to 5th to default (key 1 -> 1st to default, ..., 5 -> 5th to default). 
#todo DeltaFairSpread on the y axis.
#dataKey = ["1 -> 1st to default","2 -> 2nd to default","3 -> 3rd to default","4 -> 4th to default","5 -> 5th to default"]
return_barchart(CDSRefNamesArr,dataDic=DeltaGaussFairSpreadTweakCDS, name="Sensitivity of fair spread to a 150 bps increase in individual reference name CDS spreads (Gauss)",
                     xlabel="Altered reference name",ylabel="Change in Fair Spread")
return_barchart(CDSRefNamesArr,dataDic=DeltaTFairSpreadTweakCDS,name="Sensitivity of fair spread to 150 bps increase in individual reference name CDS spreads (T)",
                     xlabel="Altered reference name",ylabel="Change in Fair Spread")

return_barchart(CDSRefNamesArr,dataDic=DeltaGaussFairSpreadTweakCDS, name="Credit Delta at 150 bps increase for individual reference name CDS spreads (Gauss)",
                     xlabel="Altered reference name",ylabel="Credit delta",ScalingAmount=1/CreditTenorTweakAmount)
return_barchart(CDSRefNamesArr,dataDic=DeltaTFairSpreadTweakCDS,name="Credit Delta at 150 bps increase for individual reference name CDS spreads (T)",
                     xlabel="Altered reference name",ylabel="Credit delta",ScalingAmount=1/CreditTenorTweakAmount)


#todo Alter/Tweak the correlation matrix for both RankCor and Linear Cor INDIVIDUALLY and compare how sensitive the FAIR SPREAD 
#todo for the kth to defualt ref name is to the correlation matrix.

#!When tweaking the Cor Matrix, remember to keep the matrix symettric M[i,j] = M[j,i] = C
#todo Tweak selected points by +- n bps. And then plot n vs Fair Spread...
#todo Check which of the pairwise correlations the spread is most sensitive to, then change this one over a range of values and plot tweak vs delta fair spread.
TweakedRankCorP = Tweak(RankCorP,(1,2),np.min([0.1, 1-RankCorP[1,2]]))
TweakedLogRtnCorP = Tweak(LogRtnCorP,(1,2),np.min([0.1, 1-LogRtnCorP[1,2]]))
FairSpreadGaussTweakCor,FairSpreadTTweakCor,t12 = FullMCFairSpreadValuation(t11,TweakedLogRtnCorP,TweakedRankCorP,NumbGen,M,HistCreditSpreads,TransformedHistDataDic,TenorCreditSpreads,InvPWCDF,
                                                                      DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity)

return_lineChart(np.arange(1,6,1,dtype=np.int),np.array([GaussFairSpread,FairSpreadGaussTweakCor]),"Tweaked Correlation between Barclays & JPMorgan by 0,1 (Gaussian)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])
return_lineChart(np.arange(1,6,1,dtype=np.int),np.array([TFairSpread,FairSpreadTTweakCor]),"Tweaked Correlation between Barclays & JPMorgan by 0,1 (Students T)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])

TweakedRankCorP = Tweak(RankCorP,(2,4),np.min([0.1, 1-RankCorP[2,4]]))
TweakedLogRtnCorP = Tweak(LogRtnCorP,(2,4),np.min([0.1, 1-LogRtnCorP[2,4]]))
FairSpreadGaussTweakCor,FairSpreadTTweakCor,t13 = FullMCFairSpreadValuation(t12,TweakedLogRtnCorP,TweakedRankCorP,NumbGen,M,HistCreditSpreads,TransformedHistDataDic,TenorCreditSpreads,InvPWCDF,
                                                                      DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity)
return_lineChart(np.arange(1,6,1,dtype=np.int),np.array([GaussFairSpread,FairSpreadGaussTweakCor]),"Tweaked Correlation between JPMorgan & RBS by 0,1 (Gaussian)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])
return_lineChart(np.arange(1,6,1,dtype=np.int),np.array([TFairSpread,FairSpreadTTweakCor]),"Tweaked Correlation between JPMorgan & RBS by 0,1 (Students T)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])

TweakedRankCorP = Tweak(RankCorP,(0,3),np.max([-0.1, -1+RankCorP[0,3]]))
TweakedLogRtnCorP = Tweak(LogRtnCorP,(0,3),np.max([-0.1, -1+LogRtnCorP[0,3]]))
FairSpreadGaussTweakCor,FairSpreadTTweakCor,t14 = FullMCFairSpreadValuation(t13,TweakedLogRtnCorP,TweakedRankCorP,NumbGen,M,HistCreditSpreads,TransformedHistDataDic,TenorCreditSpreads,InvPWCDF,
                                                                      DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity)
return_lineChart(np.arange(1,6,1,dtype=np.int),np.array([GaussFairSpread,FairSpreadGaussTweakCor]),"Tweaked Correlation between Deutsche Bank & Goldman Sachs by -0,1 (Gaussian)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])
return_lineChart(np.arange(1,6,1,dtype=np.int),np.array([TFairSpread,FairSpreadTTweakCor]),"Tweaked Correlation between Deutsche Bank & Goldman Sachs by -0,1 (Students T)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])

TweakedRankCorP = Tweak(RankCorP,(1,2),np.min([0.1, 1-RankCorP[1,2]]))
TweakedLogRtnCorP = Tweak(LogRtnCorP,(1,2),np.min([0.1, 1-LogRtnCorP[1,2]]))
TweakedRankCorP = Tweak(RankCorP,(0,4),np.min([0.1, 1-RankCorP[0,4]]))
TweakedLogRtnCorP = Tweak(LogRtnCorP,(0,4),np.min([0.1, 1-LogRtnCorP[0,4]]))
TweakedRankCorP = Tweak(RankCorP,(2,3),np.min([0.1, 1-RankCorP[2,3]]))
TweakedLogRtnCorP = Tweak(LogRtnCorP,(2,3),np.min([0.1, 1-LogRtnCorP[2,3]]))
FairSpreadGaussTweakCor,FairSpreadTTweakCor,t15 = FullMCFairSpreadValuation(t14,TweakedLogRtnCorP,TweakedRankCorP,NumbGen,M,HistCreditSpreads,TransformedHistDataDic,TenorCreditSpreads,InvPWCDF,
                                                                      DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity)
return_lineChart(np.arange(1,6,1,dtype=np.int),np.array([GaussFairSpread,FairSpreadGaussTweakCor]),"Tweaked Correlation between Barclays & JPMorgan, Deutsche Bank & RBS and JPMorgan & Goldman Sachs by 0,1 (Gaussian)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])
return_lineChart(np.arange(1,6,1,dtype=np.int),np.array([TFairSpread,FairSpreadTTweakCor]),"Tweaked Correlation between Barclays & JPMorgan, Deutsche Bank & RBS and JPMorgan & Goldman Sachs by 0,1 (Students T)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])
#TweakedTLegs = SimulateCDSBasketDefaultsAndValueLegsT(t6,TweakedRankCorP,NumbGen,M,HistCreditSpreads,TransformedHistDataDic,TenorCreditSpreads,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity)
#TweakedGaussLegs = SimulateCDSBasketDefaultsAndValueLegsGauss(TweakedTLegs[2],TweakedLogRtnCorP,NumbGen,M,HistCreditSpreads,TenorCreditSpreads,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity)

#TweakedGaussFairSpread,t9 = CalculateFairSpreadFromLegs(TweakedGaussLegs[0],TweakedGaussLegs[1],M,TweakedGaussLegs[2],"Gauss")
#TweakedTFairSpread,t10 = CalculateFairSpreadFromLegs(TweakedTLegs[0],TweakedTLegs[1],M,TweakedTLegs[2],"T")

TweakedRankCorP = SetArbitrarily(RankCorP,(1,2),0.9)
TweakedLogRtnCorP = SetArbitrarily(LogRtnCorP,(1,2),0.9)
FairSpreadGaussTweakCor,FairSpreadTTweakCor,t15 = FullMCFairSpreadValuation(t14,TweakedLogRtnCorP,TweakedRankCorP,NumbGen,M,HistCreditSpreads,TransformedHistDataDic,TenorCreditSpreads,InvPWCDF,
                                                                      DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity)
return_lineChart(np.arange(1,6,1,dtype=np.int),np.array([GaussFairSpread,FairSpreadGaussTweakCor]),"Correlation between Barclays & JPMorgan set to 0.9 (Gaussian)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])
return_lineChart(np.arange(1,6,1,dtype=np.int),np.array([TFairSpread,FairSpreadTTweakCor]),"Correlation between Barclays & JPMorgan set to 0.9 (Students T)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])


TweakedRankCorP = SetArbitrarily(RankCorP,(0,3),0.95)
TweakedLogRtnCorP = SetArbitrarily(LogRtnCorP,(0,3),0.95)
FairSpreadGaussTweakCor,FairSpreadTTweakCor,t15 = FullMCFairSpreadValuation(t14,TweakedLogRtnCorP,TweakedRankCorP,NumbGen,M,HistCreditSpreads,TransformedHistDataDic,TenorCreditSpreads,InvPWCDF,
                                                                      DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity)
return_lineChart(np.arange(1,6,1,dtype=np.int),np.array([GaussFairSpread,FairSpreadGaussTweakCor]),"Correlation between Deutsche Bank & Goldman Sachs set to 0.95 (Gaussian)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])
return_lineChart(np.arange(1,6,1,dtype=np.int),np.array([TFairSpread,FairSpreadTTweakCor]),"Correlation between Deutsche Bank & Goldman Sachs set to 0.95 (Students T)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])


TweakedRankCorP = SetArbitrarily(RankCorP,(3,4),0.05)
TweakedLogRtnCorP = SetArbitrarily(LogRtnCorP,(3,4),0.05)
FairSpreadGaussTweakCor,FairSpreadTTweakCor,t15 = FullMCFairSpreadValuation(t14,TweakedLogRtnCorP,TweakedRankCorP,NumbGen,M,HistCreditSpreads,TransformedHistDataDic,TenorCreditSpreads,InvPWCDF,
                                                                      DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity)
return_lineChart(np.arange(1,6,1,dtype=np.int),np.array([GaussFairSpread,FairSpreadGaussTweakCor]),"Correlation between Goldman Sachs & RBS set to 0.05 (Gaussian)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])
return_lineChart(np.arange(1,6,1,dtype=np.int),np.array([TFairSpread,FairSpreadTTweakCor]),"Correlation between Goldman Sachs & RBS set to 0.05 (Students T)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])

TweakedRankCorP = SetArbitrarily(RankCorP,(0,3),-0.95)
TweakedLogRtnCorP = SetArbitrarily(LogRtnCorP,(0,3),-0.95)
FairSpreadGaussTweakCor,FairSpreadTTweakCor,t15 = FullMCFairSpreadValuation(t14,TweakedLogRtnCorP,TweakedRankCorP,NumbGen,M,HistCreditSpreads,TransformedHistDataDic,TenorCreditSpreads,InvPWCDF,
                                                                      DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity)
return_lineChart(np.arange(1,6,1,dtype=np.int),np.array([GaussFairSpread,FairSpreadGaussTweakCor]),"Correlation between Deutsche Bank & Goldman Sachs set to -0.95 (Gaussian)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])
return_lineChart(np.arange(1,6,1,dtype=np.int),np.array([TFairSpread,FairSpreadTTweakCor]),"Correlation between Deutsche Bank & Goldman Sachs set to -0.95 (Students T)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])

TweakedRankCorP = SetWhole2DMatrixArbitrarily(RankCorP,-0.01)
TweakedLogRtnCorP = SetWhole2DMatrixArbitrarily(LogRtnCorP,-0.01)
FairSpreadGaussTweakCor,FairSpreadTTweakCor,t15 = FullMCFairSpreadValuation(t14,TweakedLogRtnCorP,TweakedRankCorP,NumbGen,M,HistCreditSpreads,TransformedHistDataDic,TenorCreditSpreads,InvPWCDF,
                                                                      DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity)
return_lineChart(np.arange(1,6,1,dtype=np.int),np.array([GaussFairSpread,FairSpreadGaussTweakCor]),"Correlation between all reference names set to -0.01 (Gaussian)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])
return_lineChart(np.arange(1,6,1,dtype=np.int),np.array([TFairSpread,FairSpreadTTweakCor]),"Correlation between all reference names set to -0.01 (Students T)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])

TweakedRankCorP = SetWhole2DMatrixArbitrarily(RankCorP,-0.99)
TweakedLogRtnCorP = SetWhole2DMatrixArbitrarily(LogRtnCorP,-0.99)
FairSpreadGaussTweakCor,FairSpreadTTweakCor,t15 = FullMCFairSpreadValuation(t14,TweakedLogRtnCorP,TweakedRankCorP,NumbGen,M,HistCreditSpreads,TransformedHistDataDic,TenorCreditSpreads,InvPWCDF,
                                                                      DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity)
return_lineChart(np.arange(1,6,1,dtype=np.int),np.array([GaussFairSpread,FairSpreadGaussTweakCor]),"Correlation between all reference names set to -0.99 (Gaussian)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])
return_lineChart(np.arange(1,6,1,dtype=np.int),np.array([TFairSpread,FairSpreadTTweakCor]),"Correlation between all reference names set to -0.99 (Students T)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])

TweakedRankCorP = SetWhole2DMatrixArbitrarily(RankCorP,0.99)
TweakedLogRtnCorP = SetWhole2DMatrixArbitrarily(LogRtnCorP,0.99)
FairSpreadGaussTweakCor,FairSpreadTTweakCor,t15 = FullMCFairSpreadValuation(t14,TweakedLogRtnCorP,TweakedRankCorP,NumbGen,M,HistCreditSpreads,TransformedHistDataDic,TenorCreditSpreads,InvPWCDF,
                                                                      DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity)
return_lineChart(np.arange(1,6,1,dtype=np.int),np.array([GaussFairSpread,FairSpreadGaussTweakCor]),"Correlation between all reference names set to 0.99 (Gaussian)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])
return_lineChart(np.arange(1,6,1,dtype=np.int),np.array([TFairSpread,FairSpreadTTweakCor]),"Correlation between all reference names set to 0.99 (Students T)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])

#todo For Credit Tweaks MEASURE the affect on EACH of the kth to default basket cds. ie (change in credit spread <Tweak>)/(change in fair spread of the KTH to default basket only.) -> Use different tweak amounts

#TODO FIT THE FUCKING TAILS WITH A GENERALISED PARETO DISTRIBUTION


debug = True
