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
from plotting import Plotter
#from Copulae import MultVarGaussianCopula, MultVarTDistnCopula
from Sorting import *
from LowDiscrepancyNumberGenerators import SobolNumbers
from SimulateLegs import TCopula_DF_MLE
from Logger import convertToLaTeX, printf
from MonteCarloCDSBasketPricer import CalculateFairSpreadFromLegs, SimulateCDSBasketDefaultsAndValueLegsGauss, SimulateCDSBasketDefaultsAndValueLegsT, FullMCFairSpreadValuation
from ProbabilityIntegralTransform import *
from scipy.stats import t, expon
from scipy.special import gamma, gammaln
from math import *
import multiprocessing as mp
import random

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
#plotter.plot_codependence_scatters(dict(enumerate(DummyRUs.transpose())),"D%","D%")
#NumbGen = SobolNumbers()
#NumbGen.initialise(5)
#DummyRUs = np.zeros(shape=(l,5))
#for j in range(0,l):
#    DummyRUs[j] = np.array(NumbGen.Generate(),np.float)
#plotter.plot_codependence_scatters(dict(enumerate(DummyRUs.transpose())),"D%","D%")

if __name__ == '__main__':
    cwd = os.getcwd()
    Datafp = cwd + '/FinalProjData.xlsx'
    TenorCreditSpreads = pd.read_excel(Datafp,'TenorCreditSpreads',0)
    HistCreditSpreads = pd.read_excel(Datafp,'HistoricalCreditSpreads',0)
    DiscountCurves = pd.read_excel(Datafp,'DiscountCurves',0)
    plotter = Plotter()
    BPS_TO_NUMBER = 10000

    #with mp.Manager() as manager:
    #    working_q = manager.Queue()
    #    output_q = manager.Queue()
    working_q = mp.Queue()
    output_q = mp.Queue()
        
    t1 = time.time()

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
    plotter.return_lineChart(np.arange(0,5.01,0.01,dtype=np.float),[pd.Series(np.arange(0,5.01,0.01,dtype=np.float)).apply(DiscountFactorCurve["Sonia"])],"Sonia Discount Curve",xlabel="Time/years",ylabel="Discount Factor")
    t2 = time.time()
    print("Took %.10f seconds to Grab Discount Factors." % (t2 - t1))

    DataTenorDic = dict()
    ImpProbDic = dict()
    ImpHazdRts = dict()
    InvPWCDF = dict()
    PWCDF = dict()
    ReferenceNameList = list()
    hazardLines = [[] for i in range(0,5)]
    hazardLegend = ["" for i in range(0,5)]
    for i in range(0,5*5,5):
        IndKey = TenorCreditSpreads['Ticker'][i]
        ReferenceNameList.append(IndKey)
        DataTenorDic[IndKey] = list(TenorCreditSpreads['DataSR'][i:(i+5)] / BPS_TO_NUMBER)
        ImpProbDic[IndKey] = BootstrapImpliedProbalities(0.4,DataTenorDic[IndKey],DiscountFactorCurve["Sonia"])
        Tenors = ImpProbDic[IndKey].index
        BootstrappedSurvProbs = ImpProbDic[IndKey]['ImpliedPrSurv']
        ImpHazdRts[IndKey] = GetHazardsFromP(BootstrappedSurvProbs,Tenors)
        hazardLegend[int(i/5)] = IndKey
        hazardLines[int(i/5)] = [0] + ImpHazdRts[IndKey]
        InvPWCDF[IndKey], PWCDF[IndKey] = ApproxPWCDFDicFromHazardRates(ImpHazdRts[IndKey],0.01)
    plotter.return_lineChart(np.arange(0,6,1,dtype=np.int),np.array(hazardLines),"Hazard rates",xlabel="Time/years",ylabel="Hazard rate", legend=hazardLegend)
    plotter.return_lineChart(np.arange(0,6,1,dtype=np.int),[np.array(ImpProbDic[IndKey]['ImpliedPrSurv']) for IndKey in ReferenceNameList],"Implied default probabilities",xlabel="Time/years",ylabel="Default probability", legend=hazardLegend)
    plotter.return_lineChart([1,2,3,4,5],list(DataTenorDic.values()),name="Credit Spreads", xlabel="Term", ylabel="Spread", legend=list(DataTenorDic.keys()))
    t3 = time.time()

    print("Took %.10f seconds to Grab Tenor Data and Init Inverse Empirical CDF functions." % (t3 - t2))
    #TenorCreditSpreads["ToDate"] = pd.to_datetime(TenorCreditSpreads["ToDate"])



    #calc log returns from historical data and then calc corrs on it.
    HistDataDic = dict()
    CanonicalMLETransformedHistDataDic = dict()
    SemiParamTransformedCDFHistDataDic = dict()
    SemiParamTransformedPDFHistDataDic = dict()
    EmpFnForHistSpread = dict()
    SemiParamametric = dict()
    LogReturnsDic = dict()
    DifferencesDic = dict()
    ConsecDiffsDic = dict()

    #ResidualsLogReturnsDic = dict()
    #def Bootstrap5yrDP(spread):
    #    return BootstrapImpliedProbalities(0.4,pd.Series([spread]),5)['ImpliedPrSurv'][1]

    for i in range(1,6):
        IndKey = HistCreditSpreads.columns[i]
        HistDataDic[IndKey] = pd.DataFrame({ 'Spreads': HistCreditSpreads[IndKey].values/BPS_TO_NUMBER}, dtype=np.float, index = HistCreditSpreads['Date'].values)
        plotter.QQPlot(HistDataDic[IndKey]['Spreads'],"QQ-Plot {0}".format(IndKey))
        #todo: Measure the autocorrelation of the non touched returns to give an argument for the need for standardised residuals.
        #From http://uk.mathworks.com/help/econ/examples/using-extreme-value-theory-and-copulas-to-evaluate-market-risk.html#d119e9608:
        #"Comparing the ACFs of the standardized residuals to the corresponding ACFs of the raw returns reveals that the standardized residuals are now approximately i.i.d., thereby far more amenable to subsequent tail estimation."
        LogReturnsDic[IndKey] = LogReturns(HistDataDic[IndKey]['Spreads'],lag=5,jump=5,averageJumpPeriod=True)
        DifferencesDic[IndKey] = AbsoluteDifferences(HistDataDic[IndKey]['Spreads'],jump=5,averageJumpPeriod=True)
        ConsecDiffsDic[IndKey] = AbsoluteDifferences(HistDataDic[IndKey]['Spreads'])
        EmpFnForHistSpread[IndKey] = Empirical_StepWise_CDF(quickSort(DifferencesDic[IndKey].values))
        u_gpdthreashold = np.percentile(HistDataDic[IndKey]['Spreads'],95)
        SemiParamametric[IndKey] = SemiParametricCDFFit(list(HistDataDic[IndKey]['Spreads']),u_gpdthreashold, True, "SemiParametricFit_%s"%(IndKey), "Historical Spreads", "Distribution")
        CanonicalMLETransformedHistDataDic[IndKey] = pd.Series(DifferencesDic[IndKey].values).apply(EmpFnForHistSpread[IndKey])
        SemiParamTransformedCDFHistDataDic[IndKey] =  pd.Series(SemiParamametric[IndKey]['%.10f'%(u_gpdthreashold)][0])
        SemiParamTransformedPDFHistDataDic[IndKey] =  pd.Series(SemiParamametric[IndKey]['%.10f'%(u_gpdthreashold)][1])

    AltHistData_SpreadsOnly = dict()
    AltEmpFnForHistSpread = dict()
    AltSemiParamametric = dict()
    AltLogReturnsDic = dict()
    AltDifferencesDic = dict()
    AltCanonicalMLETransformedHistDataDic = dict()
    AltSemiParamTransformedCDFHistDataDic = dict()
    AltSemiParamTransformedPDFHistDataDic = dict()
    AltEmpFnForHistSpread = dict()
    AltSemiParamametric = dict()
    for i in range(1,6):
        IndKey = HistCreditSpreads.columns[i]
        _forAlthistSpreads = HistDataDic[IndKey]['Spreads']
        AltHistData_SpreadsOnly[IndKey] = np.zeros(shape=(len(_forAlthistSpreads)))
        AltHistData_SpreadsOnly[IndKey][0] = _forAlthistSpreads[0]
        for l in range(1,len(_forAlthistSpreads)):
            AltHistData_SpreadsOnly[IndKey][l] = max([AltHistData_SpreadsOnly[IndKey][l-1]+ConsecDiffsDic[IndKey][l-1]+(expon.rvs(scale=0.00015)*[-1,1][random.randrange(2)]),0.0000000000000001])
        AltLogReturnsDic[IndKey] = LogReturns(pd.Series(AltHistData_SpreadsOnly[IndKey]),lag=5,jump=5,averageJumpPeriod=True)
        AltDifferencesDic[IndKey] = AbsoluteDifferences(pd.Series(AltHistData_SpreadsOnly[IndKey]),jump=5,averageJumpPeriod=True)
        AltEmpFnForHistSpread[IndKey] = Empirical_StepWise_CDF(quickSort(AltDifferencesDic[IndKey].values))
        u_gpdthreashold = np.percentile(AltHistData_SpreadsOnly[IndKey],95)
        AltSemiParamametric[IndKey] = SemiParametricCDFFit(list(AltHistData_SpreadsOnly[IndKey]),u_gpdthreashold, True, "SemiParametricFit_%s"%(IndKey), "Alternative Historical Spreads", "Distribution")
        AltCanonicalMLETransformedHistDataDic[IndKey] = pd.Series(AltDifferencesDic[IndKey].values).apply(EmpFnForHistSpread[IndKey])
        AltSemiParamTransformedCDFHistDataDic[IndKey] =  pd.Series(AltSemiParamametric[IndKey]['%.10f'%(u_gpdthreashold)][0])
        AltSemiParamTransformedPDFHistDataDic[IndKey] =  pd.Series(AltSemiParamametric[IndKey]['%.10f'%(u_gpdthreashold)][1])
    
    
    plotter.return_lineChart_dates(HistCreditSpreads['Date'].values,[
        np.array(HistDataDic[HistCreditSpreads.columns[1]]['Spreads'])*BPS_TO_NUMBER,
        np.array(HistDataDic[HistCreditSpreads.columns[2]]['Spreads'])*BPS_TO_NUMBER, 
        np.array(HistDataDic[HistCreditSpreads.columns[3]]['Spreads'])*BPS_TO_NUMBER, 
        np.array(HistDataDic[HistCreditSpreads.columns[4]]['Spreads'])*BPS_TO_NUMBER, 
        np.array(HistDataDic[HistCreditSpreads.columns[5]]['Spreads'])*BPS_TO_NUMBER
        ],name="Historical Credit Spreads Data", xlabel="Historical Date", ylabel="Spread", legend=list(HistCreditSpreads.columns[1:]))

    plotter.return_lineChart_dates(HistCreditSpreads['Date'].values,[
        np.array(AltHistData_SpreadsOnly[HistCreditSpreads.columns[1]])*BPS_TO_NUMBER,
        np.array(AltHistData_SpreadsOnly[HistCreditSpreads.columns[2]])*BPS_TO_NUMBER, 
        np.array(AltHistData_SpreadsOnly[HistCreditSpreads.columns[3]])*BPS_TO_NUMBER, 
        np.array(AltHistData_SpreadsOnly[HistCreditSpreads.columns[4]])*BPS_TO_NUMBER, 
        np.array(AltHistData_SpreadsOnly[HistCreditSpreads.columns[5]])*BPS_TO_NUMBER
        ],name="Alternative Historical Credit Spreads Data", xlabel="Historical Date", ylabel="Spread", legend=list(HistCreditSpreads.columns[1:]))
    
    t4 = time.time()
    print("Took %.10f seconds to grab Historical Spreads and Transform the data by its Empirical CDF." % (t4 - t3))  
    #ResidualCorP = CorP(ResidualsLogReturnsDic) 
    plotter.plot_histogram_array(LogReturnsDic, "Weekly Log Returns")
    plotter.plot_histogram_array(DifferencesDic, "Weekly Differences")
    plotter.plot_histogram_array(CanonicalMLETransformedHistDataDic,"Inverse ECDF (Rank)")
    plotter.plot_histogram_array(SemiParamTransformedCDFHistDataDic,"Inverse Semi-Parametric CDF (Rank)")
    plotter.plot_histogram_array(SemiParamTransformedPDFHistDataDic,"Inverse Semi-Parametric PDF (Rank)")
    #!plotter.showAllPlots()


    Fac10AltHistData_SpreadsOnly = dict()
    Fac10AltEmpFnForHistSpread = dict()
    Fac10AltSemiParamametric = dict()
    Fac10AltLogReturnsDic = dict()
    Fac10AltDifferencesDic = dict()
    Fac10AltCanonicalMLETransformedHistDataDic = dict()
    Fac10AltSemiParamTransformedCDFHistDataDic = dict()
    Fac10AltSemiParamTransformedPDFHistDataDic = dict()
    Fac10AltEmpFnForHistSpread = dict()
    Fac10AltSemiParamametric = dict()
    def times10(x):
        return x * 10
    for i in range(1,6):
        IndKey = HistCreditSpreads.columns[i]
        Fac10AltHistData_SpreadsOnly[IndKey] = HistDataDic[IndKey]['Spreads'].apply(times10)
    
        Fac10AltLogReturnsDic[IndKey] = LogReturns(pd.Series(Fac10AltHistData_SpreadsOnly[IndKey]),lag=5,jump=5,averageJumpPeriod=True)
        Fac10AltDifferencesDic[IndKey] = AbsoluteDifferences(pd.Series(Fac10AltHistData_SpreadsOnly[IndKey]),jump=5,averageJumpPeriod=True)
        Fac10AltEmpFnForHistSpread[IndKey] = Empirical_StepWise_CDF(quickSort(Fac10AltDifferencesDic[IndKey].values))
        u_gpdthreashold = np.percentile(Fac10AltHistData_SpreadsOnly[IndKey],95)
        Fac10AltSemiParamametric[IndKey] = SemiParametricCDFFit(list(Fac10AltHistData_SpreadsOnly[IndKey]),u_gpdthreashold, True, "SemiParametricFit_%s"%(IndKey), "Fac 10 Historical Spreads", "Distribution")
        Fac10AltCanonicalMLETransformedHistDataDic[IndKey] = pd.Series(Fac10AltDifferencesDic[IndKey].values).apply(EmpFnForHistSpread[IndKey])
        Fac10AltSemiParamTransformedCDFHistDataDic[IndKey] =  pd.Series(Fac10AltSemiParamametric[IndKey]['%.10f'%(u_gpdthreashold)][0])
        Fac10AltSemiParamTransformedPDFHistDataDic[IndKey] =  pd.Series(Fac10AltSemiParamametric[IndKey]['%.10f'%(u_gpdthreashold)][1])

    t4a = time.time()
    print("Took %.10f seconds to print Transformed Spreads Histograms" % (t4a - t4))


    LogRtnCorP = CorP(LogReturnsDic)    #near correlation without kernel smoothing for gaussian copula 
    pdCorLogRtnP = convertToLaTeX(plotter,pd.DataFrame(LogRtnCorP, index = ReferenceNameList, columns=ReferenceNameList, dtype=np.float),name="Pearson Correlation",centerTable=False)
    #Transform HistCreditSpreads by its own empirical distn and then calc corrln on U to get Rank Corr. This is the defn of Spearmans Rho  
    #!Consider also using KENDALLS TAU AS AN ALTERNATIVE FORMULATION FOR THE RANK CORRELATION
    RankCorP = CorP(CanonicalMLETransformedHistDataDic)
    pdCorRankP = convertToLaTeX(plotter,pd.DataFrame(RankCorP, index = ReferenceNameList, columns=ReferenceNameList, dtype=np.float),name="Rank Correlation",centerTable=False)
    diffCor = RankCorP - LogRtnCorP
    pdCorDiffs = convertToLaTeX(plotter,pd.DataFrame(diffCor, index = ReferenceNameList, columns=ReferenceNameList, dtype=np.float),name="Rank Pearson Correlation Diffs",centerTable=False)

    #!SemiParametric Correlation Matrices
    SemiParametricRankCorP = CorP(SemiParamTransformedCDFHistDataDic)
    pdSemiParametricCorRankP = convertToLaTeX(plotter,pd.DataFrame(SemiParametricRankCorP, index = ReferenceNameList, columns=ReferenceNameList, dtype=np.float),name="Semi Parametric Correlation",centerTable=False)
    diffCorSemi = SemiParametricRankCorP - LogRtnCorP
    pdSemiCorDiffs = convertToLaTeX(plotter,pd.DataFrame(diffCorSemi, index = ReferenceNameList, columns=ReferenceNameList, dtype=np.float),name="Semi Parametric Correlation Diffs",centerTable=False)
    pdRevissedCorDiffs = convertToLaTeX(plotter,pd.DataFrame(SemiParametricRankCorP - RankCorP, index = ReferenceNameList, columns=ReferenceNameList, dtype=np.float),name="Semi Parametric Correlation Diffs From Revision",centerTable=False)

    #UseSemi:
    RankCorP = SemiParametricRankCorP

    #Calculate alternative Hist Data Correlations
    AltLogRtnCorP = CorP(AltLogReturnsDic)    #near correlation without kernel smoothing for gaussian copula 
    AltpdCorLogRtnP = convertToLaTeX(plotter,pd.DataFrame(AltLogRtnCorP, index = ReferenceNameList, columns=ReferenceNameList, dtype=np.float),
                                     name="Pearson Correlation from Alternative Historical Data",centerTable=False)
    AltSemiParametricRankCorP = CorP(AltSemiParamTransformedCDFHistDataDic)
    AltpdSemiParametricCorRankP = convertToLaTeX(plotter,pd.DataFrame(AltSemiParametricRankCorP, index = ReferenceNameList, columns=ReferenceNameList,dtype=np.float),
                                                 name="Semi Parametric Correlation from Alternative Historical Data",centerTable=False)
    AltRankCorP = AltSemiParametricRankCorP

    #Calculate Fac 10 alternative Hist Data Correlations
    Fac10AltLogRtnCorP = CorP(AltLogReturnsDic)    #near correlation without kernel smoothing for gaussian copula 
    Fac10AltpdCorLogRtnP = convertToLaTeX(plotter,pd.DataFrame(Fac10AltLogRtnCorP, index = ReferenceNameList, columns=ReferenceNameList, dtype=np.float),
                                     name="Pearson Correlation from 10 times Historical Data",centerTable=False)
    Fac10AltSemiParametricRankCorP = CorP(Fac10AltSemiParamTransformedCDFHistDataDic)
    Fac10AltpdSemiParametricCorRankP = convertToLaTeX(plotter,pd.DataFrame(Fac10AltSemiParametricRankCorP, index = ReferenceNameList, columns=ReferenceNameList,dtype=np.float),
                                                 name="Semi Parametric Correlation from 10 times Historical Data",centerTable=False)
    Fac10AltRankCorP = Fac10AltSemiParametricRankCorP

    t5 = time.time()
    print("Took %.10f seconds to calculate Correlation Matrices." % (t5 - t4a))

    U_hist = np.transpose(np.asarray([np.array(ar) for ar in list(SemiParamTransformedCDFHistDataDic.values())]))


    def multivariate_t_distribution(x,mu,Sigma,df,d):
        '''
        Multivariate t-student density:
        output:
            the density of the given element
        input:
            x = parameter (d dimensional numpy array or scalar)
            mu = mean (d dimensional numpy array or scalar)
            Sigma = scale matrix (dxd numpy array)
            df = degrees of freedom
            d: dimension
        '''
        Num = gamma(1. * (d+df)/2)
        Denom = ( gamma(1.*df/2) * pow(df*pi,1.*d/2) * pow(np.linalg.det(Sigma),1./2) * pow(1 + (1./df)*np.dot(np.dot((x - mu),np.linalg.inv(Sigma)), np.transpose(x - mu)),1.* (d+df)/2))
        d = 1. * Num / Denom 
        return d


    def T_Cop_DF_Likelihood(C,v,corMInv,detCorM):
        T = C.shape[0]
        D=C.shape[1]
        logLikelihood = (-T/2.0)*np.log(detCorM)  \
            - ((v+D)/2.0) * sum([ np.log(1+np.matmul(np.matmul(np.transpose(ct),corMInv),ct)/v) for ct in C])  \
            + T * (np.log(gamma((v+D)/2)) + (D-1)*np.log(gamma(v/2)) - D*np.log(gamma((v+1)/2))) \
            #+ ((v+1)/2.0) * sum([ sum([np.log(1+(cn**2)/v) for cn in ct]) for ct in C])

        dl_dv = -0.5*sum([ np.log(1+np.matmul(np.matmul(np.transpose(ct),corMInv),ct)/v) for ct in C])  \
            + ((v+D)/(2.0*(v**2.0))) * sum([ (np.matmul(np.matmul(np.transpose(ct),corMInv),ct))/(1+np.matmul(np.matmul(np.transpose(ct),corMInv),ct)/v) for ct in C])  \
            + 0.5 * sum([ sum([np.log(1+(cn**2)/v) for cn in ct]) for ct in C])   \
            - ((v+1.0)/(2.0*(v**2.0))) * sum([ sum([(cn**2)/(1+(cn**2)/v) for cn in ct]) for ct in C])
        return logLikelihood

    def T_Cop_MLE(U_hist,mu,v,corM):
        return sum([np.log(multivariate_t_distribution(U_hist,mu,corM,v,corM.shape[0]))])

    #-----------------https://github.com/stochasticresearch/copula-py/blob/master/copulapdf.py---------------------------------------------
    def _t(u, rho, nu):
        d = u.shape[1]
        nu = float(nu)
    
        try:
            R = np.linalg.cholesky(rho)
        except np.linalg.LinAlgError:
            raise ValueError('Provided Rho matrix is not Positive Definite!')
    
        ticdf = t.ppf(u, nu)
    
        z = np.linalg.solve(R,ticdf.T)
        z = z.T
        logSqrtDetRho = np.sum(np.log(np.diag(R)))
        const = gammaln((nu+d)/2.0) + (d-1)*gammaln(nu/2.0) - d*gammaln((nu+1)/2.0) - logSqrtDetRho
        sq = np.power(z,2)
        summer = np.sum(np.power(z,2),axis=1)
        numer = -((nu+d)/2.0) * np.log(1.0 + np.sum(np.power(z,2),axis=1)/nu)
        denom = np.sum(-((nu+1)/2) * np.log(1 + (np.power(ticdf,2))/nu), axis=1)
        y = np.exp(const + numer - denom)
    
        return y

    def _t2(x,E,v,mu=0):
        '''
        Multivariate t-student density:
        output:
            the density of the given element
        input:
            x = parameter (d dimensional numpy array or scalar)
            mu = mean (d dimensional numpy array or scalar)
            E = scale matrix (dxd numpy array)
            v = degrees of freedom
        '''
        n = E.shape[0]
        if not n == E.shape[1]:
            return IndexError("Scale matrix must be square")
        return pow(np.linalg.det(E),1./2) * \
            gamma((v+n)/2.) / gamma(v/2.) * \
            pow((gamma(v/2.)/gamma((v+1.)/2.)),n) * \
            pow((1+(np.matmul(np.matmul(np.transpose(t.ppf(x,n)),np.linalg.inv(E)),t.ppf(x,v)))/v),-0.5*(v+n)) / \
            np.prod([(pow(1+(pow(t.ppf(xi,v),2))/v,-0.5*(v+1))) for xi in x])

    def T_Cop_DF_MLE(U_hist, corM, maxDF):
        T = U_hist.shape[0]
        D=U_hist.shape[1]
        corInds = np.zeros(shape=(0))
        vE = np.zeros(shape=(maxDF+1))
        nVars = corM.shape[0]
        for i in range(0, corM.shape[0]):
            for j in range(i+1, corM.shape[1]):
                if np.abs(corM[i,j]) > 0.8:
                    if i not in corInds: corInds = np.append(corInds,i)
                    if j not in corInds: corInds = np.append(corInds,j)
                    if i not in corInds and j not in corInds:
                        nVars = nVars - 1
        v = np.max([1,np.min([maxDF, int((nVars *nVars - nVars)/2-1)])])
        vStart = v
        maxV = vStart
        vCounter = 0
        maxSumMLE = float("-inf")
        corMInv = np.linalg.inv(corM)
        detCorM = np.linalg.det(corM)
        while v > 1 and v < (maxDF+1):
            sumMLE = 0.0
            C = np.zeros(shape=U_hist.shape)
            for i in range(0,T):
                for j in range(0,D):
                    C[i,j] = t.ppf(U_hist[i,j],v)
            #t_cop_df_mle, t_cop_loc, t_cop_scale = t.fit(C)
            #print("For df: {0}, the t_cop_df_mle: {1} with loc: {2} and scale: {3}.".format(v,t_cop_df_mle,t_cop_loc, t_cop_scale))
            #for i in range(0,T):
            #    ct = TCopulaDensity(U_hist_t[i],v)
            #    print(ct)
            #    sumMLE += np.log(ct)
            #sumMLE = T_Cop_DF_Likelihood(C,v,corMInv,detCorM)

            #meanArr = np.asarray([0 for ut in np.transpose(U_hist)])
            #sumMLE = sum([np.log(T_Cop_MLE(un,meanArr,v,corM)) for un in U_hist])

            #sumMLE = sum([np.log(_t(np.array([un]),corM,v)) for un in U_hist])
            dum = np.log([_t2(un,corM,v) for un in U_hist])
            sumMLE = sum(dum)
            #!sumMLE = sum(np.log(_t(U_hist,corM,v)))
            #print("sumMLE: {0}, sumMLE2: {1} for df: {2}".format(sumMLE, sumMLE2, v))
            vE[v] = sumMLE 
            if sumMLE > maxSumMLE:
                maxSumMLE = sumMLE
                maxV = v
            else: vCounter += 1

            if v == maxDF:
                v = vStart -1
                vCounter = 0
            elif v > (vStart - 1):
                v = v+1
            else:
                v = v-1
        
            #if v < 23:
            #    if v > (vStart-1): v = v+1
            #    else: v = v-1
            #else:   
            #    v = vStart - 1
            #    vCounter = 0

            #v = ( (v+1) if v > (vStart-1) else (v-1) ) if v < 23 else (vStart - 1)
            if vCounter > 5:
                if v > (vStart-1): 
                    v = vStart - 1
                    vCounter = 0
                else: 
                    v = 0
        vE[0] = maxV
        return vE




    maxDF = 70
    vE = T_Cop_DF_MLE(U_hist, RankCorP, maxDF) #solve for v using newton-raphson? or try values in l for 3 to 25 starting from most likely v...
    #vE2 = T_Cop_DF_MLE(U_hist,  CorP(CanonicalMLETransformedHistDataDic), maxDF)
    #vE3 = T_Cop_DF_MLE(U_hist,  LogRtnCorP, maxDF)
    #vE = TCopula_DF_MLE(U_hist, RankCorP)
    plotter.return_lineChart(np.arange(1,maxDF+1),[vE[1:]],name="MLE procedure for T copula degrees of freedom",xlabel="Degrees of Freedom",ylabel="Log-likelihood",trimTrailingZeros=True)
    #plotter.return_lineChart(np.arange(1,maxDF+1),[vE2[1:]],name="MLE procedure for T copula degrees of freedom",xlabel="Degrees of Freedom",ylabel="Log-likelihood")
    #plotter.return_lineChart(np.arange(1,maxDF+1),[vE3[1:]],name="MLE procedure for T copula degrees of freedom",xlabel="Degrees of Freedom",ylabel="Log-likelihood")

    #C = np.transpose(np.asarray([np.array(ar) for ar in list(HistDataDic.values())]))
    #t_cop_df_mle, t_cop_loc, t_cop_scale = t.fit(C)
    #alt_t_fit = t.fit(U_hist)

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
    plotter.plot_codependence_scatters(dict(enumerate(DummyRUs.transpose())),"D%","D%")
    #DummyRUs = np.zeros(shape=(l,LogRtnCorP.shape[0]))
    #for j in range(0,l):
    #    DummyRUs[j] = np.array(NumbGen.Generate(),np.float)
    #plotter.plot_codependence_scatters(dict(enumerate(DummyRUs.transpose())),"D%","D%")
    #for i in range(0,5000):
    #    NumbGen.Generate()    
    t6 = time.time()
    print("Took %.10f seconds to init and run 5000x%d iterations of sobol numbers." % ((t6 - t5) , LogRtnCorP.shape[0]))



    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!MONTE CARLO SIMULATION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    CDSBasketMaturity = 5.0
    CDSPaymentTenor = 0.25
    CDSPaymentTenors = np.arange(CDSPaymentTenor,CDSBasketMaturity+CDSPaymentTenor,CDSPaymentTenor,dtype=np.float)
    M = BPS_TO_NUMBER

    #TLegs = SimulateCDSBasketDefaultsAndValueLegsT(t6,RankCorP,M,HistCreditSpreads,TransformedHistDataDic,TenorCreditSpreads,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity)
    #GaussLegs = SimulateCDSBasketDefaultsAndValueLegsGauss(TLegs[2],LogRtnCorP,M,HistCreditSpreads,TenorCreditSpreads,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity)

    #GaussFairSpread,t9 = CalculateFairSpreadFromLegs(GaussLegs[0],GaussLegs[1],M,GaussLegs[2],"Gauss")
    #TFairSpread,t10 = CalculateFairSpreadFromLegs(TLegs[0],TLegs[1],M,TLegs[2],"T")


    GaussFairSpread,TFairSpread,t10 = FullMCFairSpreadValuation(plotter,t6,LogRtnCorP,RankCorP,M,HistCreditSpreads,SemiParamTransformedCDFHistDataDic, vE[0],TenorCreditSpreads,InvPWCDF,
                                                                DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,name="Initial_Fair_Spread_Calculation")


    latexFairSpreads = convertToLaTeX(plotter,pd.DataFrame(data=np.array([GaussFairSpread,TFairSpread],dtype=np.float), 
                                                   index = ["Gaussian", "Student's T"], columns=["1st to default","2nd to default","3rd to default","4th to default","5th to default"], dtype=np.float),"Fair Spreads",centerTable=False)

    plotter.return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([GaussFairSpread,TFairSpread]),
                                    "Comparison of fair spreads",xlabel="K-th to default",ylabel="Fair spread", 
                                    legend=["Gaussian", "Students T"])

    #todo: Check if the reason for the zero spreads is the SONIA Discount curve


    #-----------------------------------------------------------temp-----------------------------------------------------------------------------------------------------

    #tNew = time.time()
    #CorAllTweakFairSpreadGaussList = list()
    #CorAllTweakFairSpreadTList = list()
    #def PercTweakCors(percTweak,tNew):
    #    TweakedRankCorP = TweakWhole2DMatrixByPercent(RankCorP,percTweak)
    #    TweakedLogRtnCorP = TweakWhole2DMatrixByPercent(LogRtnCorP,percTweak)
    #    print("Calculating fair spread after tweaking correlation for whole matrix by {0}%".format(percTweak*100))
    #    tPrev = tNew
    #    CorAllTweakFairSpreadGauss,CorAllTweakFairSpreadT,tNew = FullMCFairSpreadValuation(plotter,tPrev,TweakedLogRtnCorP,TweakedRankCorP,M,HistCreditSpreads,SemiParamTransformedCDFHistDataDic, vE[0],TenorCreditSpreads,InvPWCDF,
    #                                                                          DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,
    #                                                                          name="Correlation between all reference names tweaked by {0} percent".format(percTweak*100))
    #    CorAllTweakFairSpreadGaussList.append(CorAllTweakFairSpreadGauss)
    #    CorAllTweakFairSpreadTList.append(CorAllTweakFairSpreadT)

    #    plotter.return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([GaussFairSpread,CorAllTweakFairSpreadGauss]),
    #                                    "Correlation between all reference names tweaked by {0} percent (Gaussian)".format(percTweak*100),xlabel="K-th to default",ylabel="Fair spread", 
    #                                    legend=["Fair Spreads", "Tweaked Fair Spreads"])
    #    plotter.return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([TFairSpread,CorAllTweakFairSpreadT]),
    #                                    "Correlation between all reference names tweaked by {0} percent (Students T)".format(percTweak*100),xlabel="K-th to default",ylabel="Fair spread", 
    #                                    legend=["Fair Spreads", "Tweaked Fair Spreads"])
    #corPercTweaksArr = [-0.8,-0.5,-0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 0.8]
    #for ptw in corPercTweaksArr:
    #    PercTweakCors(ptw,tNew)

    #CorAllTweakFairSpreadGaussArr = np.array(CorAllTweakFairSpreadGaussList)
    #CorAllTweakFairSpreadTArr = np.array(CorAllTweakFairSpreadTList)

    #CorAllTweakFairSpreadGaussLines = np.transpose(CorAllTweakFairSpreadGaussArr)
    #CorAllTweakFairSpreadTLines = np.transpose(CorAllTweakFairSpreadTArr)

    #plotter.return_lineChart(corPercTweaksArr,CorAllTweakFairSpreadGaussLines,name="Sensitivity of fair spreads to percentage changes to correlation (Gaussian)",
    #                 xlabel="Percentage change to correlation",ylabel="Fair spread",legend=["1st to default","2nd to default","3rd to default","4th to default","5th to default"])
    #plotter.return_lineChart(corPercTweaksArr,CorAllTweakFairSpreadTLines,name="Sensitivity of fair spreads to percentage changes to correlation (Students T)",
    #                 xlabel="Percentage change to correlation",ylabel="Fair spread",legend=["1st to default","2nd to default","3rd to default","4th to default","5th to default"])


    #def TweakCDSSpreads(TweakIndKey,TweakAmountInBps, UseConstantBump=True):
    #    TweakedDataTenorDic = dict()
    #    TweakedImpProbDic = dict()
    #    TweakedImpHazdRts = dict()
    #    TweakedInvPWCDF = dict()
    #    TweakedPWCDF = dict()
    
    #    for i in range(0,5*5,5):
    #        IndKey = TenorCreditSpreads['Ticker'][i]
    #        TweakedDataTenorDic[IndKey] = np.array(TenorCreditSpreads['DataSR'][i:(i+5)] / BPS_TO_NUMBER)
    #        if IndKey == TweakIndKey or IndKey == "All":
    #            TweakedDataTenorDic[TweakIndKey][0] += (TweakAmountInBps if UseConstantBump else expon.rvs(scale=TweakAmountInBps))
    #            for l in range(1,len(TweakedDataTenorDic[IndKey])):
    #                TweakedDataTenorDic[IndKey][l] += (TweakAmountInBps if UseConstantBump else np.min([expon.rvs(scale=TweakAmountInBps),(TweakedDataTenorDic[IndKey][l-1]-TweakedDataTenorDic[IndKey][l])]))
    #        TweakedDataTenorDic[IndKey] = list(TweakedDataTenorDic[IndKey])
    #        TweakedImpProbDic[IndKey] = BootstrapImpliedProbalities(0.4,TweakedDataTenorDic[IndKey],DiscountFactorCurve["Sonia"])
    #        Tenors = TweakedImpProbDic[IndKey].index
    #        BootstrappedSurvProbs = TweakedImpProbDic[IndKey]['ImpliedPrSurv']
    #        TweakedImpHazdRts[IndKey] = GetHazardsFromP(BootstrappedSurvProbs,Tenors)
    #        TweakedInvPWCDF[IndKey], TweakedPWCDF[IndKey] = ApproxPWCDFDicFromHazardRates(TweakedImpHazdRts[IndKey],0.01)
    #    return TweakedDataTenorDic, TweakedImpProbDic, TweakedImpHazdRts, TweakedInvPWCDF

    #GaussFairSpreadTweakCDS = np.zeros(shape=(5),dtype=np.float)
    #TFairSpreadTweakCDS = np.zeros(shape=(5),dtype=np.float)
    #M=201
    #CreditTweaksToCarryOutBps = np.array([-10, -5, 5, 10, 30, 50, 75, 100, 150, 250, 500]) / BPS_TO_NUMBER

    #GaussFairSpreadTweakCDS = np.zeros(shape=(5,5),dtype=np.float)
    #TFairSpreadTweakCDS = np.zeros(shape=(5,5),dtype=np.float)
    #DeltaGaussFairSpreadTweakCDS = dict()
    #DeltaTFairSpreadTweakCDS = dict()
    #for CreditTenorTweakAmount in CreditTweaksToCarryOutBps:
    #    for i in range(0,5*5,5):
    #        IndKey = TenorCreditSpreads['Ticker'][i]
    #        if not IndKey in DeltaGaussFairSpreadTweakCDS:
    #            DeltaGaussFairSpreadTweakCDS[IndKey] = dict()
    #            DeltaTFairSpreadTweakCDS[IndKey] = dict()
    #        TweakedDataTenorDic, TweakedImpProbDic, TweakedImpHazdRts, TweakedInvPWCDF = TweakCDSSpreads(IndKey,CreditTenorTweakAmount)
    #        print("Tweaking the credit spreads for {0} and rerunning analysis for tweak: {1}".format(IndKey,CreditTenorTweakAmount))
    #        GaussFairSpreadTweakCDS[int(i/5)], TFairSpreadTweakCDS[int(i/5)], t17 = FullMCFairSpreadValuation(plotter,time.time(),LogRtnCorP,RankCorP,M,HistCreditSpreads,SemiParamTransformedCDFHistDataDic, vE[0],TenorCreditSpreads,TweakedInvPWCDF,
    #                                                                          DiscountFactorCurve,TweakedImpHazdRts,TweakedDataTenorDic,CDSPaymentTenors,CDSBasketMaturity,name="Tweak Spread for {0} by {1}".format(IndKey,CreditTenorTweakAmount))
    #        DeltaGaussFairSpreadTweakCDS[IndKey]["{0}".format(CreditTenorTweakAmount)] = GaussFairSpreadTweakCDS[int(i/5)] - GaussFairSpread
    #        DeltaTFairSpreadTweakCDS[IndKey]["{0}".format(CreditTenorTweakAmount)] = TFairSpreadTweakCDS[int(i/5)] - TFairSpread


    #CDSRefNamesArr = np.array(TenorCreditSpreads['Ticker'][0:25:5])

    #for RefName in CDSRefNamesArr:
    #    deltaVBasket = np.transpose(list(DeltaGaussFairSpreadTweakCDS[RefName].values()))
    #    dBasket = deltaVBasket/CreditTweaksToCarryOutBps
    #    plotter.return_lineChart(CreditTweaksToCarryOutBps,deltaVBasket,
    #                     name="Plot of Credit Deltas for {0} under Gauss Assumption.".format(RefName),xlabel="Credit Delta", ylabel="Basket Spread Delta",
    #                     legend=["1st to default","2nd to default","3rd to default","4th to default","5th to default"])
    #    plotter.return_lineChart(CreditTweaksToCarryOutBps,dBasket,
    #                     name="Plot of dBasketSpread to dCreditSpread for {0} under Gauss Assumption.".format(RefName),xlabel="Credit Delta", ylabel="dV/ds",
    #                     legend=["1st to default","2nd to default","3rd to default","4th to default","5th to default"])
    #    fitY = dict()
    #    xLinY = dict()
    #    fitD = dict()
    #    xLinD = dict()
    #    for k in range(0,len(deltaVBasket)):
    #        fitYfn, rPowY, xLinY["{0}".format(k)] = plotter.SuitableRegressionFit(CreditTweaksToCarryOutBps,deltaVBasket[k],
    #                         name="Plot of Interpolated Credit Deltas for {0} under Gauss Assumption.".format(RefName))

    #        fitDfn, rPowD, xLinD["{0}".format(k)] = plotter.SuitableRegressionFit(CreditTweaksToCarryOutBps,dBasket[k],
    #                            name="Plot of Interpolated dBasketSpread to dCreditSpread for {0} under Gauss Assumption.".format(RefName))

    #        print(xLinY["{0}".format(0)])
    #        fitY["{0}".format(k)] = np.fromiter(map(lambda x: fitYfn(x), xLinY["{0}".format(0)]),dtype=np.float)
    #        fitD["{0}".format(k)] = np.fromiter(map(lambda x: fitDfn(x), xLinD["{0}".format(0)]),dtype=np.float)
    #        print(fitY["{0}".format(k)])
        
    #    plotter.return_lineChart(xLinY["{0}".format(0)],list(fitY.values()),
    #                    name="Plot of Interpolated_{1} Credit Deltas for {0} under Gauss Assumption.".format(RefName,rPowY),xlabel="Credit Delta", ylabel="Basket Spread Delta",
    #                    legend=["1st to default","2nd to default","3rd to default","4th to default","5th to default"])
    #    plotter.return_lineChart(xLinD["{0}".format(0)],list(fitD.values()),
    #                    name="Plot of Interpolated_{1} dBasketSpread to dCreditSpread for {0} under Gauss Assumption.".format(RefName,rPowD),xlabel="Credit Delta", ylabel="dV/ds",
    #                    legend=["1st to default","2nd to default","3rd to default","4th to default","5th to default"])

    
        
    #    print("Calculating Credit Deltas")
    #    deltaVBasket = np.transpose(list(DeltaTFairSpreadTweakCDS[RefName].values()))
    #    dBasket = deltaVBasket/CreditTweaksToCarryOutBps
    #    plotter.return_lineChart(CreditTweaksToCarryOutBps,deltaVBasket,
    #                     name="Plot of Credit Deltas for {0} under Gauss Assumption.".format(RefName),xlabel="Credit Delta", ylabel="Basket Spread Delta",
    #                     legend=["1st to default","2nd to default","3rd to default","4th to default","5th to default"])
    #    plotter.return_lineChart(CreditTweaksToCarryOutBps,dBasket,
    #                     name="Plot of dBasketSpread to dCreditSpread for {0} under Gauss Assumption.".format(RefName),xlabel="Credit Delta", ylabel="dV/ds",
    #                     legend=["1st to default","2nd to default","3rd to default","4th to default","5th to default"])
    #    fitY = dict()
    #    xLinY = dict()
    #    fitD = dict()
    #    xLinD = dict()
    #    for k in range(0,len(deltaVBasket)):
    #        fitYfn, rPowY, xLinY["{0}".format(k)] = plotter.SuitableRegressionFit(CreditTweaksToCarryOutBps,deltaVBasket[k],
    #                         name="Plot of Interpolated Credit Deltas for {0} under Gauss Assumption.".format(RefName))

    #        fitDfn, rPowD, xLinD["{0}".format(k)] = plotter.SuitableRegressionFit(CreditTweaksToCarryOutBps,dBasket[k],
    #                            name="Plot of Interpolated dBasketSpread to dCreditSpread for {0} under Gauss Assumption.".format(RefName))

        
    #        fitY["{0}".format(k)] = np.fromiter(map(lambda x: fitYfn(x), xLinY["{0}".format(0)]),dtype=np.float)
    #        fitD["{0}".format(k)] = np.fromiter(map(lambda x: fitDfn(x), xLinD["{0}".format(0)]),dtype=np.float)

    #    plotter.return_lineChart(xLinY["{0}".format(0)],list(fitY.values()),
    #                    name="Plot of Interpolated_{1} Credit Deltas for {0} under Gauss Assumption.".format(RefName,rPowY),xlabel="Credit Delta", ylabel="Basket Spread Delta",
    #                    legend=["1st to default","2nd to default","3rd to default","4th to default","5th to default"])
    #    plotter.return_lineChart(xLinD["{0}".format(0)],list(fitD.values()),
    #                    name="Plot of Interpolated_{1} dBasketSpread to dCreditSpread for {0} under Gauss Assumption.".format(RefName,rPowD),xlabel="Credit Delta", ylabel="dV/ds",
    #                    legend=["1st to default","2nd to default","3rd to default","4th to default","5th to default"])
    #M=BPS_TO_NUMBER
    #----------------------------------------------------------------------temp----------------------------------------------------------------------------------------------
results_bank = []
def worker(working_queue, output_queue, fn_dic, plotter,M,HistCreditSpreads,SemiParamTransformedCDFHistDataDic, vE,
                          TenorCreditSpreads,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,
                          CDSPaymentTenors,CDSBasketMaturity, 
                          GaussFairSpread,TFairSpread,RankCorP,LogRtnCorP,ReferenceNameList,
                          TweakCDSSpreads,BPS_TO_NUMBER,
                          Fac10AltLogRtnCorP,Fac10AltRankCorP,Fac10AltSemiParamTransformedCDFHistDataDic,
                          AltLogRtnCorP,AltRankCorP,AltSemiParamTransformedCDFHistDataDic):
    while True:
        if working_queue.empty() == True:
            break #this is the so-called 'poison pill'    
        else:
            picked_fn_name = working_queue.get()
            print("Performing work for: {0}".format(picked_fn_name))
            tstart = time.time()
            fn_dic[picked_fn_name](plotter=plotter,M=M,HistCreditSpreads=HistCreditSpreads,SemiParamTransformedCDFHistDataDic=SemiParamTransformedCDFHistDataDic, vE=vE,
                          TenorCreditSpreads=TenorCreditSpreads,InvPWCDF=InvPWCDF,DiscountFactorCurve=DiscountFactorCurve,ImpHazdRts=ImpHazdRts,DataTenorDic=DataTenorDic,
                          CDSPaymentTenors=CDSPaymentTenors,CDSBasketMaturity=CDSBasketMaturity, 
                          GaussFairSpread=GaussFairSpread,TFairSpread=TFairSpread,RankCorP=RankCorP,LogRtnCorP=LogRtnCorP,ReferenceNameList=ReferenceNameList,
                          TweakCDSSpreads=TweakCDSSpreads,BPS_TO_NUMBER=BPS_TO_NUMBER,
                          Fac10AltLogRtnCorP=Fac10AltLogRtnCorP,Fac10AltRankCorP=Fac10AltRankCorP,Fac10AltSemiParamTransformedCDFHistDataDic=Fac10AltSemiParamTransformedCDFHistDataDic,
                          AltLogRtnCorP=AltLogRtnCorP,AltRankCorP=AltRankCorP,AltSemiParamTransformedCDFHistDataDic=AltSemiParamTransformedCDFHistDataDic)
            print("Finished work for: {0} and took {1} mins".format(picked_fn_name,(time.time()-tstart)/60))
            output_queue.put(picked_fn_name)
    return

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!CHECK OF FAIR SPREADS BY MONTE CARLO SIMULATION USING ESTIMATED FAIR SPREAD!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def pCheckFairSpead( plotter,M,HistCreditSpreads,SemiParamTransformedCDFHistDataDic, vE,
                          TenorCreditSpreads,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,
                          CDSPaymentTenors,CDSBasketMaturity, 
                          GaussFairSpread,TFairSpread,RankCorP,LogRtnCorP,ReferenceNameList,
                          TweakCDSSpreads,BPS_TO_NUMBER,
                          Fac10AltLogRtnCorP,Fac10AltRankCorP,Fac10AltSemiParamTransformedCDFHistDataDic,
                          AltLogRtnCorP,AltRankCorP,AltSemiParamTransformedCDFHistDataDic):
    TLegsWFairSpread = SimulateCDSBasketDefaultsAndValueLegsT(plotter,time.time(),RankCorP,M,HistCreditSpreads,SemiParamTransformedCDFHistDataDic, vE[0],TenorCreditSpreads,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,TFairSpread,name="Check_plausability_of_fair_spread")
    GaussLegsWFairSpread = SimulateCDSBasketDefaultsAndValueLegsGauss(plotter,TLegsWFairSpread[2],LogRtnCorP,M,HistCreditSpreads,TenorCreditSpreads,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,GaussFairSpread,name="Check_plausability_of_fair_spread")
    t11 = GaussLegsWFairSpread[2]
    #GaussLegsWFairSpread,TLegsWFairSpread,t11 = FullMCFairSpreadValuation(plotter,t10,LogRtnCorP,RankCorP,M,HistCreditSpreads,TransformedHistDataDic,TenorCreditSpreads,InvPWCDF,
    #                                                                      DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,TFairSpread,GaussFairSpread)
    TFairDiff = TLegsWFairSpread[0] - TLegsWFairSpread[1] 
    GaussFairDiff = GaussLegsWFairSpread[0] - GaussLegsWFairSpread[1]

    AverageGaussFairDiff = np.array([np.sum(GaussFairDiffD) for GaussFairDiffD in np.transpose(GaussFairDiff)])/GaussFairDiff.shape[0]
    AverageTFairDiff = np.array([np.sum(TFairDiffD) for TFairDiffD in np.transpose(TFairDiff)])/TFairDiff.shape[0]

    convertToLaTeX(plotter,pd.DataFrame(data=np.array([AverageGaussFairDiff,AverageTFairDiff],dtype=np.float), 
                                                    index = ["Gaussian", "Student's T"], 
                                                    columns=["1st to default","2nd to default","3rd to default","4th to default","5th to default"], 
                                                    dtype=np.float),"Averarge CDS leg discrepancy from resimulation of basket using fair spread",centerTable=False)

    gaussCheckFairSpreadDic = dict()
    tCheckFairSpreadDic = dict()
    for iten in range(0,TFairDiff.shape[1]):
        gaussCheckFairSpreadDic["%d-th to default basket CDS" % (iten+1)] = GaussFairDiff[:,iten]
        tCheckFairSpreadDic["%d-th to default basket CDS" % (iten+1)] = TFairDiff[:,iten]
    plotter.plot_histogram_array(gaussCheckFairSpreadDic,"CDS Basket Spreads",name="CDS Basket Outcome Using Gaussian-Copula to check calculated fair spreads")
    plotter.plot_histogram_array(tCheckFairSpreadDic, "CDS Basket Spreads", name="CDS Basket Outcome Using T-Copula to check calculated fair spreads")
#For Kendalls Tau, we have X1 and X2 from the data with empirical cdf, we then also simulate X3 and X4 from the emp distributions of X1 and x2 resp. We then defn pTau := E[sign((x1-x3)*(x2-x4))] 
#Now Consider altering the tail dependence of the copulas upper and lower separately.

#Step 1: Calculate the empirical cdf for each hist Ref name, both using kernel smoothing(pdf -> cdf) and just empirical stewpise constant cdf.
#Step 2: Transform historical spreads, X, by their CDF F(X) = U
#Step 3: Obtain Normally, T distributed R.v.s by applying the respective inverse CDF to the U.
#Step 4: Calculate Correlations from Standardised Credit Spread returns / Default Probs and use these near and rank correlation matrices to simulate Normal or T distributed variables respectively.
#Step 5: Compare the difference between 1-3 and 4.
#plotter.showAllPlots()



#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  RISK AND SENSITIVITY ANALYSIS   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#!Sensitiivity of basket to interest rates------------------------------------------------------------------------------------------
def IRTweak( plotter,M,HistCreditSpreads,SemiParamTransformedCDFHistDataDic, vE,
                          TenorCreditSpreads,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,
                          CDSPaymentTenors,CDSBasketMaturity, 
                          GaussFairSpread,TFairSpread,RankCorP,LogRtnCorP,ReferenceNameList,
                          TweakCDSSpreads,BPS_TO_NUMBER,
                          Fac10AltLogRtnCorP,Fac10AltRankCorP,Fac10AltSemiParamTransformedCDFHistDataDic,
                          AltLogRtnCorP,AltRankCorP,AltSemiParamTransformedCDFHistDataDic):
    DFTweakFairSpreadGaussList = list()
    DFTweakFairSpreadTList = list()
    DFTweakFairSpreadGaussDiffP = list()
    DFTweakFairSpreadTDiffP = list()
    NonZeroGFAirSpread = np.copy(GaussFairSpread)
    NonZeroGFAirSpread[NonZeroGFAirSpread <= 0] = 0.0000001
    NonZeroTFAirSpread = np.copy(TFairSpread)
    NonZeroTFAirSpread[NonZeroTFAirSpread <= 0] = 0.0000001
    def Tweak_DF(twk):
        TweakedDiscountFactorCurveDic = dict()
        for key in DiscountFactorCurve.keys():
            def f(t):
                return DiscountFactorCurve[key](t) + twk
            TweakedDiscountFactorCurveDic[key] = f
        print("Calculating fair spread after tweaking Discount rates by {0} bps".format(twk*BPS_TO_NUMBER))
        DFTweakFairSpreadGauss,DFTweakFairSpreadT,tdummy = FullMCFairSpreadValuation(plotter,time.time(),LogRtnCorP,RankCorP,M,HistCreditSpreads,SemiParamTransformedCDFHistDataDic, vE[0],
                                                                                    TenorCreditSpreads,InvPWCDF,TweakedDiscountFactorCurveDic,ImpHazdRts,DataTenorDic,CDSPaymentTenors,
                                                                                    CDSBasketMaturity,name="Discount rates tweaked by {0} bps".format(twk*BPS_TO_NUMBER))
        DFTweakFairSpreadGaussList.append(DFTweakFairSpreadGauss)
        DFTweakFairSpreadTList.append(DFTweakFairSpreadT)
        DFTweakFairSpreadGaussDiffP.append(((DFTweakFairSpreadGauss-GaussFairSpread)/NonZeroGFAirSpread)*100)
        DFTweakFairSpreadTDiffP.append(((DFTweakFairSpreadT-TFairSpread)/NonZeroTFAirSpread)*100)

    irTweaks = np.arange(-500,600,100)/BPS_TO_NUMBER
    for rate_twk in irTweaks:
        Tweak_DF(rate_twk)
    
    DFTweakFairSpreadGaussArr = np.array(DFTweakFairSpreadGaussList)
    DFTweakFairSpreadTArr = np.array(DFTweakFairSpreadTList)

    DFTweakFairSpreadGaussLines = np.transpose(DFTweakFairSpreadGaussArr)
    DFTweakFairSpreadTLines = np.transpose(DFTweakFairSpreadTArr)

    DFTweakFairSpreadGaussArrDiff = np.array(DFTweakFairSpreadGaussDiffP)
    DFTweakFairSpreadTArrDiff = np.array(DFTweakFairSpreadTDiffP)

    DFTweakFairSpreadGaussLinesDiff = np.transpose(DFTweakFairSpreadGaussArrDiff)
    DFTweakFairSpreadTLinesDiff = np.transpose(DFTweakFairSpreadTArrDiff)

    plotter.return_lineChart(irTweaks,DFTweakFairSpreadGaussLines,name="Sensitivity of fair spreads to percentage changes in SONIA rate (Gaussian)",
                        xlabel="Change in SONIA rate",ylabel="Fair spread",legend=["1st to default","2nd to default","3rd to default","4th to default","5th to default"])
    plotter.return_lineChart(irTweaks,DFTweakFairSpreadGaussLinesDiff,name="Sensitivity of percentage changes in fair spreads to changes in SONIA rate (Gaussian)",
                        xlabel="Change in SONIA rate",ylabel="% change in fair spread",legend=["1st to default","2nd to default","3rd to default","4th to default","5th to default"])

    plotter.return_lineChart(irTweaks,DFTweakFairSpreadTLines,name="Sensitivity of fair spreads to percentage changes in SONIA rate (Students T)",
                        xlabel="Change in SONIA rate",ylabel="Fair spread",legend=["1st to default","2nd to default","3rd to default","4th to default","5th to default"])
    plotter.return_lineChart(irTweaks,DFTweakFairSpreadTLinesDiff,name="Sensitivity of percentage changes in fair spreads to percentage changes in SONIA rate (Students T)",
                        xlabel="Change in SONIA rate",ylabel="% change in fair spread",legend=["1st to default","2nd to default","3rd to default","4th to default","5th to default"])    


#!Sensitivity of basket to Recovery Rates------------------------------------------------------------------------------------------
def RTweak( plotter,M,HistCreditSpreads,SemiParamTransformedCDFHistDataDic, vE,
                          TenorCreditSpreads,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,
                          CDSPaymentTenors,CDSBasketMaturity, 
                          GaussFairSpread,TFairSpread,RankCorP,LogRtnCorP,ReferenceNameList,
                          TweakCDSSpreads,BPS_TO_NUMBER,
                          Fac10AltLogRtnCorP,Fac10AltRankCorP,Fac10AltSemiParamTransformedCDFHistDataDic,
                          AltLogRtnCorP,AltRankCorP,AltSemiParamTransformedCDFHistDataDic):
    itera = 0
    Rrng = np.arange(0.0,1.0,0.1)
    GaussFairSpreadTweakR = np.zeros(shape=(len(Rrng),5),dtype=np.float)
    TFairSpreadTweakR = np.zeros(shape=(len(Rrng),5),dtype=np.float)
    tNew = time.time()
    for RAlt in Rrng:
        print("Tweaking recovery rates to rate: {0} and rerunning analysis".format(RAlt))
        GaussFairSpreadTweakR[itera], TFairSpreadTweakR[itera], tNew = FullMCFairSpreadValuation(plotter,time.time(),LogRtnCorP,RankCorP,M,HistCreditSpreads,SemiParamTransformedCDFHistDataDic, vE[0],TenorCreditSpreads,InvPWCDF,
                                                                            DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,RAlt,name="Alternative recovery rate: ({0})".format(RAlt))
        itera += 1
    t18 = tNew
    plotter.return_scatter_multdependencies(Rrng,GaussFairSpreadTweakR.transpose(),"Sensitivity of FairSpread to changing Recovery Rate (Gauss)", 
                        xlabel="Recovery Rate",ylabel="FairSpread",legend=["1st to default","2nd to default","3rd to default","4th to default","5th to default"])
    plotter.return_scatter_multdependencies(Rrng,TFairSpreadTweakR.transpose(),"Sensitivity of FairSpread to changing Recovery Rate (T)", 
                        xlabel="Recovery Rate",ylabel="FairSpread",legend=["1st to default","2nd to default","3rd to default","4th to default","5th to default"])
    
#!Fair spread with Alternative Historical Data------------------------------------------------------------------------------------------
def AltHistDataTweak( plotter,M,HistCreditSpreads,SemiParamTransformedCDFHistDataDic, vE,
                          TenorCreditSpreads,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,
                          CDSPaymentTenors,CDSBasketMaturity, 
                          GaussFairSpread,TFairSpread,RankCorP,LogRtnCorP,ReferenceNameList,
                          TweakCDSSpreads,BPS_TO_NUMBER,
                          Fac10AltLogRtnCorP,Fac10AltRankCorP,Fac10AltSemiParamTransformedCDFHistDataDic,
                          AltLogRtnCorP,AltRankCorP,AltSemiParamTransformedCDFHistDataDic):
    GaussFairSpreadTweakHistData, TFairSpreadTweakHistData, t16 = FullMCFairSpreadValuation(plotter,time.time(),AltLogRtnCorP,AltRankCorP,M,HistCreditSpreads,
                                                                                                        AltSemiParamTransformedCDFHistDataDic, vE[0],TenorCreditSpreads,InvPWCDF,DiscountFactorCurve,
                                                                                                        ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,
                                                                                                        name="Alternative Historical Data")
    DeltaGaussFairSpreadTweakHistData = GaussFairSpreadTweakHistData - GaussFairSpread
    DeltaTFairSpreadTweakHistData = TFairSpreadTweakHistData - TFairSpread

    latexAltHistFairSpreads = convertToLaTeX(plotter,pd.DataFrame(data=np.array([GaussFairSpreadTweakHistData,TFairSpreadTweakHistData],dtype=np.float), 
                                                    index = ["Gaussian", "Student's T"], columns=["1st to default","2nd to default","3rd to default","4th to default","5th to default"], dtype=np.float),
                                            "Fair Spreads from alternative term structure of spreads",centerTable=False)


    
def TweakCDSSpreads(TweakIndKey,TweakAmountInBps,TenorCreditSpreads,BPS_TO_NUMBER,DiscountFactorCurve, UseConstantBump=True, UseProduct=False):
    TweakedDataTenorDic = dict()
    TweakedImpProbDic = dict()
    TweakedImpHazdRts = dict()
    TweakedInvPWCDF = dict()
    TweakedPWCDF = dict()
    
    for i in range(0,5*5,5):
        IndKey = TenorCreditSpreads['Ticker'][i]
        TweakedDataTenorDic[IndKey] = np.array(TenorCreditSpreads['DataSR'][i:(i+5)] / BPS_TO_NUMBER)
    for i in range(0,5*5,5):
        IndKey = TenorCreditSpreads['Ticker'][i]
        if IndKey == TweakIndKey or TweakIndKey == "All":
            twk = (TweakAmountInBps if UseConstantBump else expon.rvs(scale=TweakAmountInBps))
            if UseProduct:
                TweakedDataTenorDic[IndKey][0] *= twk
            else:
                TweakedDataTenorDic[IndKey][0] += twk
            for l in range(1,len(TweakedDataTenorDic[IndKey])):
                TweakedDataTenorDic[IndKey][l] += (TweakAmountInBps if UseConstantBump else np.min([expon.rvs(scale=TweakAmountInBps),(TweakedDataTenorDic[IndKey][l-1]-TweakedDataTenorDic[IndKey][l])]))
        TweakedDataTenorDic[IndKey] = list(TweakedDataTenorDic[IndKey])
        TweakedImpProbDic[IndKey] = BootstrapImpliedProbalities(0.4,TweakedDataTenorDic[IndKey],DiscountFactorCurve["Sonia"])
        Tenors = TweakedImpProbDic[IndKey].index
        BootstrappedSurvProbs = TweakedImpProbDic[IndKey]['ImpliedPrSurv']
        TweakedImpHazdRts[IndKey] = GetHazardsFromP(BootstrappedSurvProbs,Tenors)
        TweakedInvPWCDF[IndKey], TweakedPWCDF[IndKey] = ApproxPWCDFDicFromHazardRates(TweakedImpHazdRts[IndKey],0.01)
    return TweakedDataTenorDic, TweakedImpProbDic, TweakedImpHazdRts, TweakedInvPWCDF
        


#!Fair spread after tweaking Credit Spreads and Hist Credit Spreads by a factor of 10------------------------------------------------------------------------------------------
def AllFac10CreditAndHist(plotter, M,HistCreditSpreads,SemiParamTransformedCDFHistDataDic, vE,
                          TenorCreditSpreads,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,
                          CDSPaymentTenors,CDSBasketMaturity, 
                          GaussFairSpread,TFairSpread,RankCorP,LogRtnCorP,ReferenceNameList,
                          TweakCDSSpreads,BPS_TO_NUMBER,
                          Fac10AltLogRtnCorP,Fac10AltRankCorP,Fac10AltSemiParamTransformedCDFHistDataDic,
                          AltLogRtnCorP,AltRankCorP,AltSemiParamTransformedCDFHistDataDic):
    GaussFairSpreadTweakCDS = np.zeros(shape=(5),dtype=np.float)
    TFairSpreadTweakCDS = np.zeros(shape=(5),dtype=np.float)

    TweakedDataTenorDic, TweakedImpProbDic, TweakedImpHazdRts, TweakedInvPWCDF = TweakCDSSpreads("All",10,TenorCreditSpreads,BPS_TO_NUMBER,DiscountFactorCurve,True,True)
    LatexTweakedDataTenorDic = convertToLaTeX(plotter,pd.DataFrame.from_dict(TweakedDataTenorDic),name="10 Times Credit Spreads",centerTable=False)
    GaussFairSpreadTweakCDS, TFairSpreadTweakCDS, t17 = FullMCFairSpreadValuation(plotter,time.time(),Fac10AltLogRtnCorP,Fac10AltRankCorP,M,HistCreditSpreads,
                                                                                                        Fac10AltSemiParamTransformedCDFHistDataDic, vE[0],TenorCreditSpreads,TweakedInvPWCDF,DiscountFactorCurve,
                                                                                                        TweakedImpHazdRts,TweakedDataTenorDic,CDSPaymentTenors,CDSBasketMaturity,
                                                                                                        name="10 Times Credit Spreads fair spread")

    latexAltFairSpreads = convertToLaTeX(plotter,pd.DataFrame(data=np.array([GaussFairSpreadTweakCDS,TFairSpreadTweakCDS],dtype=np.float), 
                                                    index = ["Gaussian", "Student's T"], columns=["1st to default","2nd to default","3rd to default","4th to default","5th to default"], dtype=np.float),
                                            "Fair Spreads from 10 times term structure of spreads and historical spreads",centerTable=False)

#Tweak Credit Spreads
def CreditSpreadTweak(plotter,M,HistCreditSpreads,SemiParamTransformedCDFHistDataDic, vE,
                          TenorCreditSpreads,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,
                          CDSPaymentTenors,CDSBasketMaturity, 
                          GaussFairSpread,TFairSpread,RankCorP,LogRtnCorP,ReferenceNameList,
                          TweakCDSSpreads,BPS_TO_NUMBER,
                          Fac10AltLogRtnCorP,Fac10AltRankCorP,Fac10AltSemiParamTransformedCDFHistDataDic,
                          AltLogRtnCorP,AltRankCorP,AltSemiParamTransformedCDFHistDataDic):
        
    #!Fair spread with alternative termstructure of credit spreads all tweaked------------------------------------------------------------------------------------------
    GaussFairSpreadTweakCDS = np.zeros(shape=(5),dtype=np.float)
    TFairSpreadTweakCDS = np.zeros(shape=(5),dtype=np.float)

    TweakedDataTenorDic, TweakedImpProbDic, TweakedImpHazdRts, TweakedInvPWCDF = TweakCDSSpreads("All",0.0002,TenorCreditSpreads,BPS_TO_NUMBER,DiscountFactorCurve,False)
    LatexTweakedDataTenorDic = convertToLaTeX(plotter,pd.DataFrame.from_dict(TweakedDataTenorDic),name="Alternative Term Structure of Credit Spreads",centerTable=False)
    print("Alternative Term Structure of Hazard Rates applied.")
    GaussFairSpreadTweakCDS, TFairSpreadTweakCDS, t17 = FullMCFairSpreadValuation(plotter,time.time(),LogRtnCorP,RankCorP,M,HistCreditSpreads,
                                                                                                        SemiParamTransformedCDFHistDataDic, vE[0],TenorCreditSpreads,TweakedInvPWCDF,DiscountFactorCurve,
                                                                                                        TweakedImpHazdRts,TweakedDataTenorDic,CDSPaymentTenors,CDSBasketMaturity,
                                                                                                        name="Alternative Term Structure of credit spreads")
    DeltaGaussFairSpreadTweakCDS = GaussFairSpreadTweakCDS - GaussFairSpread
    DeltaTFairSpreadTweakCDS = TFairSpreadTweakCDS - TFairSpread

    latexAltFairSpreads = convertToLaTeX(plotter,pd.DataFrame(data=np.array([GaussFairSpreadTweakCDS,TFairSpreadTweakCDS],dtype=np.float), 
                                                    index = ["Gaussian", "Student's T"], columns=["1st to default","2nd to default","3rd to default","4th to default","5th to default"], dtype=np.float),
                                            "Fair Spreads from alternative term structure of spreads",centerTable=False)

    #!Fair spread by tweaking individual credit spreads to calculate basket delta wrt individual reference names.------------------------------------------------------------------------------------------
    CreditTweaksToCarryOutBps = np.array([-10, -5, 5, 10, 50, 100, 250, 500, 750, 1000]) / BPS_TO_NUMBER

    GaussFairSpreadTweakCDS = np.zeros(shape=(5,5),dtype=np.float)
    TFairSpreadTweakCDS = np.zeros(shape=(5,5),dtype=np.float)
    DeltaGaussFairSpreadTweakCDS = dict()
    DeltaTFairSpreadTweakCDS = dict()
    for CreditTenorTweakAmount in CreditTweaksToCarryOutBps:
        for i in range(0,5*5,5):
            IndKey = TenorCreditSpreads['Ticker'][i]
            if not IndKey in DeltaGaussFairSpreadTweakCDS:
                DeltaGaussFairSpreadTweakCDS[IndKey] = dict()
                DeltaTFairSpreadTweakCDS[IndKey] = dict()
            TweakedDataTenorDic, TweakedImpProbDic, TweakedImpHazdRts, TweakedInvPWCDF = TweakCDSSpreads(IndKey,CreditTenorTweakAmount,TenorCreditSpreads,BPS_TO_NUMBER,DiscountFactorCurve)
            print("Tweaking the credit spreads for {0} and rerunning analysis for tweak: {1}".format(IndKey,CreditTenorTweakAmount))
            GaussFairSpreadTweakCDS[int(i/5)], TFairSpreadTweakCDS[int(i/5)], t17 = FullMCFairSpreadValuation(plotter,time.time(),LogRtnCorP,RankCorP,M,HistCreditSpreads,
                                                                                                                SemiParamTransformedCDFHistDataDic, vE[0],TenorCreditSpreads,TweakedInvPWCDF,
                                                                                                                DiscountFactorCurve,TweakedImpHazdRts,TweakedDataTenorDic,CDSPaymentTenors,
                                                                                                                CDSBasketMaturity,
                                                                                                                name="Tweak Spread for {0} by {1}".format(IndKey,CreditTenorTweakAmount))
            DeltaGaussFairSpreadTweakCDS[IndKey]["{0}".format(CreditTenorTweakAmount)] = GaussFairSpreadTweakCDS[int(i/5)] - GaussFairSpread
            DeltaTFairSpreadTweakCDS[IndKey]["{0}".format(CreditTenorTweakAmount)] = TFairSpreadTweakCDS[int(i/5)] - TFairSpread
        TweakedDataTenorDic, TweakedImpProbDic, TweakedImpHazdRts, TweakedInvPWCDF = TweakCDSSpreads("All",CreditTenorTweakAmount,TenorCreditSpreads,BPS_TO_NUMBER,DiscountFactorCurve)
        print("Tweaking the credit spreads for All reference names and rerunning analysis for tweak: {0}".format(CreditTenorTweakAmount))
        GaussFairSpreadTweakAllCDS, TFairSpreadTweakAllCDS, t175 = FullMCFairSpreadValuation(plotter,time.time(),LogRtnCorP,RankCorP,M,HistCreditSpreads,
                                                                                                        SemiParamTransformedCDFHistDataDic, vE[0],TenorCreditSpreads,TweakedInvPWCDF,
                                                                                                        DiscountFactorCurve,TweakedImpHazdRts,TweakedDataTenorDic,CDSPaymentTenors,
                                                                                                        CDSBasketMaturity,
                                                                                                        name="Bump all credit spreads by {0}".format(CreditTenorTweakAmount))
        if not "All" in DeltaGaussFairSpreadTweakCDS:
            DeltaGaussFairSpreadTweakCDS["All"] = dict()
            DeltaTFairSpreadTweakCDS["All"] = dict()
        DeltaGaussFairSpreadTweakCDS["All"]["{0}".format(CreditTenorTweakAmount)] = GaussFairSpreadTweakAllCDS - GaussFairSpread
        DeltaTFairSpreadTweakCDS["All"]["{0}".format(CreditTenorTweakAmount)] = TFairSpreadTweakAllCDS - TFairSpread

    CDSRefNamesArr = np.array(list(TenorCreditSpreads['Ticker'][0:25:5])+["All"])

    for RefName in CDSRefNamesArr:
        print("Calculating Credit Deltas under Gaussian Copula for ReferenceName {0}".format(RefName))
        deltaVBasket = np.transpose(list(DeltaGaussFairSpreadTweakCDS[RefName].values()))
        dBasket = deltaVBasket/CreditTweaksToCarryOutBps
        plotter.return_lineChart(CreditTweaksToCarryOutBps,deltaVBasket,
                            name="Credit Deltas ({0}) (Gauss)".format(RefName),xlabel="Credit Delta", ylabel="Basket Spread Delta",
                            legend=["1st to default","2nd to default","3rd to default","4th to default","5th to default"])
        plotter.return_lineChart(CreditTweaksToCarryOutBps,dBasket,
                            name="dBasketSpread_dCreditSpread ({0}) (Gauss)".format(RefName),xlabel="Credit Delta", ylabel="dV/ds",
                            legend=["1st to default","2nd to default","3rd to default","4th to default","5th to default"])
        fitY = dict()
        xLinY = dict()
        fitD = dict()
        xLinD = dict()
        for k in range(0,len(deltaVBasket)):
            fitYfn, rPowY, xLinY["{0}".format(k)] = plotter.SuitableRegressionFit(CreditTweaksToCarryOutBps,deltaVBasket[k],
                                name="Credit Deltas ({0}) (Gauss) Interpolated".format(RefName),startingPower=3)

            fitDfn, rPowD, xLinD["{0}".format(k)] = plotter.SuitableRegressionFit(CreditTweaksToCarryOutBps,dBasket[k],
                                name="dBasketSpread_dCreditSpread ({0}) (Gauss) Interpolated".format(RefName),startingPower=3)

            #print(xLinY["{0}".format(0)])
            fitY["{0}".format(k)] = np.fromiter(map(lambda x: fitYfn(x), xLinY["{0}".format(0)]),dtype=np.float)
            fitD["{0}".format(k)] = np.fromiter(map(lambda x: fitDfn(x), xLinD["{0}".format(0)]),dtype=np.float)
        
        plotter.return_lineChart(xLinY["{0}".format(0)],list(fitY.values()),
                        name="Plot of Interpolated_{1} Credit Deltas for {0} under Gauss Assumption".format(RefName,rPowY),xlabel="Credit Delta", ylabel="Basket Spread Delta",
                        legend=["1st to default","2nd to default","3rd to default","4th to default","5th to default"])
        plotter.return_lineChart(xLinD["{0}".format(0)],list(fitD.values()),
                        name="Plot of Interpolated_{1} dBasketSpread to dCreditSpread for {0} under Gauss Assumption".format(RefName,rPowD),xlabel="Credit Delta", ylabel="dV/ds",
                        legend=["1st to default","2nd to default","3rd to default","4th to default","5th to default"])

    
        
        print("Calculating Credit Deltas under Students T Copula for ReferenceName {0}".format(RefName))
        deltaVBasket = np.transpose(list(DeltaTFairSpreadTweakCDS[RefName].values()))
        dBasket = deltaVBasket/CreditTweaksToCarryOutBps
        plotter.return_lineChart(CreditTweaksToCarryOutBps,deltaVBasket,
                            name="Credit Deltas ({0}) (T)".format(RefName),xlabel="Credit Delta", ylabel="Basket Spread Delta",
                            legend=["1st to default","2nd to default","3rd to default","4th to default","5th to default"])
        plotter.return_lineChart(CreditTweaksToCarryOutBps,dBasket,
                            name="dBasketSpread_dCreditSpread ({0}) (T)".format(RefName),xlabel="Credit Delta", ylabel="dV/ds",
                            legend=["1st to default","2nd to default","3rd to default","4th to default","5th to default"])
        fitY = dict()
        xLinY = dict()
        fitD = dict()
        xLinD = dict()
        for k in range(0,len(deltaVBasket)):
            fitYfn, rPowY, xLinY["{0}".format(k)] = plotter.SuitableRegressionFit(CreditTweaksToCarryOutBps,deltaVBasket[k],
                                name="Credit Deltas ({0}) (T) Interpolated".format(RefName),startingPower=3)

            fitDfn, rPowD, xLinD["{0}".format(k)] = plotter.SuitableRegressionFit(CreditTweaksToCarryOutBps,dBasket[k],
                                name="dBasketSpread_dCreditSpread ({0}) (T) Interpolated".format(RefName),startingPower=3)

        
            fitY["{0}".format(k)] = np.fromiter(map(lambda x: fitYfn(x), xLinY["{0}".format(0)]),dtype=np.float)
            fitD["{0}".format(k)] = np.fromiter(map(lambda x: fitDfn(x), xLinD["{0}".format(0)]),dtype=np.float)

        plotter.return_lineChart(xLinY["{0}".format(0)],list(fitY.values()),
                        name="Plot of Interpolated_{1} Credit Deltas for {0} under Students T Assumption".format(RefName,rPowY),xlabel="Credit Delta", ylabel="Basket Spread Delta",
                        legend=["1st to default","2nd to default","3rd to default","4th to default","5th to default"])
        plotter.return_lineChart(xLinD["{0}".format(0)],list(fitD.values()),
                        name="Plot of Interpolated_{1} dBasketSpread to dCreditSpread for {0} under Students T Assumption".format(RefName,rPowD),xlabel="Credit Delta", ylabel="dV/ds",
                        legend=["1st to default","2nd to default","3rd to default","4th to default","5th to default"])




#!Sensitivity of basket to tweaking entire correlation matrix by percent using Fisher Transform.------------------------------------------------------------------------------------------
def CorTweaksAll(plotter, M,HistCreditSpreads,SemiParamTransformedCDFHistDataDic, vE,
                          TenorCreditSpreads,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,
                          CDSPaymentTenors,CDSBasketMaturity, 
                          GaussFairSpread,TFairSpread,RankCorP,LogRtnCorP,ReferenceNameList,
                          TweakCDSSpreads,BPS_TO_NUMBER,
                          Fac10AltLogRtnCorP,Fac10AltRankCorP,Fac10AltSemiParamTransformedCDFHistDataDic,
                          AltLogRtnCorP,AltRankCorP,AltSemiParamTransformedCDFHistDataDic):
    CorAllTweakFairSpreadGaussList = list()
    CorAllTweakFairSpreadTList = list()
    CorAllTweakFairSpreadGaussDiffP = list()
    CorAllTweakFairSpreadTDiffP = list()
    def PercTweakCors(percTweak):
        TweakedRankCorP = TweakWhole2DMatrixByPercent(RankCorP,percTweak)
        TweakedLogRtnCorP = TweakWhole2DMatrixByPercent(LogRtnCorP,percTweak)
        convertToLaTeX(plotter,pd.DataFrame(TweakedRankCorP, index = ReferenceNameList, columns=ReferenceNameList, dtype=np.float),
                                        name="Correlation sensitivity Rank Correlation tweaked by {0} percent".format(percTweak),centerTable=False)
        convertToLaTeX(plotter,pd.DataFrame(TweakedLogRtnCorP, index = ReferenceNameList, columns=ReferenceNameList, dtype=np.float),
                                        name="Correlation sensitivity Pearson Correlation tweaked by {0} percent".format(percTweak),centerTable=False)
        print("Calculating fair spread after tweaking correlation for whole matrix by {0}%".format(percTweak*100))
        CorAllTweakFairSpreadGauss,CorAllTweakFairSpreadT,tNew = FullMCFairSpreadValuation(plotter,time.time(),TweakedLogRtnCorP,TweakedRankCorP,M,HistCreditSpreads,SemiParamTransformedCDFHistDataDic, vE[0],TenorCreditSpreads,InvPWCDF,
                                                                                DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,
                                                                                name="Correlation between all reference names tweaked by {0} percent".format(percTweak*100))
        CorAllTweakFairSpreadGaussList.append(CorAllTweakFairSpreadGauss)
        CorAllTweakFairSpreadTList.append(CorAllTweakFairSpreadT)
        NonZeroGFAirSpread = np.copy(GaussFairSpread)
        NonZeroGFAirSpread[NonZeroGFAirSpread <= 0] = 0.0000001
        NonZeroTFAirSpread = np.copy(TFairSpread)
        NonZeroTFAirSpread[NonZeroTFAirSpread <= 0] = 0.0000001
        CorAllTweakFairSpreadGaussDiffP.append(((CorAllTweakFairSpreadGauss-GaussFairSpread)/NonZeroGFAirSpread)*100)
        CorAllTweakFairSpreadTDiffP.append(((CorAllTweakFairSpreadT-TFairSpread)/NonZeroTFAirSpread)*100)

        #plotter.return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([GaussFairSpread,CorAllTweakFairSpreadGauss]),
        #                                "Correlation between all reference names tweaked by {0} percent (Gaussian)".format(percTweak*100),xlabel="K-th to default",ylabel="Fair spread", 
        #                                legend=["Fair Spreads", "Tweaked Fair Spreads"])
        #plotter.return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([TFairSpread,CorAllTweakFairSpreadT]),
        #                                "Correlation between all reference names tweaked by {0} percent (Students T)".format(percTweak*100),xlabel="K-th to default",ylabel="Fair spread", 
        #                                legend=["Fair Spreads", "Tweaked Fair Spreads"])
    corPercTweaksArr = [-0.8,-0.5,-0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 0.8]
    for ptw in corPercTweaksArr:
        PercTweakCors(ptw)

    CorAllTweakFairSpreadGaussArr = np.array(CorAllTweakFairSpreadGaussList)
    CorAllTweakFairSpreadTArr = np.array(CorAllTweakFairSpreadTList)

    CorAllTweakFairSpreadGaussLines = np.transpose(CorAllTweakFairSpreadGaussArr)
    CorAllTweakFairSpreadTLines = np.transpose(CorAllTweakFairSpreadTArr)

    CorAllTweakFairSpreadGaussArrDiff = np.array(CorAllTweakFairSpreadGaussDiffP)
    CorAllTweakFairSpreadTArrDiff = np.array(CorAllTweakFairSpreadTDiffP)

    CorAllTweakFairSpreadGaussLinesDiff = np.transpose(CorAllTweakFairSpreadGaussArrDiff)
    CorAllTweakFairSpreadTLinesDiff = np.transpose(CorAllTweakFairSpreadTArrDiff)

    plotter.return_lineChart(corPercTweaksArr,CorAllTweakFairSpreadGaussLines,name="Sensitivity of fair spreads to percentage changes to correlation (Gaussian)",
                        xlabel="Percentage change to correlation",ylabel="Fair spread",legend=["1st to default","2nd to default","3rd to default","4th to default","5th to default"])
    plotter.return_lineChart(corPercTweaksArr,CorAllTweakFairSpreadGaussLinesDiff,name="Sensitivity of percentage changes in fair spreads to percentage changes to correlation (Gaussian)",
                        xlabel="Percentage change to correlation",ylabel="% change in fair spread",legend=["1st to default","2nd to default","3rd to default","4th to default","5th to default"])

    plotter.return_lineChart(corPercTweaksArr,CorAllTweakFairSpreadTLines,name="Sensitivity of fair spreads to percentage changes to correlation (Students T)",
                        xlabel="Percentage change to correlation",ylabel="Fair spread",legend=["1st to default","2nd to default","3rd to default","4th to default","5th to default"])
    plotter.return_lineChart(corPercTweaksArr,CorAllTweakFairSpreadTLinesDiff,name="Sensitivity of percentage changes in fair spreads to percentage changes to correlation (Students T)",
                        xlabel="Percentage change to correlation",ylabel="% change in fair spread",legend=["1st to default","2nd to default","3rd to default","4th to default","5th to default"])
    
    
#!Sensitivity of basket to tweaking individual pairwise correlations.------------------------------------------------------------------------------------------
def TweakPairwiseCors( plotter,M,HistCreditSpreads,SemiParamTransformedCDFHistDataDic, vE,
                          TenorCreditSpreads,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,
                          CDSPaymentTenors,CDSBasketMaturity, 
                          GaussFairSpread,TFairSpread,RankCorP,LogRtnCorP,ReferenceNameList,
                          TweakCDSSpreads,BPS_TO_NUMBER,
                          Fac10AltLogRtnCorP,Fac10AltRankCorP,Fac10AltSemiParamTransformedCDFHistDataDic,
                          AltLogRtnCorP,AltRankCorP,AltSemiParamTransformedCDFHistDataDic):
    t19 = time.time()
    print("Tweaking indiviual correlations by individual amounts")
    TweakedRankCorP = Tweak(RankCorP,(1,2),np.min([0.1, 1-RankCorP[1,2]]))
    TweakedLogRtnCorP = Tweak(LogRtnCorP,(1,2),np.min([0.1, 1-LogRtnCorP[1,2]]))
    FairSpreadGaussTweakCor,FairSpreadTTweakCor,t20 = FullMCFairSpreadValuation(plotter,t19,TweakedLogRtnCorP,TweakedRankCorP,M,HistCreditSpreads,SemiParamTransformedCDFHistDataDic, vE[0],TenorCreditSpreads,InvPWCDF,
                                                                            DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,name="Tweaked Correlation between Barclays & JPMorgan by 0,1")

    plotter.return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([GaussFairSpread,FairSpreadGaussTweakCor]),"Tweaked Correlation between Barclays & JPMorgan by 0,1 (Gaussian)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])
    plotter.return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([TFairSpread,FairSpreadTTweakCor]),"Tweaked Correlation between Barclays & JPMorgan by 0,1 (Students T)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])

    TweakedRankCorP = Tweak(RankCorP,(2,4),np.min([0.1, 1-RankCorP[2,4]]))
    TweakedLogRtnCorP = Tweak(LogRtnCorP,(2,4),np.min([0.1, 1-LogRtnCorP[2,4]]))
    FairSpreadGaussTweakCor,FairSpreadTTweakCor,t21 = FullMCFairSpreadValuation(plotter,t20,TweakedLogRtnCorP,TweakedRankCorP,M,HistCreditSpreads,SemiParamTransformedCDFHistDataDic, vE[0],TenorCreditSpreads,InvPWCDF,
                                                                            DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,name="Tweaked Correlation between JPMorgan & RBS by 0,1")
    plotter.return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([GaussFairSpread,FairSpreadGaussTweakCor]),"Tweaked Correlation between JPMorgan & RBS by 0,1 (Gaussian)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])
    plotter.return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([TFairSpread,FairSpreadTTweakCor]),"Tweaked Correlation between JPMorgan & RBS by 0,1 (Students T)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])

    TweakedRankCorP = Tweak(RankCorP,(0,3),np.max([-0.1, -1+RankCorP[0,3]]))
    TweakedLogRtnCorP = Tweak(LogRtnCorP,(0,3),np.max([-0.1, -1+LogRtnCorP[0,3]]))
    FairSpreadGaussTweakCor,FairSpreadTTweakCor,t22 = FullMCFairSpreadValuation(plotter,t21,TweakedLogRtnCorP,TweakedRankCorP,M,HistCreditSpreads,SemiParamTransformedCDFHistDataDic, vE[0],TenorCreditSpreads,InvPWCDF,
                                                                            DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,name="Tweaked Correlation between Deutsche Bank & Goldman Sachs by -0,1")
    plotter.return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([GaussFairSpread,FairSpreadGaussTweakCor]),"Tweaked Correlation between Deutsche Bank & Goldman Sachs by -0,1 (Gaussian)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])
    plotter.return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([TFairSpread,FairSpreadTTweakCor]),"Tweaked Correlation between Deutsche Bank & Goldman Sachs by -0,1 (Students T)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])

    TweakedRankCorP = Tweak(RankCorP,(1,2),np.min([0.1, 1-RankCorP[1,2]]))
    TweakedLogRtnCorP = Tweak(LogRtnCorP,(1,2),np.min([0.1, 1-LogRtnCorP[1,2]]))
    TweakedRankCorP = Tweak(RankCorP,(0,4),np.min([0.1, 1-RankCorP[0,4]]))
    TweakedLogRtnCorP = Tweak(LogRtnCorP,(0,4),np.min([0.1, 1-LogRtnCorP[0,4]]))
    TweakedRankCorP = Tweak(RankCorP,(2,3),np.min([0.1, 1-RankCorP[2,3]]))
    TweakedLogRtnCorP = Tweak(LogRtnCorP,(2,3),np.min([0.1, 1-LogRtnCorP[2,3]]))
    FairSpreadGaussTweakCor,FairSpreadTTweakCor,t23 = FullMCFairSpreadValuation(plotter,t22,TweakedLogRtnCorP,TweakedRankCorP,M,HistCreditSpreads,SemiParamTransformedCDFHistDataDic, vE[0],TenorCreditSpreads,InvPWCDF,
                                                                            DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,name="Tweaked Correlation between Barclays & JPMorgan, Deutsche Bank & RBS and JPMorgan & Goldman Sachs by 0,1")
    plotter.return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([GaussFairSpread,FairSpreadGaussTweakCor]),"Tweaked Correlation between Barclays & JPMorgan, Deutsche Bank & RBS and JPMorgan & Goldman Sachs by 0,1 (Gaussian)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])
    plotter.return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([TFairSpread,FairSpreadTTweakCor]),"Tweaked Correlation between Barclays & JPMorgan, Deutsche Bank & RBS and JPMorgan & Goldman Sachs by 0,1 (Students T)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])
    #TweakedTLegs = SimulateCDSBasketDefaultsAndValueLegsT(t6,TweakedRankCorP,M,HistCreditSpreads,TransformedHistDataDic,TenorCreditSpreads,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity)
    #TweakedGaussLegs = SimulateCDSBasketDefaultsAndValueLegsGauss(TweakedTLegs[2],TweakedLogRtnCorP,M,HistCreditSpreads,TenorCreditSpreads,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity)

    #TweakedGaussFairSpread,t9 = CalculateFairSpreadFromLegs(TweakedGaussLegs[0],TweakedGaussLegs[1],M,TweakedGaussLegs[2],"Gauss")
    #TweakedTFairSpread,t10 = CalculateFairSpreadFromLegs(TweakedTLegs[0],TweakedTLegs[1],M,TweakedTLegs[2],"T")

    TweakedRankCorP = SetArbitrarily(RankCorP,(1,2),0.9)
    TweakedLogRtnCorP = SetArbitrarily(LogRtnCorP,(1,2),0.9)
    FairSpreadGaussTweakCor,FairSpreadTTweakCor,t24 = FullMCFairSpreadValuation(plotter,t23,TweakedLogRtnCorP,TweakedRankCorP,M,HistCreditSpreads,SemiParamTransformedCDFHistDataDic, vE[0],TenorCreditSpreads,InvPWCDF,
                                                                            DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,name="Correlation between Barclays & JPMorgan set to 0.9")
    plotter.return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([GaussFairSpread,FairSpreadGaussTweakCor]),"Correlation between Barclays & JPMorgan set to 0.9 (Gaussian)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])
    plotter.return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([TFairSpread,FairSpreadTTweakCor]),"Correlation between Barclays & JPMorgan set to 0.9 (Students T)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])


    TweakedRankCorP = SetArbitrarily(RankCorP,(0,3),0.95)
    TweakedLogRtnCorP = SetArbitrarily(LogRtnCorP,(0,3),0.95)
    FairSpreadGaussTweakCor,FairSpreadTTweakCor,t25 = FullMCFairSpreadValuation(plotter,t24,TweakedLogRtnCorP,TweakedRankCorP,M,HistCreditSpreads,SemiParamTransformedCDFHistDataDic, vE[0],TenorCreditSpreads,InvPWCDF,
                                                                            DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,name="Correlation between Deutsche Bank & Goldman Sachs set to 0.95 ")
    plotter.return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([GaussFairSpread,FairSpreadGaussTweakCor]),"Correlation between Deutsche Bank & Goldman Sachs set to 0.95 (Gaussian)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])
    plotter.return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([TFairSpread,FairSpreadTTweakCor]),"Correlation between Deutsche Bank & Goldman Sachs set to 0.95 (Students T)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])


    TweakedRankCorP = SetArbitrarily(RankCorP,(3,4),0.05)
    TweakedLogRtnCorP = SetArbitrarily(LogRtnCorP,(3,4),0.05)
    FairSpreadGaussTweakCor,FairSpreadTTweakCor,t26 = FullMCFairSpreadValuation(plotter,t25,TweakedLogRtnCorP,TweakedRankCorP,M,HistCreditSpreads,SemiParamTransformedCDFHistDataDic, vE[0],TenorCreditSpreads,InvPWCDF,
                                                                            DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,name="Correlation between Goldman Sachs & RBS set to 0.05")
    plotter.return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([GaussFairSpread,FairSpreadGaussTweakCor]),"Correlation between Goldman Sachs & RBS set to 0.05 (Gaussian)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])
    plotter.return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([TFairSpread,FairSpreadTTweakCor]),"Correlation between Goldman Sachs & RBS set to 0.05 (Students T)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])

    TweakedRankCorP = SetArbitrarily(RankCorP,(0,3),-0.95)
    TweakedLogRtnCorP = SetArbitrarily(LogRtnCorP,(0,3),-0.95)
    FairSpreadGaussTweakCor,FairSpreadTTweakCor,t27 = FullMCFairSpreadValuation(plotter,t26,TweakedLogRtnCorP,TweakedRankCorP,M,HistCreditSpreads,SemiParamTransformedCDFHistDataDic, vE[0],TenorCreditSpreads,InvPWCDF,
                                                                            DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,name="Correlation between Deutsche Bank & Goldman Sachs set to -0.95")
    plotter.return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([GaussFairSpread,FairSpreadGaussTweakCor]),"Correlation between Deutsche Bank & Goldman Sachs set to -0.95 (Gaussian)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])
    plotter.return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([TFairSpread,FairSpreadTTweakCor]),"Correlation between Deutsche Bank & Goldman Sachs set to -0.95 (Students T)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])

    TweakedRankCorP = SetWhole2DMatrixArbitrarily(RankCorP,-0.01)
    TweakedLogRtnCorP = SetWhole2DMatrixArbitrarily(LogRtnCorP,-0.01)
    FairSpreadGaussTweakCor,FairSpreadTTweakCor,t28 = FullMCFairSpreadValuation(plotter,t27,TweakedLogRtnCorP,TweakedRankCorP,M,HistCreditSpreads,SemiParamTransformedCDFHistDataDic, vE[0],TenorCreditSpreads,InvPWCDF,
                                                                            DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,name="Correlation between all reference names set to -0.01")
    plotter.return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([GaussFairSpread,FairSpreadGaussTweakCor]),"Correlation between all reference names set to -0.01 (Gaussian)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])
    plotter.return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([TFairSpread,FairSpreadTTweakCor]),"Correlation between all reference names set to -0.01 (Students T)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])

    TweakedRankCorP = SetWhole2DMatrixArbitrarily(RankCorP,-0.99)
    TweakedLogRtnCorP = SetWhole2DMatrixArbitrarily(LogRtnCorP,-0.99)
    FairSpreadGaussTweakCor,FairSpreadTTweakCor,t29 = FullMCFairSpreadValuation(plotter,t28,TweakedLogRtnCorP,TweakedRankCorP,M,HistCreditSpreads,SemiParamTransformedCDFHistDataDic, vE[0],TenorCreditSpreads,InvPWCDF,
                                                                            DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,name="Correlation between all reference names set to -0.99")
    plotter.return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([GaussFairSpread,FairSpreadGaussTweakCor]),"Correlation between all reference names set to -0.99 (Gaussian)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])
    plotter.return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([TFairSpread,FairSpreadTTweakCor]),"Correlation between all reference names set to -0.99 (Students T)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])

    TweakedRankCorP = SetWhole2DMatrixArbitrarily(RankCorP,0.99)
    TweakedLogRtnCorP = SetWhole2DMatrixArbitrarily(LogRtnCorP,0.99)
    FairSpreadGaussTweakCor,FairSpreadTTweakCor,t30 = FullMCFairSpreadValuation(plotter,t29,TweakedLogRtnCorP,TweakedRankCorP,M,HistCreditSpreads,SemiParamTransformedCDFHistDataDic, vE[0],TenorCreditSpreads,InvPWCDF,
                                                                            DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,name="Correlation between all reference names set to 0.99")
    plotter.return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([GaussFairSpread,FairSpreadGaussTweakCor]),"Correlation between all reference names set to 0.99 (Gaussian)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])
    plotter.return_scatter_multdependencies(np.arange(1,6,1,dtype=np.int),np.array([TFairSpread,FairSpreadTTweakCor]),"Correlation between all reference names set to 0.99 (Students T)",xlabel="K-th to default",ylabel="Fair spread", legend=["Fair Spreads", "Tweaked Fair Spreads"])

results_bank=[]
#if __name__ == '__main__':
    
    
#    fn_dict = dict()
#    fns = [pCheckFairSpead,IRTweak,RTweak,AltHistDataTweak,AllFac10CreditAndHist,CreditSpreadTweak,CorTweaksAll,TweakPairwiseCors]
#    for fn in fns:
#        working_q.put(fn.__name__)
#        fn_dict[fn.__name__] = fn

#    #kwsargs = dict([(M,M),(HistCreditSpreads,HistCreditSpreads),(SemiParamTransformedCDFHistDataDic,SemiParamTransformedCDFHistDataDic),( vE,vE),(
#    #                      TenorCreditSpreads,TenorCreditSpreads),(InvPWCDF,InvPWCDF),(DiscountFactorCurve,DiscountFactorCurve),(ImpHazdRts,ImpHazdRts),(DataTenorDic,DataTenorDic),(
#    #                      CDSPaymentTenors,CDSPaymentTenors),(CDSBasketMaturity,CDSBasketMaturity),( 
#    #                      GaussFairSpread,GaussFairSpread),(TFairSpread,TFairSpread),(RankCorP,RankCorP),(LogRtnCorP,LogRtnCorP),(ReferenceNameList,ReferenceNameList),(
#    #                      TweakCDSSpreads,TweakCDSSpreads),(BPS_TO_NUMBER,BPS_TO_NUMBER),(
#    #                      Fac10AltLogRtnCorP,Fac10AltLogRtnCorP),(Fac10AltRankCorP,Fac10AltRankCorP),(Fac10AltSemiParamTransformedCDFHistDataDic,Fac10AltSemiParamTransformedCDFHistDataDic),(
#    #                      AltLogRtnCorP,AltLogRtnCorP),(AltRankCorP,AltRankCorP),(AltSemiParamTransformedCDFHistDataDic,AltSemiParamTransformedCDFHistDataDic)])

#    processes = [mp.Process(target=worker,args=(working_q, output_q, fn_dict,M,HistCreditSpreads,SemiParamTransformedCDFHistDataDic, vE,
#                          TenorCreditSpreads,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,
#                          CDSPaymentTenors,CDSBasketMaturity, 
#                          GaussFairSpread,TFairSpread,RankCorP,LogRtnCorP,ReferenceNameList,
#                          TweakCDSSpreads,BPS_TO_NUMBER,
#                          Fac10AltLogRtnCorP,Fac10AltRankCorP,Fac10AltSemiParamTransformedCDFHistDataDic,
#                          AltLogRtnCorP,AltRankCorP,AltSemiParamTransformedCDFHistDataDic)) 
#                 for i in range(mp.cpu_count())]

#    #while working_q.empty() == False:
#    #    worker(working_q, output_q, fn_dict,M=M,HistCreditSpreads=HistCreditSpreads,SemiParamTransformedCDFHistDataDic=SemiParamTransformedCDFHistDataDic, vE=vE,
#    #                      TenorCreditSpreads=TenorCreditSpreads,InvPWCDF=InvPWCDF,DiscountFactorCurve=DiscountFactorCurve,ImpHazdRts=ImpHazdRts,DataTenorDic=DataTenorDic,
#    #                      CDSPaymentTenors=CDSPaymentTenors,CDSBasketMaturity=CDSBasketMaturity, 
#    #                      GaussFairSpread=GaussFairSpread,TFairSpread=TFairSpread,RankCorP=RankCorP,LogRtnCorP=LogRtnCorP,ReferenceNameList=ReferenceNameList,
#    #                      TweakCDSSpreads=TweakCDSSpreads,BPS_TO_NUMBER=BPS_TO_NUMBER,
#    #                      Fac10AltLogRtnCorP=Fac10AltLogRtnCorP,Fac10AltRankCorP=Fac10AltRankCorP,Fac10AltSemiParamTransformedCDFHistDataDic=Fac10AltSemiParamTransformedCDFHistDataDic,
#    #                      AltLogRtnCorP=AltLogRtnCorP,AltRankCorP=AltRankCorP,AltSemiParamTransformedCDFHistDataDic=AltSemiParamTransformedCDFHistDataDic)

#    for i,p in enumerate(processes):
#        print("Starting processes")
#        p.start()
#        print("started %d" %i)

#    for i,p in enumerate(processes):
#        print("joining %d" %i)
#        p.join()

#    #with mp.Pool(processes=mp.cpu_count()) as pool:
#    #    # launching multiple evaluations asynchronously *may* use more processes
#    #    multiple_results = [pool.apply_async(worker, (working_q, output_q, fn_dict,M,HistCreditSpreads,SemiParamTransformedCDFHistDataDic, vE,
#    #                      TenorCreditSpreads,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity, 
#    #                      GaussFairSpread,TFairSpread,RankCorP,LogRtnCorP,ReferenceNameList,TweakCDSSpreads,BPS_TO_NUMBER,
#    #                      Fac10AltLogRtnCorP,Fac10AltRankCorP,Fac10AltSemiParamTransformedCDFHistDataDic,AltLogRtnCorP,AltRankCorP,AltSemiParamTransformedCDFHistDataDic)) 
#    #                        for i in range(mp.cpu_count())]
#    #    [res.get() for res in multiple_results]
#    #processes = [mp.Process(target=worker,args=(working_q, output_q)) for i in range(mp.cpu_count())]
#    #for proc in processes:
#    #    proc.start()
#    #for proc in processes:
#    #    proc.join()
#    while True:
#       if output_q.empty() == True:
#           break
#       results_bank.append(output_q.get_nowait())
#    print(results_bank)
#    print("Program took: {0} minutes to finish, Enter any key to finish. (Debug Point)".format((time.time()-t1)/60))

#plotter.save_all_figs()
#userIn = input()

#debug = True

#AltHistDataTweak(plotter,M,HistCreditSpreads,SemiParamTransformedCDFHistDataDic, vE,
#                        TenorCreditSpreads,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,
#                        CDSPaymentTenors,CDSBasketMaturity, 
#                        GaussFairSpread,TFairSpread,RankCorP,LogRtnCorP,ReferenceNameList,
#                        TweakCDSSpreads,BPS_TO_NUMBER,
#                        Fac10AltLogRtnCorP,Fac10AltRankCorP,Fac10AltSemiParamTransformedCDFHistDataDic,
#                        AltLogRtnCorP,AltRankCorP,AltSemiParamTransformedCDFHistDataDic)

def f(x):
    return x*x

print("hello_outside")

if __name__ == '__main__':
    print("hello_inside")

    fn_dict = dict()
    #fns = [pCheckFairSpead,IRTweak,RTweak,AltHistDataTweak,AllFac10CreditAndHist,CreditSpreadTweak,CorTweaksAll,TweakPairwiseCors]
    fns = [CreditSpreadTweak]
    for fn in fns:
        working_q.put(fn.__name__)
        fn_dict[fn.__name__] = fn

    processes = [mp.Process(target=worker,args=(working_q, output_q, fn_dict,Plotter(),M,HistCreditSpreads,SemiParamTransformedCDFHistDataDic, vE,
                        TenorCreditSpreads,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,
                        CDSPaymentTenors,CDSBasketMaturity, 
                        GaussFairSpread,TFairSpread,RankCorP,LogRtnCorP,ReferenceNameList,
                        TweakCDSSpreads,BPS_TO_NUMBER,
                        Fac10AltLogRtnCorP,Fac10AltRankCorP,Fac10AltSemiParamTransformedCDFHistDataDic,
                        AltLogRtnCorP,AltRankCorP,AltSemiParamTransformedCDFHistDataDic)) 
                for i in range(mp.cpu_count())]
    for i,p in enumerate(processes):
        print("Starting processes")
        p.start()
        print("started %d" %i)

    for i,p in enumerate(processes):
        print("joining %d" %i)
        p.join()

    while True:
       if output_q.empty() == True:
           break
       results_bank.append(output_q.get_nowait())
    print(results_bank)
    print("Program took: {0} minutes to finish, Enter any key to finish. (Debug Point)".format((time.time()-t1)/60))
    plotter.save_all_figs()

    mpmm = mp.Manager()
    # start 4 worker processes
    with mp.Pool(processes=4) as pool:

        ## print "[0, 1, 4,..., 81]"
        #print(pool.map(f, range(10)))

        ## print same numbers in arbitrary order
        #for i in pool.imap_unordered(f, range(10)):
        #    print(i)

        ## evaluate "f(20)" asynchronously
        #res = pool.apply_async(f, (20,))      # runs in *only* one process
        #print(res.get(timeout=1))             # prints "400"

        ## evaluate "os.getpid()" asynchronously
        #res = pool.apply_async(os.getpid, ()) # runs in *only* one process
        #print(res.get(timeout=1))             # prints the PID of that process

        ## launching multiple evaluations asynchronously *may* use more processes
        #multiple_results = [pool.apply_async(os.getpid, ()) for i in range(4)]
        #print([res.get(timeout=1) for res in multiple_results])

        # make a single worker sleep for 10 secs
        res = pool.apply_async(time.sleep, (10,))

        try:
            print(res.get())
        except mp.TimeoutError:
            print("We lacked patience and got a multiprocessing.TimeoutError")

        print("For the moment, the pool remains available for more work")

    # exiting the 'with'-block has stopped the pool
    print("Now the pool is closed and no longer available")


