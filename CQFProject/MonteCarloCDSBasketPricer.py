import math
import numpy as np
from LowDiscrepancyNumberGenerators import SobolNumbers
import time
import queue
import threading
from multiprocessing.pool import ThreadPool
#import multiprocessing

#from HazardRates import *
#from Returns import *
#from EmpiricalFunctions import *
#from Sorting import *
from SimulateLegs import SimulateLegPricesFromCorrelationNormal, SimulateLegPricesFromCorrelationT, UnifFromGaussCopula, UnifFromTCopula
from RunningMoments import RunningAverage, RunningVarianceOfRunningAverage
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!MONTE CARLO SIMULATION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

pool = ThreadPool(processes=1)

def SimulateCDSBasketDefaultsAndValueLegsT(Plotter,TimeAtStart,CorP,M,HistCreditSpreads,TransformedHistDataDic,T_df,TenorCreditSpreads,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,SpreadArr=[],R=0.4,name=""):
    if len(SpreadArr) == 0:
        SpreadArr = np.ones(shape=(5))
    CompensationLegSumsForEachRefNameT = np.zeros(shape=(M,5),dtype=np.float)
    PremiumLegSumsForEachRefNameT = np.zeros(shape=(M,5),dtype=np.float)
    NumbGen = SobolNumbers()
    NumbGen.initialise(CorP.shape[0])
    UT = UnifFromTCopula(CorP,NumbGen,T_df,M)
    UniformityDic = dict()
    for i in range(2,4):
        UniformityDic["T Copula %d" % (i+1)] = UT[i,:]
    Plotter.lock()
    Plotter.plot_histogram_array(UniformityDic,"Simulated Ui T Copula",name=name)
    Plotter.plot_codependence_scatters(UniformityDic,"Simulated Ui T Copula", "Simulated Uj T Copula",name)
    Plotter.unlock()
    M_Min = 100
    Tolerance = 0.000001
    CompLegRunningAv = np.zeros(shape=(1,5))
    PremLegRunningAv = np.zeros(shape=(1,5))
    runningVarCompLeg = []
    runningVarPremLeg = []
    out_results = []

    def RunMonteCarlo(InnerM,baseM,CompensationLegSumsForEachRefNameT,PremiumLegSumsForEachRefNameT,PremLegRunningAv,CompLegRunningAv,runningVarCompLeg,runningVarPremLeg,out_results):
        for m in range(baseM+1,baseM+InnerM+1):
            TLegsSum = SimulateLegPricesFromCorrelationT(HistCreditSpreads,TenorCreditSpreads,CDSPaymentTenors,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,UT[:,m-1],int(CDSBasketMaturity),SpreadArr,R,0.25)
            CompensationLegSumsForEachRefNameT[m-1] = [TLegsSum[1][0],TLegsSum[2][0],TLegsSum[3][0],TLegsSum[4][0],TLegsSum[5][0]]
            PremiumLegSumsForEachRefNameT[m-1] = [TLegsSum[1][1],TLegsSum[2][1],TLegsSum[3][1],TLegsSum[4][1],TLegsSum[5][1]]
            #CompLegRunningAv[m-1,:] = ((CompLegRunningAv[m-2,:] * (m-1)) + CompensationLegSumsForEachRefNameT[m-1]) / (m)
            #PremLegRunningAv[m-1,:] = ((PremLegRunningAv[m-2,:] * (m-1)) + PremiumLegSumsForEachRefNameT[m-1]) / (m)
            xd = CompLegRunningAv.shape[0]
            CompLegRunningAv = np.concatenate((CompLegRunningAv, [((CompLegRunningAv[xd-1,:] * (xd)) + CompensationLegSumsForEachRefNameT[m-1]) / (xd+1)]), axis=0)
            PremLegRunningAv = np.concatenate((PremLegRunningAv, [((PremLegRunningAv[xd-1,:] * (xd)) + PremiumLegSumsForEachRefNameT[m-1]) / (xd+1)]), axis=0)
            if (m-1) % M_Min == 0 and m > M_Min + baseM:
                #runningVarCompLeg = RunningVarianceOfRunningAverage(np.array(CompLegRunningAv[0:(m-1),:]).transpose(),M_Min)
                #runningVarPremLeg = RunningVarianceOfRunningAverage(np.array(PremLegRunningAv[0:(m-1),:]).transpose(),M_Min)
                runningVarCompLeg = RunningVarianceOfRunningAverage(np.array(CompLegRunningAv[:,:]).transpose(),M_Min)
                runningVarPremLeg = RunningVarianceOfRunningAverage(np.array(PremLegRunningAv[:,:]).transpose(),M_Min)
                supRunningVarsCompLeg = max(runningVarCompLeg.transpose()[runningVarCompLeg.shape[1]-1])
                supRunningVarsPremLeg = max(runningVarPremLeg.transpose()[runningVarPremLeg.shape[1]-1])
                sup = max(supRunningVarsCompLeg,supRunningVarsPremLeg)
                if sup < Tolerance:
                    print("Tolerance hit for T")
                    out_results.append((runningVarCompLeg, runningVarPremLeg))
                    break
                elif (m-1+M_Min) > baseM+InnerM:
                    out_results.insert(0,(runningVarCompLeg, runningVarPremLeg)) 
    threads = 1   
    jobs = []
    innerM = round(M/threads)
    for r in range(0,threads):
        thread = threading.Thread(target=RunMonteCarlo(innerM,r*innerM,CompensationLegSumsForEachRefNameT,PremiumLegSumsForEachRefNameT,
                                                       CompLegRunningAv,PremLegRunningAv,runningVarCompLeg,runningVarPremLeg,out_results))
        jobs.append(thread)

	# Start the threads (i.e. calculate the random number lists)
    for j in jobs:
        j.start()
    
    # Ensure all of the threads have finished
    for j in jobs:
        j.join()

    runningVarCompLeg = out_results[-1][0]
    runningVarPremLeg = out_results[-1][1]
    Plotter.lock()
    Plotter.return_lineChart(np.arange(0,M),CompLegRunningAv.transpose(),name+"_"+"Compensation Leg Running Average (Student's T Copula)",xlabel="Iteration",ylabel="Running Average", legend=["1st to Default","2nd to Default","3rd to Default","4th to Default","5th to Default"],numberPlot=1,noOfPlotsW=2,noOfPlotsH=2,trimTrailingZeros=True)
    Plotter.return_lineChart(np.arange(0,M),PremLegRunningAv.transpose(),name+"_"+"Premium Leg Running Average (Student's T Copula)",xlabel="Iteration",ylabel="Running Average", legend=["1st to Default","2nd to Default","3rd to Default","4th to Default","5th to Default"],numberPlot=2,noOfPlotsW=2,noOfPlotsH=2,trimTrailingZeros=True)
    Plotter.return_lineChart(np.arange(0,runningVarCompLeg.shape[1]),runningVarCompLeg,name+"_"+"Running Variance of Compensation Leg (Student's T Copula)",legend=["1st to Default","2nd to Default","3rd to Default","4th to Default","5th to Default"],numberPlot=3,noOfPlotsW=2,noOfPlotsH=2,trimTrailingZeros=True)
    Plotter.return_lineChart(np.arange(0,runningVarCompLeg.shape[1]),runningVarPremLeg,name+"_"+"Running Variance of Premium Leg (Student's T Copula)",legend=["1st to Default","2nd to Default","3rd to Default","4th to Default","5th to Default"],numberPlot=4,noOfPlotsW=2,noOfPlotsH=2,trimTrailingZeros=True)
    Plotter.unlock()
    tEnd = time.time()
    print("Took %.10f seconds to simulate %d iterations of T copulae and calculate legs from them." % (tEnd - TimeAtStart,M))

    return CompensationLegSumsForEachRefNameT,PremiumLegSumsForEachRefNameT,tEnd

def SimulateCDSBasketDefaultsAndValueLegsGauss(Plotter,TimeAtStart,CorP,M,HistCreditSpreads,TenorCreditSpreads,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,SpreadArr=[],R=0.4,name=""):
    if len(SpreadArr) == 0:
        SpreadArr = np.ones(shape=(5))
    CompensationLegSumsForEachRefNameGauss = np.zeros(shape=(M,5),dtype=np.float)
    PremiumLegSumsForEachRefNameGauss = np.zeros(shape=(M,5),dtype=np.float)
    NumbGen = SobolNumbers()
    NumbGen.initialise(CorP.shape[0])
    UNorm = UnifFromGaussCopula(CorP,NumbGen,M)
    UniformityDic = dict()
    for i in range(0,2):
        UniformityDic["Gaussian Copula %d" % (i+1)] = UNorm[i,:]
    Plotter.lock()
    Plotter.plot_histogram_array(UniformityDic,"Simulated Ui Gaussian Copula",name=name)
    Plotter.plot_codependence_scatters(UniformityDic,"Simulated Ui Gaussian Copula","Simulated Uj Gaussian Copula",name)
    Plotter.unlock()
    M_Min = 100
    Tolerance = 0.000001
    CompLegRunningAv = np.zeros(shape=(M,5))
    PremLegRunningAv = np.zeros(shape=(M,5))
    for m in range(1,M):
        GaussLegsSum = SimulateLegPricesFromCorrelationNormal(HistCreditSpreads,TenorCreditSpreads,CDSPaymentTenors,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,UNorm[:,m-1],int(CDSBasketMaturity),SpreadArr,R,0.25)
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
                print("Tolerance hit for Gaussian")
                break
    Plotter.lock()
    Plotter.return_lineChart(np.arange(0,M),CompLegRunningAv.transpose(),name+"_"+"Compensation Leg Running Average (Gaussian Copula)",xlabel="Iteration",ylabel="Spread", legend=["1st to Default","2nd to Default","3rd to Default","4th to Default","5th to Default"],numberPlot=1,noOfPlotsW=2,noOfPlotsH=2,trimTrailingZeros=True)
    Plotter.return_lineChart(np.arange(0,M),PremLegRunningAv.transpose(),name+"_"+"Premium Leg Running Average (Gaussian Copula)",xlabel="Iteration",ylabel="Spread", legend=["1st to Default","2nd to Default","3rd to Default","4th to Default","5th to Default"],numberPlot=2,noOfPlotsW=2,noOfPlotsH=2,trimTrailingZeros=True)
    Plotter.return_lineChart(np.arange(0,len(runningVarCompLeg[0])),runningVarCompLeg,name+"_"+"Running Variance of Compensation Leg (Gaussian Copula)",xlabel="Iteration",ylabel="Spread",legend=["1st to Default","2nd to Default","3rd to Default","4th to Default","5th to Default"],numberPlot=3,noOfPlotsW=2,noOfPlotsH=2,trimTrailingZeros=True)
    Plotter.return_lineChart(np.arange(0,len(runningVarCompLeg[0])),runningVarPremLeg,name+"_"+"Running Variance of Premium Leg (Gaussian Copula)",xlabel="Iteration",ylabel="Spread",legend=["1st to Default","2nd to Default","3rd to Default","4th to Default","5th to Default"],numberPlot=4,noOfPlotsW=2,noOfPlotsH=2,trimTrailingZeros=True)
    Plotter.unlock()
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

def FullMCFairSpreadValuation(Plotter,startTime,LogRtnCorP,RankCorP,M,HistCreditSpreads,TransformedHistDataDic,T_df,TenorCreditSpreads,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,R=0.4,name=""):

    TLegs = SimulateCDSBasketDefaultsAndValueLegsT(Plotter,startTime,RankCorP,M,HistCreditSpreads,TransformedHistDataDic,T_df,TenorCreditSpreads,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,[],R,name)
    GaussLegs = SimulateCDSBasketDefaultsAndValueLegsGauss(Plotter,TLegs[2],LogRtnCorP,M,HistCreditSpreads,TenorCreditSpreads,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,[],R,name)

    #def T_sim():
    #    return SimulateCDSBasketDefaultsAndValueLegsT(startTime,RankCorP,M,HistCreditSpreads,TransformedHistDataDic,T_df,TenorCreditSpreads,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,[],R,name)
    #def G_sim():
    #    return SimulateCDSBasketDefaultsAndValueLegsGauss(startTime,LogRtnCorP,M,HistCreditSpreads,TenorCreditSpreads,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,CDSBasketMaturity,[],R,name)
    
    #async_result_T = pool.apply_async(SimulateCDSBasketDefaultsAndValueLegsT, (startTime,RankCorP,M,HistCreditSpreads,TransformedHistDataDic,T_df,
    #                                                                         TenorCreditSpreads,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,
    #                                                                         CDSPaymentTenors,CDSBasketMaturity,[],R,name)) # tuple of args for foo
    #proc1 = multiprocessing.Process(target=SimulateCDSBasketDefaultsAndValueLegsT,args=(startTime,RankCorP,M,HistCreditSpreads,TransformedHistDataDic,T_df,
    #                                                                            TenorCreditSpreads,InvPWCDF,DiscountFactorCurve,ImpHazdRts,DataTenorDic,
    #                                                                            CDSPaymentTenors,CDSBasketMaturity,[],R,name))

    ## do some other stuff in the main process
    ##async_result_G = pool.apply_async(SimulateCDSBasketDefaultsAndValueLegsGauss, (startTime,LogRtnCorP,M,HistCreditSpreads,TenorCreditSpreads,InvPWCDF,
    ##                                                                               DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,
    ##                                                                               CDSBasketMaturity,[],R,name)) # tuple of args for foo
    #proc2 = multiprocessing.Process(target=SimulateCDSBasketDefaultsAndValueLegsGauss,args=(startTime,LogRtnCorP,M,HistCreditSpreads,TenorCreditSpreads,InvPWCDF,
    #                                                                               DiscountFactorCurve,ImpHazdRts,DataTenorDic,CDSPaymentTenors,
    #                                                                               CDSBasketMaturity,[],R,name))
    #TLegs = async_result_T.get()  # get the return value from your function.
    #GaussLegs = async_result_G.get()  # get the return value from your function.


    #if __name__ == "__main__":
    #    procs = [T_sim, G_sim]
    
    #    jobs = []
    #    #pipe_list = []
    #    manager = multiprocessing.Manager()
    #    result_dict = dict()

    #    for i in range(len(procs)):
    #        p = multiprocessing.process(target=workerWrapper,args=(i,result_dict,procs[i]))
    #        jobs.append(p)
    #        p.start()

    #    for j in jobs:
    #        j.join()


    #    TLegs = result_dict[0]
    #    GaussLegs  = result_dict[1]

    GaussFairSpread,t9 = CalculateFairSpreadFromLegs(GaussLegs[0],GaussLegs[1],M,GaussLegs[2],"Gauss")
    TFairSpread,t10 = CalculateFairSpreadFromLegs(TLegs[0],TLegs[1],M,t9,"T")
    tEnd = time.time()
    
    print("Took {0} secs to run FullMCSpead Analysis".format(tEnd-startTime))

    return GaussFairSpread, TFairSpread, t10


#def workerWrapper(procNum, result_dict, fn):
#    '''worker function'''
#    result = fn()
#    result_dict[procNum] = result



#_threadId = 0
#def getThreadId():
#    _threadId += 1
#    return _threadId
#class myThread (threading.Thread):
#   def __init__(self, threadID, name, fn_noargs):
#      threading.Thread.__init__(self)
#      self.threadID = threadID
#      self.name = name
#      self.fn = fn_noargs
#   def run(self):
#      print ("Starting " + self.name)
#      # Get lock to synchronize threads
#      threadLock.acquire()
#      #print_time(self.name, self.counter, 3)
#      fn()
#      # Free lock to release next thread
#      threadLock.release()

#threadLock = threading.Lock()
#threads = []

## Create new threads
#id = getThreadId()
#thread1 = myThread(id, "Thread-{0}".format(id), )
#id = getThreadId()
#thread2 = myThread(id, "Thread-{0}".format(id), 2)

## Add threads to thread list
#threads.append(thread1)
#threads.append(thread2)