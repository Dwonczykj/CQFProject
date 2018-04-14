from Copulae import MultVarGaussianCopula, MultVarTDistnCopula
from HazardRates import CreateCDSPVLegsForExactDefault
import operator
import pandas as pd


def SimulateLegPricesFromCorrelationNormal(HistCreditSpreads,TenorCreditSpreads,InvPWCDF,DiscountFactors,ImpHazdRts,LogRtnCorP,NumbGen):
    U_correlatedNorm = MultVarGaussianCopula(LogRtnCorP,NumbGen)    
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
    
    qDataHazards = pd.DataFrame(index=TenorCreditSpreads['Tenor'][0:(5)])
    qDataHazards['DF0_T'] = list(DiscountFactors["6mLibor"].values())[1:]
    #!Is there an issue with using spreads to calculate implied hazards rates/default probs and then using them again to calculate leg payments
    #todo: COnsider the effect of changing the recovery rate or grab the rr from bloomberg?
    for i in range(0,5):
        IndKey_Tenor = OrderedExactDefaultTimesGauss[i][0]
        qDataHazards['Hazards-NonCum'] = ImpHazdRts[IndKey_Tenor]
        CDSLegsN[i+1] = CreateCDSPVLegsForExactDefault(OrderedExactDefaultTimesGauss[i][1],qDataHazards,DataTenorDic[IndKey_Tenor],0.4)
        CDSLegsSumN[i+1] = [sum(CDSLegsN[i+1].CompensationLeg), sum(CDSLegsN[i+1].PremiumLeg)]
    return CDSLegsSumN

def SimulateLegPricesFromCorrelationT(HistCreditSpreads,TenorCreditSpreads,InvPWCDF,DiscountFactors,ImpHazdRts,RankCorP,NumbGen,TransformedHistDataDic):
    U_correlatedT = MultVarTDistnCopula(RankCorP, len(TransformedHistDataDic[HistCreditSpreads.columns[1]]) - 1,NumbGen)
    ExactDefaultTimesT = dict()
    CDSLegsT = dict()
    CDSLegsSumT = dict()

    for i in range(0,5):
        i_TenorData = 5*i
        i_HistData = i + 1
        IndKey_Hist = HistCreditSpreads.columns[i_HistData]
        IndKey_Tenor = TenorCreditSpreads['Ticker'][i_TenorData]
        ExactDefaultTimesT[IndKey_Tenor] = InvPWCDF[IndKey_Tenor](U_correlatedT[i])
    #!Is there an issue with using spreads to calculate implied hazards rates/default probs and then using them again to calculate leg payments
    #todo: COnsider the effect of changing the recovery rate or grab the rr from bloomberg?
    OrderedExactDefaultTimesT= sorted(ExactDefaultTimesT.items(), key=operator.itemgetter(1))
    qDataHazards = pd.DataFrame(index=TenorCreditSpreads['Tenor'][0:(5)])
    qDataHazards['DF0_T'] = list(DiscountFactors["6mLibor"].values())[1:]
    for i in range(0,5):
        IndKey_Tenor = OrderedExactDefaultTimesT[i][0]
        qDataHazards['Hazards-NonCum'] = ImpHazdRts[IndKey_Tenor]
        CDSLegsT[i+1] = CreateCDSPVLegsForExactDefault(OrderedExactDefaultTimesT[i][1],qDataHazards,DataTenorDic[IndKey_Tenor],0.4)
        CDSLegsSumT[i+1] = [sum(CDSLegsT[i+1].CompensationLeg), sum(CDSLegsT[i+1].PremiumLeg)]
    return CDSLegsSumT
