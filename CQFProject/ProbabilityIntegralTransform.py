from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats.distributions import norm
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from scipy.stats import genpareto
from Returns import mean, sd
from bisect import bisect_left
from EmpiricalFunctions import *

def measure(n):
    "Measurement model, return two coupled measurements."
    m1 = np.random.normal(size=n)
    m2 = np.random.normal(scale=0.5, size=n)
    return m1+m2, m1-m2

def takeClosest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
       return after
    else:
       return before

def takeIndexOfClosest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest index of value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
       return after
    else:
       return before

def kde_statsmodels_u(x, x_grid, bandwidth=0.2, **kwargs):
    """Univariate Kernel Density Estimation with Statsmodels"""
    kde = KDEUnivariate(x)
    kde.fit(bw=bandwidth, **kwargs)
    return kde.evaluate(x_grid)

def kde_statsmodels_m_pdf(x, x_grid, bandwidth=0.2, **kwargs):
    """Multivariate Kernel Density Estimation with Statsmodels"""
    #kde = KDEMultivariate(x, bw=bandwidth * np.ones_like(x),
    #                      var_type='c', **kwargs)
    #! bw = "cv_ml", "cv_ls", "normal_reference", np.array([0.23])    
    pdf = kde_statsmodels_m_pdf_output(x, x_grid, bandwidth, **kwargs)
    x_grid_sorted = sorted(x_grid)
    def sub(x):
        def f(t):
            ind = int(takeIndexOfClosest(x_grid_sorted,t))
            if pdf[ind] <= t:
                return ((pdf[ind+1] - pdf[ind])/(x_grid_sorted[ind + 1] - x_grid_sorted[ind]))*(t - x_grid_sorted[ind]) if ind < len(pdf) else 0
            else:
                return ((pdf[ind] - pdf[ind-1])/(x_grid_sorted[ind] - x_grid_sorted[ind-1]))*(t-x_grid_sorted[ind-1]) if ind > 0 else 0
        m = np.fromiter(map(f,list(x)),dtype=np.float)
        return m[0] if len(m) == 1 else m
    return sub

def kde_statsmodels_m_pdf_output(x, x_grid, bandwidth=0.2, **kwargs):
    """Multivariate Kernel Density Estimation with Statsmodels"""
    #kde = KDEMultivariate(x, bw=bandwidth * np.ones_like(x),
    #                      var_type='c', **kwargs)
    #! bw = "cv_ml", "cv_ls", "normal_reference", np.array([0.23])
    kde = KDEMultivariate(data=x,
        var_type='c', bw="cv_ml")
    #print(kde.bw)
    x_grid_sorted = sorted(x_grid)
    pdf = kde.pdf(x_grid_sorted)
    return pdf

def kde_statsmodels_m_cdf(x, x_grid, bandwidth=0.2, **kwargs):
    """Multivariate Kernel Cumulative Density Estimation with Statsmodels"""
    #kde = KDEMultivariate(x, bw=bandwidth * np.ones_like(x),
    #                      var_type='c', **kwargs)
    #! bw = "cv_ml", "cv_ls", "normal_reference", np.array([0.23])
    cdf = kde_statsmodels_m_cdf_output(x, x_grid, bandwidth, **kwargs)
    x_grid_sorted = sorted(x_grid)
    def sub(x):
        def f(t):
            ind = int(takeIndexOfClosest(x_grid_sorted,t))
            if cdf[ind] <= t:
                return ((cdf[ind+1] - cdf[ind])/(x_grid_sorted[ind + 1] - x_grid_sorted[ind]))*(t - x_grid_sorted[ind]) if ind < len(cdf) else 0
            else:
                return ((cdf[ind] - cdf[ind-1])/(x_grid_sorted[ind] - x_grid_sorted[ind-1]))*(t-x_grid_sorted[ind-1]) if ind > 0 else 0
        m = np.fromiter(map(f,list(x)),dtype=np.float)
        return m[0] if len(m) == 1 else m
    return sub

def kde_statsmodels_m_cdf_output(x, x_grid, bandwidth=0.2, **kwargs):
    """Multivariate Kernel Cumulative Density Estimation with Statsmodels"""
    #kde = KDEMultivariate(x, bw=bandwidth * np.ones_like(x),
    #                      var_type='c', **kwargs)
    #! bw = "cv_ml", "cv_ls", "normal_reference", np.array([0.23])
    kde = KDEMultivariate(data=x,
        var_type='c', bw="cv_ml")
    #print(kde.bw)
    x_grid_sorted = sorted(x_grid)
    cdf = kde.cdf(x_grid_sorted)
    return cdf

def test():
    # The grid we'll use for plotting
    x_grid = np.linspace(-4.5, 3.5, 1000)

    # Draw points from a bimodal distribution in 1D to fit a Kernel Smoother
    np.random.seed(0)
    x = np.concatenate([norm(-1, 1.).rvs(400),
                        norm(1, 0.3).rvs(100)]).reshape((500,1))
    
    #Compare Kernel Smoother to the follwoing curve
    pdf_true = (0.8 * norm(-1, 1).pdf(x_grid) +
                0.2 * norm(1, 0.3).pdf(x_grid))

    fig, ax = plt.subplots(1, 4, sharey=True,
                       figsize=(13, 3))
    fig.subplots_adjust(wspace=0)
    plt.subplot(1,4,1)

    pdf = kde_statsmodels_u(x, x_grid, bandwidth=0.2)
    plt.plot(x_grid, pdf, color='blue', alpha=0.5, lw=3)
    plt.fill(x_grid, pdf_true, ec='gray', fc='gray', alpha=0.4)
    plt.title("StatsModel-U")
    plt.xlim(-4.5, 3.5)

    plt.subplot(1,4,2)
    pdf = kde_statsmodels_m_pdf_output(x,x_grid, bandwidth=0.2)
    plt.plot(x_grid, pdf, color='blue', alpha=0.5, lw=3)
    plt.fill(x_grid, pdf_true, ec='gray', fc='gray', alpha=0.4)
    plt.title("StatsModel-M")
    plt.xlim(-4.5, 3.5)

    cdf_true = (0.8 * norm(-1, 1).cdf(x_grid) +
                0.2 * norm(1, 0.3).cdf(x_grid))
    cdf = kde_statsmodels_m_cdf_output(x, x_grid, bandwidth=0.2)
    plt.subplot(1,4,3)
    plt.plot(x_grid, cdf, color='red', alpha=0.5, lw=3)
    plt.plot(x_grid, norm(-1,0.5).cdf(x_grid), color='green', alpha=0.5, lw=3)
    plt.fill(x_grid, cdf_true, ec='gray', fc='gray', alpha=0.4)
    plt.title("StatsModel-M_CDF")
    plt.xlim(-4.5, 3.5)

    plt.subplot(1,4,4)
    nobs = 300
    np.random.seed(1234)  # Seed random generator
    c1 = np.random.normal(size=(nobs,1))
    c2 = np.random.normal(2, 1, size=(nobs,1))
    dens_u = KDEMultivariate(data=c1,
        var_type='c', bw='normal_reference')
    pdf = dens_u.pdf(x_grid)
    plt.plot(x_grid, pdf, color='blue', alpha=0.5, lw=3)
    #plt.fill(x_grid, pdf_true, ec='gray', fc='gray', alpha=0.4)
    plt.title("StatsModel-M")
    plt.xlim(-4.5, 3.5)
    plt.show()



def GenerateExceedances():
    nobs = 300
    c1 = np.random.standard_t(3,size=(nobs))
    exceedances = list()
    for rvs in c1:
        if rvs > 2:
            exceedances.append(rvs - 2)
    c=0.1
    rv = genpareto(c)
    x = np.linspace(min(exceedances), max(exceedances),100)
    fits = genpareto.fit(exceedances)
    frozen_rv_fitted = genpareto(fits)
    plt.hist(np.array(exceedances), bins=10, normed=True)
    y =  rv.pdf(x)#dN(x, np.mean(data), np.std(data))
    plt.title("Generalised Pareto on Normal Exceedances")
    plt.plot(x, y, linewidth=2)
    plt.plot(x, genpareto.pdf(x,fits[0],loc=fits[1],scale=fits[2]), linewidth=2)
    plt.legend(["Guess","Fitted"],loc='best')
    
            
    return exceedances

def displayGPDPDF():
    c=0.1
    x = np.linspace(genpareto.ppf(0.01, c),
                genpareto.ppf(0.99, c), 100)

    fig, ax = plt.subplots(1, 2, sharey=True,
                       figsize=(7, 7))
    fig.subplots_adjust(wspace=0)
    plt.subplot(1,2,1)

    plt.plot(x, genpareto.pdf(x, c),
       'r-', lw=5, alpha=0.6, label='genpareto pdf')

    

    plt.subplot(1,2,2)
    x = np.linspace(-genpareto.ppf(0.99, c),-genpareto.ppf(0.01, c),100)
    plt.plot(x, genpareto.pdf(-x, c),
       'r-', lw=5, alpha=0.6, label='genpareto pdf')

    plt.show()


def GenerateEVTKernelSmoothing():
    nobs = 3000
    c1 = np.random.standard_t(3,size=(nobs))
    
    x = np.linspace(min(c1), max(c1),1000)

    
    us = [norm.ppf(0.975)]
    fig, ax = plt.subplots(2, len(us), sharey=True,
                           figsize=(7, 7*len(us)))
    fig.subplots_adjust(wspace=0)
    i = 1
    for u in us:
        exceedances = list()
        internals = list()
        for rvs in c1:
            if abs(rvs) > u:
                exceedances.append(abs(rvs) - u)
            else:
                internals.append(rvs)
    
        fits = genpareto.fit(exceedances)
        internals = np.array(internals).reshape((len(internals),1))
        #c1s = np.array(c1).reshape((len(c1),1))
        #cdf_smoother = kde_statsmodels_m_cdf(internals,x,bandwidth=0.2)
        #pdf_smoother = kde_statsmodels_m_pdf(internals,x,bandwidth=0.2)
        plt.subplot(2,len(us),i)
        plt.plot(x, HybridNormalGPDCDF(x,u,mean(c1),sd(c1),fits[0],loc=fits[1],scale=fits[2]), linewidth=2)
        plt.plot(x, norm.cdf(x,mean(c1),sd(c1)), linewidth=2)
        plt.plot(x, HybridNormalGPDPDF(x,u,mean(c1),sd(c1),fits[0],loc=fits[1],scale=fits[2]), linewidth=2)
        plt.plot(x, norm.pdf(x,mean(c1),sd(c1)), linewidth=2)
        plt.hist(np.array(c1), bins=15, normed=True)
        plt.title("Generalised Pareto on Normal Exceedances")
        plt.legend(["Fitted_HybridCDF", "Fitted_NormalCDF", "Fitted_HybridPDF", "Fitted_Normal_CDF", "Student_T Hist"],loc='best')

        plt.subplot(2,len(us),i+1)
        r1,r2,r3,r4 = HybridSemiParametricGPDCDF(x,u,c1,fits[0],loc=fits[1],scale=fits[2])
        plt.plot(r1, r2, linewidth=2)
        plt.plot(r3, r4, linewidth=2)
        plt.plot(r1, norm.cdf(r1,mean(c1),sd(c1)), linewidth=2)
        r1,r2,r3,r4 = HybridSemiParametricGPDPDF(x,u,c1,fits[0],loc=fits[1],scale=fits[2])
        plt.plot(r1, r2, linewidth=2)
        plt.plot(r3, r4, linewidth=2)
        plt.plot(r1, norm.pdf(r1,mean(c1),sd(c1)), linewidth=2)
        plt.hist(np.array(c1), bins=15, normed=True)
        plt.title("Generalised Pareto on Normal Exceedances")
        plt.legend(["Fitted_HybridCDF", "CDF_Smoother", "Fitted_NormalCDF", "Fitted_HybridPDF", "PDF_Smoother", "Fitted_Normal_CDF", "Student_T Hist"],loc='best')
        i += 2

    plt.show()
    return 0

def SemiParametricCDFFit(c1,u,plotvsc1=False,name="Semi-Parametric Fit",xlabel="",ylabel=""):
    '''
    Calculates a SemiParametric fit to the data in c1.
    Uses a gaussian kernal estimation within the centre of the distribution of c1 which is decided by the threshold u.
    Uses a Generalised Pareto distribution to fit both tails outside of the threshold governed by u.
    Returns a tuple containing the the range (y points) of the (SemiPara-CDF,SemiPara-PDF); 
    if (plotvsc1 = False) => the y points depend on 1000 equally spaced points between min(c1) and max(c1).
    if (plotvsc1 = True) => the y points depend on the points in c1 and maintain the order of c1 in the outputted array. i.e. F_n(c1) where F_n is the semiparametric fitted function.
    '''
    'https://mglerner.github.io/posts/histograms-and-kernel-density-estimation-kde-2.html?p=28'
    x = np.linspace(min(c1), max(c1),1000) if plotvsc1 == False else c1

    
    us = list([u])
    fig, ax = plt.subplots(3, len(us), sharey=True,
                           figsize=(7, 7*len(us)))
    fig.subplots_adjust(wspace=0)
    fig.canvas.set_window_title(name)
    fig.canvas.figure.set_label(name)
    result = dict()
    i = 1
    for u in us:
        exceedances = list()
        internals = list()
        for rvs in c1:
            if abs(rvs) > u:
                exceedances.append(abs(rvs) - u)
            else:
                internals.append(rvs)
    
        fits = genpareto.fit(exceedances)
        internals = np.array(internals).reshape((len(internals),1))
        #c1s = np.array(c1).reshape((len(c1),1))
        #cdf_smoother = kde_statsmodels_m_cdf(internals,x,bandwidth=0.2)
        #pdf_smoother = kde_statsmodels_m_pdf(internals,x,bandwidth=0.2)
        plt.subplot(3,len(us),i)
        plt.plot(x, HybridNormalGPDCDF(x,u,mean(c1),sd(c1),fits[0],loc=fits[1],scale=fits[2]), linewidth=2)
        plt.plot(x, norm.cdf(x,mean(c1),sd(c1)), linewidth=2)
        plt.plot(x, HybridNormalGPDPDF(x,u,mean(c1),sd(c1),fits[0],loc=fits[1],scale=fits[2]), linewidth=2)
        plt.plot(x, norm.pdf(x,mean(c1),sd(c1)), linewidth=2)
        plt.hist(np.array(c1), bins=15, normed=True)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title("Generalised Pareto Tails on Gaussian Fitted Center")
        plt.legend(["Fitted_HybridCDF", "Fitted_Normal_CDF", "Fitted_HybridPDF", "Fitted_Normal_PDF", "Data Histogram"],loc='best')

        plt.subplot(3,len(us),i+1)
        r1,r2c,r3,r4 = HybridSemiParametricGPDCDF(x,u,c1,fits[0],loc=fits[1],scale=fits[2])
        emp = pd.Series(r1).apply(Empirical_StepWise_CDF(sorted(c1)))
        r1s,r2cs = DualSortByL1(r1,r2c)
        plt.plot(r3, r4, linewidth=2)
        plt.plot(r1, emp, linewidth=2)
        plt.plot(r1s, r2cs, linewidth=2)
        #plt.plot(r1, norm.cdf(r1,mean(c1),sd(c1)), linewidth=2)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title("Semi Parametric CDF")
        #plt.legend(["Fitted_HybridCDF", "ECDF Comparison", "CDF_Smoother", "Fitted_NormalCDF", "Fitted_HybridPDF", "PDF_Smoother", "Fitted_Normal_PDF", "Student_T Hist"],loc='best')
        plt.legend(["Fitted_HybridCDF", "ECDF Comparison", "CDF_Smoother"],loc='best')

        plt.subplot(3,len(us),i+2)
        r1,r2p,r3,r4 = HybridSemiParametricGPDPDF(x,u,c1,fits[0],loc=fits[1],scale=fits[2])
        r1s,r2ps = DualSortByL1(r1,r2p)
        plt.plot(r3, r4, linewidth=2)
        plt.plot(r1s, r2ps, linewidth=2)
        #plt.plot(r1, norm.pdf(r1,mean(c1),sd(c1)), linewidth=2)
        plt.hist(np.array(c1), bins=15, normed=True)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title("Semi Parametric PDF")
        plt.legend(["Fitted_HybridPDF", "PDF_Smoother", "Data Histogram"],loc='best')

        result['%.10f'%(u)] = (r2c,r2p)
        i += 3

        plt.subplots_adjust(hspace=0.48)
    return result

def DualSortByL1(L1,L2):
    return (list(t) for t in zip(*sorted(zip(L1,L2))))

def HybridNormalGPDCDF(xs, u, mu, sigma, shape, loc, scale):
    '''
    Params: 
        xs: unsorted list of datat to fit semi-parametric CDF to.
        u: threshold to move from Gaussian CDF Fit in center to GPD tail fitting.
        mu:  mean of the data.
        sigma: standard deviation of the data.
        shape: gpd least squares estimated shape parameter.
        loc: gpd least squares estimated location parameter.
        scale: gpd least squares estimated scale parameter.
    Returns:
        an array that would result from xs.apply(semiparametric_fittedfunction) or F_n(xs) where F_n is the CDF fit.
    '''
    out = list()
    l = (mu - abs(u - mu))
    h = (mu + abs(u - mu))
    #print('u = %.10f,l = %.10f,h = %.10f'%(u,l,h))
    for x in xs:
        if x < l:
            nrm = norm.cdf(l,mu,sigma)
            out.append(nrm*(1-genpareto.cdf(l - x, shape, loc=loc, scale=scale)))
        elif x >= h:
            nrm = norm.cdf(h,mu,sigma)
            out.append((1 - nrm)*genpareto.cdf(x - h, shape, loc=loc, scale=scale) + nrm)
        else:
            out.append(norm.cdf(x,mu,sigma))
    return out

def HybridNormalGPDPDF(xs, u, mu, sigma, shape, loc, scale):
    '''
    Params: 
        xs: unsorted list of datat to fit semi-parametric PDF to.
        u: threshold to move from Gaussian PDF Fit in center to GPD tail fitting.
        mu:  mean of the data.
        sigma: standard deviation of the data.
        shape: gpd least squares estimated shape parameter.
        loc: gpd least squares estimated location parameter.
        scale: gpd least squares estimated scale parameter.
    Returns:
        an array that would result from xs.apply(semiparametric_fittedfunction) or F_n(xs) where F_n is the PDF fit.
    '''
    out = list()
    l = (mu - abs(u - mu))
    h = (mu + abs(u - mu))
    #print('u = %.10f,l = %.10f,h = %.10f'%(u,l,h))
    for x in xs:
        if x < l:
            out.append(norm.cdf(l,mu,sigma)*genpareto.pdf(l-x,shape, loc=loc, scale=scale))
        elif x >= h:
            out.append((1 - norm.cdf(h,mu,sigma))*genpareto.pdf(x-h, shape, loc=loc, scale=scale))
        else:
            out.append(norm.pdf(x,mu,sigma))
    return out

def HybridSemiParametricGPDCDF(xs, u, ydata, shape, loc, scale):
    '''
    Params: 
        xs: unsorted list of datat to fit semi-parametric CDF to.
        u: threshold to move from Gaussian Kernel estimation to GPD tail fitting.
        mu:  mean of the data.
        sigma: standard deviation of the data.
        shape: gpd least squares estimated shape parameter.
        loc: gpd least squares estimated location parameter.
        scale: gpd least squares estimated scale parameter.
    Returns:
        an array that would result from xs.apply(semiparametric_fittedfunction) or F_n(xs) where F_n is the CDF fit.
    '''
    #print("Starting Canonical Maximum Likelihood")
    out = list()
    mu = mean(ydata)
    l = (mu - abs(u - mu))
    h = (mu + abs(u - mu))
    #print('u = %.10f,l = %.10f,h = %.10f'%(u,l,h))
    srtdxs = sorted(list(xs)+[l,h])
    cdf_smoother = kde_statsmodels_m_cdf_output(ydata,srtdxs,bandwidth=0.2)
    d = dict(zip(srtdxs,cdf_smoother))
    
    for x in xs:
        if x < l:
            nrm = d[l]
            out.append(nrm*(1-genpareto.cdf(l - x, shape, loc=loc, scale=scale)))
        elif x >= h:
            nrm = d[h]
            out.append((1 - nrm)*genpareto.cdf(x - h, shape, loc=loc, scale=scale) + nrm)
        else:
            out.append(d[x])
    return xs,out,srtdxs,cdf_smoother

def HybridSemiParametricGPDPDF(xs, u, ydata, shape, loc, scale):
    '''
    Params: 
        xs: unsorted list of datat to fit semi-parametric PDF to.
        u: threshold to move from Gaussian Kernel estimation to GPD tail fitting.
        mu:  mean of the data.
        sigma: standard deviation of the data.
        shape: gpd least squares estimated shape parameter.
        loc: gpd least squares estimated location parameter.
        scale: gpd least squares estimated scale parameter.
    Returns:
        an array that would result from xs.apply(semiparametric_fittedfunction) or F_n(xs) where F_n is the PDF fit.
    '''
    out = list()
    mu = mean(ydata)
    l = (mu - abs(u - mu))
    h = (mu + abs(u - mu))
    #print('u = %.10f,l = %.10f,h = %.10f'%(u,l,h))
    srtdxs = sorted(list(xs)+[l,h])
    cdf_smoother = kde_statsmodels_m_cdf_output(ydata,srtdxs,bandwidth=0.2)
    d_cdf = dict(zip(srtdxs,cdf_smoother))
    pdf_smoother = kde_statsmodels_m_pdf_output(ydata,srtdxs,bandwidth=0.2)
    d_pdf = dict(zip(srtdxs,pdf_smoother))
    for x in xs:
        if x < l:
            out.append(d_cdf[l]*genpareto.pdf(l-x,shape, loc=loc, scale=scale))
        elif x >= h:
            out.append((1 - d_cdf[h])*genpareto.pdf(x-h, shape, loc=loc, scale=scale))
        else:
            out.append(d_pdf[x])
    return xs,out,srtdxs,pdf_smoother

#def NonParametricKernelSmoother():
