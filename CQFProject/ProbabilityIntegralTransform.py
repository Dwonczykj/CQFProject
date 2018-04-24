from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.distributions import norm
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from scipy.stats import genpareto
from Returns import mean, sd

def measure(n):
    "Measurement model, return two coupled measurements."
    m1 = np.random.normal(size=n)
    m2 = np.random.normal(scale=0.5, size=n)
    return m1+m2, m1-m2

def kde_statsmodels_u(x, x_grid, bandwidth=0.2, **kwargs):
    """Univariate Kernel Density Estimation with Statsmodels"""
    kde = KDEUnivariate(x)
    kde.fit(bw=bandwidth, **kwargs)
    return kde.evaluate(x_grid)

def kde_statsmodels_m(x, x_grid, bandwidth=0.2, **kwargs):
    """Multivariate Kernel Density Estimation with Statsmodels"""
    #kde = KDEMultivariate(x, bw=bandwidth * np.ones_like(x),
    #                      var_type='c', **kwargs)
    #! bw = "cv_ml", "cv_ls", "normal_reference", np.array([0.23])
    kde = KDEMultivariate(data=x,
        var_type='c', bw="cv_ml")
    print(kde.bw)
    return kde.pdf(x_grid)

def test():
    # The grid we'll use for plotting
    x_grid = np.linspace(-4.5, 3.5, 1000)

    # Draw points from a bimodal distribution in 1D
    np.random.seed(0)
    x = np.concatenate([norm(-1, 1.).rvs(400),
                        norm(1, 0.3).rvs(100)]).reshape((500,1))
    
    #x = np.random.normal(-1,1.,size=(400,1))

    pdf_true = (0.8 * norm(-1, 1).pdf(x_grid) +
                0.2 * norm(1, 0.3).pdf(x_grid))

    fig, ax = plt.subplots(1, 3, sharey=True,
                       figsize=(13, 3))
    fig.subplots_adjust(wspace=0)
    plt.subplot(1,3,1)

    pdf = kde_statsmodels_u(x, x_grid, bandwidth=0.2)
    plt.plot(x_grid, pdf, color='blue', alpha=0.5, lw=3)
    plt.fill(x_grid, pdf_true, ec='gray', fc='gray', alpha=0.4)
    plt.title("StatsModel-U")
    plt.xlim(-4.5, 3.5)

    plt.subplot(1,3,2)
    #x_grid = np.linspace(-0.5, 0.5, 1000)
    pdf = kde_statsmodels_m(x, x_grid, bandwidth=0.2)
    plt.plot(x_grid, pdf, color='blue', alpha=0.5, lw=3)
    plt.fill(x_grid, pdf_true, ec='gray', fc='gray', alpha=0.4)
    plt.title("StatsModel-M")
    plt.xlim(-4.5, 3.5)

    plt.subplot(1,3,3)

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
    plt.show()
            
    return exceedances

def GenerateEVTKernelSmoothing():
    nobs = 300
    c1 = np.random.standard_t(3,size=(nobs))
    
    x = np.linspace(min(c1), max(c1),100)

    exceedances = list()
    u = 2
    for rvs in c1:
        if abs(rvs) > u:
            exceedances.append(abs(rvs) - u)
    
    fits = genpareto.fit(exceedances)
    plt.hist(np.array(c1), bins=15, normed=True)
    plt.title("Generalised Pareto on Normal Exceedances")

    plt.plot(x, HybridNormalGPDCDF(x,u,mean(c1),sd(c1),fits[0],loc=fits[1],scale=fits[2]), linewidth=2)
    plt.plot(x, HybridNormalGPDPDF(x,u,mean(c1),sd(c1),fits[0],loc=fits[1],scale=fits[2]), linewidth=2)
    plt.legend(["Guess","Fitted_CDF","Fitted_PDF"],loc='best')

    plt.show()

def HybridNormalGPDCDF(xs, u, mu, sigma, shape, loc, scale):
    out = list()
    l = (mu - abs(u - mu))
    h = (mu + abs(u - mu))
    print('u = %.10f,l = %.10f,h = %.10f'%(u,l,h))
    for x in xs:
        if x < l:
            nrm = norm.cdf(l,mu,sigma)
            out.append(nrm*(1-genpareto.cdf(l - x, shape, loc=loc, scale=scale)))
        elif x > (mu + abs(u - mu)):
            nrm = norm.cdf(h,mu,sigma)
            out.append((1 - nrm)*genpareto.cdf(x - h, shape, loc=loc, scale=scale) + nrm)
        else:
            out.append(norm.cdf(x,mu,sigma))
    return out

def HybridNormalGPDPDF(xs, u, mu, sigma, shape, loc, scale):
    out = list()
    for x in xs:
        if x < (mu - abs(u - mu)):
            nrm_p = norm.pdf((mu - abs(u - mu)),mu,sigma)
            nrm_c = norm.cdf((mu - abs(u - mu)),mu,sigma)
            out.append((nrm_p*(1-genpareto.cdf((mu - abs(u - mu)) - x,shape, loc=loc, scale=scale))) + (nrm_c*(1-genpareto.pdf((mu - abs(u - mu)) - x,shape, loc=loc, scale=scale))))
        elif x > (mu + abs(u - mu)):
            nrm_c = norm.cdf((mu + abs(u - mu)),mu,sigma)
            nrm_p = norm.pdf((mu + abs(u - mu)),mu,sigma)
            out.append(((1 - nrm_p)*genpareto.cdf(x-(mu + abs(u - mu)), shape, loc=loc, scale=scale) + (1 - nrm_c)*genpareto.pdf(x-(mu + abs(u - mu)), shape, loc=loc, scale=scale) + nrm_p))
        else:
            out.append(norm.pdf(x,mu,sigma))
    return out