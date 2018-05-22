import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import MaxNLocator
import numpy as np
import math
import pandas as pd
from CumulativeAverager import CumAverage
from scipy.interpolate import interp1d
from scipy import interpolate
from Returns import AIC, mean, sd
import os
import time

SubmissionFilePath= os.path.join('C:\\', 'Users', 'Joe.Dwonczyk', 'Documents', 'CQF', 'CVASection', 'Submission') + "\\CDSBasketProj\\" + time.strftime("%Y%m%d-%H%M%S") + "\\"
if not os.path.exists(SubmissionFilePath):
    os.makedirs(SubmissionFilePath)

def pyplot_memcheck():
    fignums = plt.get_fignums()
    if len(fignums) > 19:
        save_all_figs()

def save_all_figs():
    fignums=plt.get_fignums()
    for i in fignums:
            fig = plt.figure(i)
            name = SubmissionFilePath + fig.canvas.get_window_title().replace(" ","_").replace(".",",")
            if not os.path.exists(name+".png"):
                plt.savefig(name)
            else:
                j = 1
                tname = name
                while os.path.exists(tname+".png"):
                    tname = name + "_%d"%(j)
                    j +=1
                plt.savefig(tname)
            plt.close(i)

def plot_DefaultProbs(x,y,name,legendArray):
    pyplot_memcheck()
    fig = plt.figure(figsize=(7.5, 4.5))
    fig.canvas.set_window_title(name)
    fig.canvas.figure.set_label(name)
    noOfLines = len(y)
    i = 0
    lines = []
    while i < noOfLines:
        plt.plot(x, y[i], color=colorPicker(i), linewidth=2)
        lines.append(mlines.Line2D([],[],color=colorPicker(i),label=legendArray[i]))
        i += 1
    
    
    plt.xlabel('Equity Volatility')
    plt.ylabel('PD')
    plt.legend(bbox_to_anchor=(0.65, 1), loc=2, borderaxespad=0.,handles=lines)
    plt.tight_layout()
    #plt.show()

def showAllPlots():
    plt.show()

def plot_codependence_scatters(dataDic,xlabel,ylabel=""):
    pyplot_memcheck()
    keys = list(dataDic.keys())
    ln = len(keys)
    nPlt = int((ln * (ln - 1))/2)
    n = nPlt
    #keep on dividing by 2 with no remainder until it is not possible:
    i = 1
    if not( n % 2 == 0) and n > 2:
        n -= 1
    if n > 1:
        while (n % 2 == 0 and n > 1) or n > 5:
            i += 1
            if not n % 2 == 0:
                n -= 1
            n >>= 1   
    
    numCols = int(nPlt / i)
    if nPlt % i > 0:
        numCols += 1
    numRows = i
    j = 0
    for j1 in range(0,ln):
        for j2 in range(j1+1,ln):
            key1 = keys[j1]
            key2 = keys[j2]
            return_scatter(dataDic[key1],dataDic[key2],"%s vs %s" % (key1,key2),j+1,numCols,numRows,xlabel.replace("%","%s"%(key1)) if "%" in xlabel else xlabel,ylabel.replace("%","%s"%(key2)) if "%" in ylabel else ylabel) if ylabel != "" else return_scatter(dataDic[key1],dataDic[key2],"%s vs %s" % (key1,key2),j+1,numCols,numRows,xlabel.replace("%","%s"%(key1)) if "%" in xlabel else xlabel)
            j += 1

def return_scatter(xdata,ydata,name,numberPlot=1,noOfPlotsW=1, noOfPlotsH=1,xlabel = "", ylabel="frequency/probability",legend=[], xticks=[], yticks=[]):
    ''' Plots a scatter plot showing any co-dependency between 2 variables. '''
    pyplot_memcheck()
    if numberPlot==1:
        fig = plt.figure(figsize=(10, 6))
        fig.canvas.set_window_title(name)
        fig.canvas.figure.set_label(name)
    plt.subplot(noOfPlotsH,noOfPlotsW,numberPlot)
    x = np.linspace(min(xdata), max(xdata), 100)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if(len(xticks)>0):
        plt.xticks(xticks)
    if(len(yticks)>0):
        plt.yticks(yticks)
    plt.title(name[:70] + '\n' + name[70:])
    if len(legend) > 0:
        plt.legend(legend,loc='best')
    plt.grid(True)
    plt.scatter(x=xdata,y=ydata)

def return_scatter_multdependencies(xdata,arrydata,name,numberPlot=1,noOfPlotsW=1, noOfPlotsH=1,xlabel = "", ylabel="frequency/probability",legend=[], xticks=[], yticks=[]):
    ''' Plots a scatter plot showing any co-dependency between 2 variables. '''
    pyplot_memcheck()
    if numberPlot==1:
        fig = plt.figure(figsize=(10, 6))
        fig.canvas.set_window_title(name)
        fig.canvas.figure.set_label(name)
    plt.subplot(noOfPlotsW,noOfPlotsH,numberPlot)
    x = np.linspace(min(xdata), max(xdata), 100)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if(len(xticks)>0):
        plt.xticks(xticks)
    if(len(yticks)>0):
        plt.yticks(yticks)
    plt.title(name[:70] + '\n' + name[70:])
    if len(legend) > 0:
        plt.legend(legend,loc='best')
    plt.grid(True)
    for y in arrydata:
        plt.scatter(x=xdata,y=y)

def Plot_Converging_Averages(ArrOfArrays,baseName):
    pyplot_memcheck()
    ln = len(ArrOfArrays[0])
    #keep on dividing by 2 with no remainder until it is not possible:
    i = 0
    if not( ln % 2 == 0) and ln > 2:
        ln -= 1
    while ln % 2 == 0 and ln > 1:
        i += 1
        ln >>= 1
    ln = len(ArrOfArrays[0])
    numCols = int(ln / i)
    if ln % i > 0:
        numCols += 1
    numRows = i
    for j in range(0,ln):
        Plot_Converging_Average(CumAverage(np.asarray([item[j] for item in ArrOfArrays],dtype=np.float)),"%s %dth to default" % (baseName,j+1),j+1,numCols,numRows)

def Plot_Converging_Average(data,name,numberPlot=1,noOfPlotsW=1, noOfPlotsH=1):
    ''' Plots a line plot showing the convergence against iterations of the average. '''
    pyplot_memcheck()
    if numberPlot==1:
        fig = plt.figure(figsize=(10, 6))
        fig.canvas.set_window_title(name)
        fig.canvas.figure.set_label(name)
    plt.subplot(noOfPlotsW,noOfPlotsH,numberPlot)
    x = np.linspace(0, len(data), 100)
    plt.xlabel("Number of iterations")
    plt.ylabel('Average')
    plt.title(name[:70] + '\n' + name[70:])
    plt.grid(True)
    y = dN(x, np.mean(data), np.std(data))
    plt.plot(x, y, linewidth=2)


def return_Densitys(datatable,name,legendArray,noOfLines):
    ''' Plots Normal PDFs on an array of returns. '''
    pyplot_memcheck()
    fig = plt.figure(figsize=(7.5, 4.5))
    fig.canvas.set_window_title(name)
    fig.canvas.figure.set_label(name)
    
    #plt.hist(np.array(data), bins=50, normed=True)
    i = 0
    if type(datatable) is pd.DataFrame:
        dt0 = np.asarray(datatable[0].dropna())
    else:
        dt0 = datatable[0]
    x = np.linspace(min(dt0), max(dt0), 100)
    lines = []
    while i < noOfLines:
        if type(datatable) is pd.DataFrame:
            dti = np.asarray(datatable[i].dropna())
        else:
            dti = datatable[i]
        y = dN(x, np.mean(dti), np.std(dti))
        plt.plot(x, y,color=colorPicker(i,True), linewidth=2)
        lines.append(mlines.Line2D([],[],color=colorPicker(i,True),label=legendArray[i]))
        i += 1
    plt.xlabel('Discounted Payoff')
    plt.ylabel('frequency/probability')
    plt.legend(bbox_to_anchor=(0.65, 1), loc=2, borderaxespad=0.,handles=lines)
    ' \n '.join()
    plt.title(name[:70] + '\n' + name[70:])
    
    plt.grid(True)
    plt.tight_layout()#rect=[0, 0, 0.7, 1]

def dN(x, mu, sigma):
    ''' Probability density function of a normal random variable x.
    Parameters
    ==========
    mu : float
        expected value
    sigma : float
        standard deviation
    Returns
    =======
    pdf : float
        value of probability density function
    '''
    z = (x - mu) / sigma
    pdf = np.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi * sigma ** 2)
    return pdf

def dExp(x, rate):
    ''' Probability density function of an Exponential random variable x.
    Parameters
    ==========
    rate : float
        rate of exponential distn
    =======
    pdf : float
        value of probability density function
    '''
    expL = lambda x: 0 if x < 0 else rate * np.exp(-1 * rate * x)
    return np.fromiter(map(expL,x),dtype=np.float)

def QQPlot(rvs,name):
    ''' QQ plot to check normaility of random variables.
    Parameters
    ==========
    rvs : float[]
        rvs to be tested for normality
    '''
    pyplot_memcheck()
    measurements = np.random.normal(loc = 20, scale = 5, size=100)
    fig = plt.figure(figsize=(10, 6))
    fig.canvas.set_window_title(name)
    fig.canvas.figure.set_label(name)
    plt.title(name[:70] + '\n' + name[70:])
    plt.grid(True)
    stats.probplot(measurements, dist="norm", plot=plt)

#def dGeneralisedPareto(x, k=0, sigma=1, theta=0):
#    ''' Probability density function of an Generalised Pareto random variable x.
#    Parameters
#    ==========
#    k : float
#        shape of Generalised Pareto distn
#        defaults to 0
#    ==========
#    sigma : float
#        scale of Generalised Pareto distn
#        defaults to 1
#    ==========
#    theta : float
#        threshold location parameter of Generalised Pareto distn
#        defaults to 0
#    =======
#    pdf : float
#        value of probability density function
#    '''

def cdfGEV(x, k=0, sigma=1, theta=0):
    ''' Cumulative density function for family of Generalised Extreme Value distribution.
    Parameters
    ==========
    k : float
        shape of Generalised Pareto distn
        defaults to 0
    ==========
    sigma : float
        scale of Generalised Pareto distn
        defaults to 1
    ==========
    theta : float
        threshold location parameter of Generalised Pareto distn
        defaults to 0
    =======
    pdf : float
        value of probability density function
    '''
    cd = 0
    if not k == 0:
        cd = math.exp(-1 * math.pow(1 + (k*(x-theta))/(sigma), (-1)/(k) ) )
    else:
        cd = math.exp(-1 * math.exp( (-1*(x-theta))/(sigma) ) )
    return cd

def dExpForPiecewiselambda(rates, tenors):
    ''' Probability density function of an Exponential random variable x using piecewise constant lambda to display varying rates.
    Parameters
    ==========
    rates : array(float)
        piecewise rates of exponential distn
    =======
    pdf : float
        value of probability density function
    '''
    x = np.linspace(0, max(tenors), 100)
    def rate(s):
        k = 1
        while k <= max(tenors):
            if s <= tenors[k]:
                return rates[k-1]
            k += 1
        return "how is x > than our biggest Tenor"
    expL = lambda x: 0 if x < 0 else rate(x) * np.exp(-1 * rate(x) * x)
    y = np.fromiter(map(expL,x),dtype=np.float)

    fig = plt.figure(figsize=(7.5, 4.5))
    name = "Exponential Distribution for Piecewise constant lambda"
    fig.canvas.set_window_title(name)
    fig.canvas.figure.set_label(name)
    plt.plot(x, y,color=colorPicker(5), linewidth=2)
    lines = [mlines.Line2D([],[],color=colorPicker(5),label="Exponential Distn")]
    plt.xlabel('t')
    plt.ylabel('Density')
    plt.legend(bbox_to_anchor=(0.65, 1), loc=2, borderaxespad=0.,handles=lines)
    plt.tight_layout()
    return y

# histogram of annualized daily log returns
def return_histogram(data,name,numberPlot=1,noOfPlotsW=1, noOfPlotsH=1,xlabel = "", length = 1, f = None, displayMiddlePercentile=100, figSize = (10,6)):
    ''' Plots a histogram of the returns. '''
    pyplot_memcheck()
    if f == None:
        fig = plt.figure(figsize=figSize)
        fig.canvas.set_window_title(name)
        fig.canvas.figure.set_label(name)
    else:
        fig = f

    plt.subplot(noOfPlotsW,noOfPlotsH,numberPlot)
    c = int((100 - min(max(0,displayMiddlePercentile),100))/2)
    x = np.linspace(np.percentile(data,c), np.percentile(data,100-c), 100)
    plt.hist(np.array(data), bins=50, normed=True)
    lw = 0.5 if length > 8 else 2.0
    y = dN(x, np.mean(data), np.std(data))
    plt.plot(x, y, linewidth=lw)
    if min(data) >= 0 and not np.mean(data) == 0:
        w = dExp(x, 1 / np.mean(data))
        plt.plot(x, w, linewidth=lw)
    
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.title(name[:70] + '\n' + name[70:])
    plt.grid(True)
    #if numberPlot==length:
    #    #fig.tight_layout()
    #    fig.subplots_adjust(hspace=1.0,
    #                wspace=0.20)
    return fig

def plot_histogram_array(dataDic,xlabel, displayMiddlePercentile=100, outPercentiles=[1, 5, 10, 25, 75, 90, 95, 99]):
    '''
    Pass a dictionary will plot a histogram for the data on each key.
    displayMiddlePercentile allows user to plot middle percent of the distribution only and ignore the tails.
    outPercentiles define the sample percentile values to return for each dataset in the dataDic.
    Returns a dict of {[key]: (mean, sd, dict(percentile, value))}
    '''
    keys = list(dataDic.keys())
    nPlt = len(keys)
    n = nPlt
    i = 1
    if not( n % 2 == 0) and n > 2:
        n -= 1
    if n > 1:
        while (n % 2 == 0 and n > 1) or n > 5:
            i += 1
            if not n % 2 == 0:
                n -= 1
            n >>= 1   
    
    numCols = int(nPlt / i)
    if nPlt % i > 0:
        numCols += 1
    numRows = i
    f = None
    figSize=(8 if numCols <= 3 else 16,14 if numRows > 3 else 6)
    result = dict()
    for j in range(0,nPlt):
        key = keys[j]
        f = return_histogram(dataDic[key],key,j+1,numCols,numRows,xlabel, nPlt, f, displayMiddlePercentile, figSize)
        result[key] = (mean(dataDic[key]), sd(dataDic[key]), np.fromiter(map(lambda p: np.percentile(dataDic[key],p),outPercentiles),dtype=np.float))
    plt.tight_layout()
    plt.subplots_adjust(hspace=1.0,wspace=0.2)
    return result


def SuitableRegressionFit(x,y,name="",numberOfAdditionalPointsToReturn=100,startingPower=0):
    #Use AIC to estimate best power to use as this is a PREDICTIVE MODEL for volaty functions that will be used to simulate future volatilites.
    first_Power = startingPower
    k = 1
    f_test, xL = FittedValuesLinear(x,y,"Regression",first_Power,name,numberOfAdditionalPointsToReturn)
    min_AIC = AIC(y, f_test(x),k)
    j = first_Power
    for i in range(first_Power+1,11):
        try_f_test, xLin = FittedValuesLinear(x,y,"Regression",i,name,numberOfAdditionalPointsToReturn)
        try_AIC = AIC(y, try_f_test(x),k)
        if(try_AIC < min_AIC):
            f_test = try_f_test
            min_AIC = try_AIC
            j = i
            xL = xLin
    return f_test, j, xL

def FittedValuesLinear(x,y,IsInterpolationOrRegression="Interpolation",PowerOfRegression=3,name="",numberOfAdditionalPointsToReturn=100):
    pyplot_memcheck()
    fig = plt.figure(figsize=(10, 6))
    fig.canvas.set_window_title(name)
    fig.canvas.figure.set_label(name)
    x = np.array(x,dtype=np.float)
    tck = interpolate.splrep(x, y, s=0)
    def poly(p):
        coef = np.polyfit(x,y,p)
        def f(xs):
            g = xs if hasattr(xs,"__iter__") else np.array([xs])
            return np.fromiter(map(lambda xp: sum([c*(xz**ij) for c,xz,ij in zip(coef,np.full(len(coef),xp),np.arange(len(coef)-1,-1,-1))]),g),dtype=np.float)
        return f
    f_cubic = interp1d(x, y, kind='cubic') if IsInterpolationOrRegression == "Interpolation" else poly(3)
    f_cust = poly(PowerOfRegression)
    f_lin = interp1d(x, y) if IsInterpolationOrRegression == "Interpolation" else poly(1)
    xLin = np.linspace(min(x),max(x),endpoint=True,num=numberOfAdditionalPointsToReturn)
    xLin = np.array(sorted(np.append(xLin,x))) #if IsInterpolationOrRegression == "Interpolation" else x
    def f_spline_zerothderivative(xs):
        return interpolate.splev(xs,tck,der=0)
    yLin = f_spline_zerothderivative(xLin)
    plt.plot(x, y, 'o', xLin, f_lin(xLin), ':', xLin, f_cubic(xLin), '--', xLin, yLin, '-.', xLin, f_cust(xLin), '-')
    plt.legend(['data', 'linear', 'cubic', 'spline', 'P(%d) regression'%(PowerOfRegression)], loc='best')
    plt.title(name[:70] + '\n' + name[70:])
    plt.grid(True)
    if IsInterpolationOrRegression=="Interpolation":
        if PowerOfRegression>3:
            return f_spline_zerothderivative, xLin
        else:
            return f_cubic, xLin if PowerOfRegression == 3 else f_lin, xLin
    else:
        return f_cust, xLin

def return_lineChart(x,arrLines,name,numberPlot=1,noOfPlotsW=1, noOfPlotsH=1,xlabel = "",ylabel="",legend=[], xticks=[], yticks=[]):
    ''' Plots all lines on the same chart against x.'''
    pyplot_memcheck()
    if numberPlot==1:
        fig = plt.figure(figsize=(10, 6))
        fig.canvas.set_window_title(name)
        fig.canvas.figure.set_label(name)
    plt.subplot(noOfPlotsW,noOfPlotsH,numberPlot)
    xLin = np.linspace(min(x), max(x), 100)
    #y = dN(x, np.mean(x), np.std(x))
    for y in arrLines:
        plt.plot(x, y, linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if(len(xticks)>0):
        plt.xticks(xticks)
    if(len(yticks)>0):
        plt.yticks(yticks)
    plt.title(name[:70] + '\n' + name[70:])
    if len(legend) > 0:
        plt.legend(legend,loc='best')
    plt.grid(True)


def return_lineChart_dates(x,arrLines,name,numberPlot=1,noOfPlotsW=1, noOfPlotsH=1,xlabel = "",ylabel="",legend=[], yticks=[]):
    ''' Plots all lines on the same chart against x.'''
    pyplot_memcheck()
    if numberPlot==1:
        fig = plt.figure(figsize=(10, 6))
        fig.canvas.set_window_title(name)
        fig.canvas.figure.set_label(name)
    plt.subplot(noOfPlotsW,noOfPlotsH,numberPlot)
    xLin = np.linspace(pd.Timestamp(x[0]).value, pd.Timestamp(x[-1]).value, 100)
    #y = dN(x, np.mean(x), np.std(x))
    for y in arrLines:
        plt.plot(x, y, linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if(len(yticks)>0):
        plt.yticks(yticks)
    plt.title(name[:70] + '\n' + name[70:])
    if len(legend) > 0:
        plt.legend(legend,loc='best')
    plt.grid(True)


def colorPicker(i): #https://matplotlib.org/users/colors.html
    colors = ["aqua","green","magenta","navy","red","salmon","sienna","yellow","olive","orange","chartreuse","coral","crimson","cyan","black","brown","darkgreen","fuchsia","gold","grey","khaki","lavender","pink","purple"]
    return colors[(i % len(colors))]

def return_barchart_old(categories,dataDic,name="",xlabel="",ylabel=""):
    #  create the figure
    pyplot_memcheck()
    fig, ax1 = plt.subplots(figsize=(9, 7))
    fig.subplots_adjust(left=0.115, right=0.88)
    fig.canvas.set_window_title(name)

    pos =  np.arange(len(categories))
    barLengths = list([dataDic[k][0] for k in categories])
    #rects = ax1.bar(pos, height=[0.5]*len(barLengths), width=barLengths, align='center')#,
    #                #color=colorPicker(0), 
    #                #tick_label=categories
    #                #)
    #ax1.set_title(name)

    bar_width = 0.35

    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    #use barh below for horizontal bars.
    noOfRects = len(dataDic[list(dataDic.keys())[0]][:])
    rects = dict()
    colors = ["aqua","green","magenta","navy","red"]
    for i in range(0,noOfRects):
        barLengths = list([dataDic[k][i] for k in categories]).sort(reverse=True)
        color = colorPicker(i*5)
        rects[i] = ax1.bar(pos, height=barLengths,
                          width=[bar_width]*len(barLengths), color=colors[i],   
                           alpha=opacity,label="%d to default" % (i+1))
                           #yerr=std_men, error_kw=error_config,
                           
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(name[:70] + '\n' + name[70:])
    ax1.set_xticks(pos + bar_width / 2)
    ax1.set_xticklabels(list(dataDic.keys()))
    ax1.legend()

    fig.tight_layout()



    #rect_labels = []
    ## Lastly, write in the ranking inside each bar to aid in interpretation
    #for rect in rects:
    #    # Rectangle widths are already integer-valued but are floating
    #    # type, so it helps to remove the trailing decimal point and 0 by
    #    # converting width to int type
    #    width = int(rect.get_width())

    #    rankStr = attach_ordinal(width)
    #    # The bars aren't wide enough to print the ranking inside
    #    if (width < 5):
    #        # Shift the text to the right side of the right edge
    #        xloc = width + 1
    #        # Black against white background
    #        clr = 'black'
    #        align = 'left'
    #    else:
    #        # Shift the text to the left side of the right edge
    #        xloc = 0.98*width
    #        # White on magenta
    #        clr = 'white'
    #        align = 'right'

    #    # Center the text vertically in the bar
    #    yloc = rect.get_y() + rect.get_height()/2.0
    #    label = ax1.text(xloc, yloc, rankStr, horizontalalignment=align,
    #                     verticalalignment='center', color=clr, weight='bold',
    #                     clip_on=True)
    #    rect_labels.append(label)

    return {'fig': fig,
            'ax': ax1,
            #'ax_right': ax2,
            'bars': rects,
            'perc_labels': categories,
            #'cohort_label': cohort_label
            }
    #https://matplotlib.org/gallery/statistics/barchart_demo.html

def return_barchart(categories,dataDic,name="",xlabel="",ylabel="", ScalingAmount=1.0):

    pyplot_memcheck()
    fig = plt.figure(figsize=(10, 6))
    fig.subplots_adjust(left=0.115, right=0.88)
    fig.canvas.set_window_title(name)
    fig.canvas.figure.set_label(name)

    columns = tuple(categories.values)
    rows = ['%d to default' % (x+1) for x in np.arange(len(dataDic[list(dataDic.keys())[0]][:]))] 

    data = list([dataDic[k][:] for k in categories])

    # Get some pastel shades for the colors
    colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
    n_rows = len(data)

    index = np.arange(len(columns)) + 0.3
    bar_width = 0.4

    # Initialize the vertical-offset for the stacked bar chart.
    y_offset = np.zeros(len(columns))

    # Plot bars and create text labels for the table
    cell_text = []
    for row in range(n_rows):
        plt.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])
        y_offset = y_offset + data[row]
        cell_text.append(['%1.10f' % (x/ScalingAmount) for x in y_offset])

    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=cell_text,
                          rowLabels=rows,
                          rowColours=colors,
                          colLabels=columns,
                          loc='bottom')

    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.2, bottom=0.2)

    plt.ylabel(ylabel)
    #plt.yticks(values * value_increment, ['%d' % val for val in values])
    plt.xticks([])
    plt.title(name[:70] + '\n' + name[70:])

    #https://matplotlib.org/gallery/misc/table_demo.html#sphx-glr-gallery-misc-table-demo-py