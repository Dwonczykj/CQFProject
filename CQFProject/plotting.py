import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import MaxNLocator
import numpy as np
import math
import pandas as pd
from CumulativeAverager import CumAverage
from scipy.interpolate import interp1d
from scipy import interpolate
from scipy.stats import norm, genpareto
from Returns import AIC, mean, sd
import os
import time
import multiprocessing as mp
import threading
from EmpiricalFunctions import *
import warnings

if __name__ == '__main__':
    mplock = mp.Lock()  
    ThreadSafe_SubmissionFilePath= os.path.join('C:\\', 'Users', 'Joe.Dwonczyk', 'Documents', 'CQF', 'CVASection', 'Submission') + "\\CDSBasketProj\\" + time.strftime("%Y%m%d-%H%M%S") + "\\"
    s = mp.sharedctypes.Array('c',b"{0}".format(ThreadSafe_SubmissionFilePath),True)
    Array('c', b'hello world', lock=lock)
    mplock.acquire()
    try:
        if not os.path.exists(ThreadSafe_SubmissionFilePath):
            os.makedirs(ThreadSafe_SubmissionFilePath)
    except:
        pass
    finally:
        mplock.release()

class Plotter(object):
    def __init__(self):
        self.mplock = mp.Lock()  
        ThreadSafe_SubmissionFilePath= os.path.join('C:\\', 'Users', 'Joe.Dwonczyk', 'Documents', 'CQF', 'CVASection', 'Submission') + "\\CDSBasketProj\\" + time.strftime("%Y%m%d-%H%M%S") + "\\"
        #s = mp.sharedctypes.Array('c',b"{0}".format(ThreadSafe_SubmissionFilePath),True)
        self.SubmissionFilePath = ThreadSafe_SubmissionFilePath
        self.queue = mp.Queue()
        self.processingPlots = False
        self.mplock.acquire()
        try:
            if not os.path.exists(ThreadSafe_SubmissionFilePath):
                os.makedirs(ThreadSafe_SubmissionFilePath)
        except:
            pass
        finally:
            self.mplock.release()

    def pyplot_memcheck(self,noToAllow=1):
        fignums = plt.get_fignums()
        if len(fignums) > noToAllow:
            self.save_all_figs()

    def save_all_figs(self):
        fignums=plt.get_fignums()
        for i in fignums:
                fig = plt.figure(i)
                name = self.SubmissionFilePath + fig.canvas.get_window_title().replace(" ","_").replace(".",",")
                if not os.path.exists(name+".png"):
                    plt.savefig(name)
                    print("saved file with name: {0}".format(name))
                else:
                    j = 1
                    tname = name
                    while os.path.exists(tname+".png"):
                        tname = name + "_%d"%(j)
                        j +=1
                    plt.savefig(tname)
                    print("saved file with name: {0}".format(tname))
                plt.close(i)

    def QueuePlots(self, fnNameArgsList):
        [self.queue.put(fnNameArgs) for fnNameArgs in fnNameArgsList]
        if not self.processingPlots:
            self.DestinationThread()

    def DestinationThread(self) :
        while True :
            self.processingPlots = True
            try:
                fnArgs = self.queue.get(False)
            except:
                # Handle empty queue here
                self.processingPlots = False
                #print("Plotting Queue is empty")
                break
            else:
                # Handle task here and call q.task_done()
                fnName = fnArgs[0]
                args = fnArgs[1:]
                #print("Plotting function: {0} to location {1}".format(fnName,self.SubmissionFilePath))
                getattr(Plotter, fnName)(self,*args)

    def showAllPlots(self):
        plt.show()

    def lock(self):
        self.mplock.acquire()

    def unlock(self):
        self.mplock.release()

    def plot_DefaultProbs(self,x,y,name,legendArray):
        self.QueuePlots([["_plot_DefaultProbs",x,y,name,legendArray]])


    def plot_codependence_scatters(self,dataDic,xlabel,ylabel="",name=""):
        self.QueuePlots([["_plot_codependence_scatters",dataDic,xlabel,ylabel,name]])

    def return_scatter(self,xdata,ydata,name,numberPlot=1,noOfPlotsW=1, noOfPlotsH=1,xlabel = "", ylabel="frequency/probability",legend=[], xticks=[], yticks=[]):
        ''' Plots a scatter plot showing any co-dependency between 2 variables. '''
        self.QueuePlots([["_return_scatter",xdata,ydata,name,numberPlot,noOfPlotsW, noOfPlotsH,xlabel, ylabel,legend, xticks, yticks]])

    def return_scatter_multdependencies(self,xdata,arrydata,name,numberPlot=1,noOfPlotsW=1, noOfPlotsH=1,xlabel = "", ylabel="frequency/probability",legend=[], xticks=[], yticks=[]):
        ''' Plots a scatter plot showing any co-dependency between 2 variables. '''
        self.QueuePlots([["_return_scatter_multdependencies",xdata,arrydata,name,numberPlot,noOfPlotsW, noOfPlotsH,xlabel, ylabel,legend, xticks, yticks]])

    def Plot_Converging_Averages(self,ArrOfArrays,baseName):
        self.QueuePlots([["_Plot_Converging_Averages",ArrOfArrays,baseName]])
        

    def Plot_Converging_Average(self,data,name,numberPlot=1,noOfPlotsW=1, noOfPlotsH=1):
        ''' Plots a line plot showing the convergence against iterations of the average. '''
        self.QueuePlots([["_Plot_Converging_Average",data,name,numberPlot,noOfPlotsW, noOfPlotsH]])


    def return_Densitys(self,datatable,name,legendArray,noOfLines):
        ''' Plots Normal PDFs on an array of returns. '''
        self.QueuePlots([["_return_Densitys",data,datatable,name,legendArray,noOfLines]])

    def dN(self,x, mu, sigma):
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

    def dExp(self,x, rate):
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

    def QQPlot(self,rv,name):
        ''' QQ plot to check normaility of random variables.
        Parameters
        ==========
        rvs : float[]
            rvs to be tested for normality
        '''
        self.QueuePlots([["_QQPlot",rv,name]])

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

    def cdfGEV(self,x, k=0, sigma=1, theta=0):
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

    def dExpForPiecewiselambda(self,rates, tenors):
        ''' Probability density function of an Exponential random variable x using piecewise constant lambda to display varying rates.
        Parameters
        ==========
        rates : array(float)
            piecewise rates of exponential distn
        =======
        pdf : float
            value of probability density function
        '''
        self.QueuePlots([["_dExpForPiecewiselambda",rates, tenors]])

    # histogram of annualized daily log returns
    def return_histogram(self,data,name,numberPlot=1,noOfPlotsW=1, noOfPlotsH=1,xlabel = "", length = 1, f = None, displayMiddlePercentile=100, figSize = (10,6)):
        ''' Plots a histogram of the returns. '''
        self.QueuePlots([["_return_histogram",data,name,numberPlot,noOfPlotsW, noOfPlotsH,xlabel, length, f, displayMiddlePercentile, figSize]])

    def plot_histogram_array(self,dataDic,xlabel, displayMiddlePercentile=100, outPercentiles=[1, 5, 10, 25, 75, 90, 95, 99],name=""):
        '''
        Pass a dictionary will plot a histogram for the data on each key.
        displayMiddlePercentile allows user to plot middle percent of the distribution only and ignore the tails.
        outPercentiles define the sample percentile values to return for each dataset in the dataDic.
        Returns a dict of {[key]: (mean, sd, dict(percentile, value))}
        '''
        self.QueuePlots([["_plot_histogram_array",dataDic,xlabel, displayMiddlePercentile, outPercentiles,name]])


    def SuitableRegressionFit(self,x,y,name="",numberOfAdditionalPointsToReturn=100,startingPower=0):
        #Use AIC to estimate best power to use as this is a PREDICTIVE MODEL for volaty functions that will be used to simulate future volatilites.
        first_Power = startingPower
        k = 1
        f_test, xL = self._FittedValuesLinear(x,y,"Regression",first_Power,name,numberOfAdditionalPointsToReturn)
        min_AIC = AIC(y, f_test(x),k)
        j = first_Power
        for i in range(first_Power+1,11):
            try_f_test, xLin = self._FittedValuesLinear(x,y,"Regression",i,name,numberOfAdditionalPointsToReturn)
            try_AIC = AIC(y, try_f_test(x),k)
            if(try_AIC < min_AIC):
                f_test = try_f_test
                min_AIC = try_AIC
                j = i
                xL = xLin
        return f_test, j, xL

    def FittedValuesLinear(self,x,y,IsInterpolationOrRegression="Interpolation",PowerOfRegression=3,name="",numberOfAdditionalPointsToReturn=100):
        self.QueuePlots([["_FittedValuesLinear",x,y,IsInterpolationOrRegression,PowerOfRegression,name,numberOfAdditionalPointsToReturn]])

    def return_lineChart(self,x,arrLines,name,numberPlot=1,noOfPlotsW=1, noOfPlotsH=1,xlabel = "",ylabel="",legend=[], xticks=[], yticks=[], trimTrailingZeros=False):
        ''' Plots all lines on the same chart against x.'''
        self.QueuePlots([["_return_lineChart",x,arrLines,name,numberPlot,noOfPlotsW, noOfPlotsH,xlabel,ylabel,legend, xticks, yticks, trimTrailingZeros]])


    def return_lineChart_dates(self,x,arrLines,name,numberPlot=1,noOfPlotsW=1, noOfPlotsH=1,xlabel = "",ylabel="",legend=[], yticks=[]):
        ''' Plots all lines on the same chart against x.'''
        self.QueuePlots([["_return_lineChart_dates",x,arrLines,name,numberPlot,noOfPlotsW, noOfPlotsH,xlabel,ylabel,legend, yticks]])


    def colorPicker(self,i): #https://matplotlib.org/users/colors.html
        colors = ["aqua","green","magenta","navy","red","salmon","sienna","yellow","olive","orange","chartreuse","coral","crimson","cyan","black","brown","darkgreen","fuchsia","gold","grey","khaki","lavender","pink","purple"]
        return colors[(i % len(colors))]

    def return_barchart_old(self,categories,dataDic,name="",xlabel="",ylabel=""):
        self.QueuePlots([["_return_barchart_old",categories,dataDic,name,xlabel,ylabel]])

    def return_barchart(self,categories,dataDic,name="",xlabel="",ylabel="", ScalingAmount=1.0):
        self.QueuePlots([["_return_barchart",categories,dataDic,name,xlabel,ylabel, ScalingAmount]])
        
    def PlotSemiParametricFitResults(self,c1,u,r1c,r2c,r3c,r4c, bwArr_Cdf,r1p,r2p,r3p,r4p, bwArr_pdf, plotvsc1=False,name="Semi-Parametric Fit",xlabel="",ylabel=""):
        '''
        Wrapper to plot the results of SemiParametricCDFFit
        returns void
        '''
        self.QueuePlots([["_PlotSemiParametricFitResults",c1,u,r1c,r2c,r3c,r4c, bwArr_Cdf,r1p,r2p,r3p,r4p, bwArr_pdf,plotvsc1,name,xlabel,ylabel]])

    def _plot_DefaultProbs(self,x,y,name,legendArray):
        self.pyplot_memcheck()
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
    
        plt.subplot(1,1,1)
        plt.xlabel('Equity Volatility')
        plt.ylabel('PD')
        plt.legend(bbox_to_anchor=(0.65, 1), loc=2, borderaxespad=0.,handles=lines)
        #plt.tight_layout()
        #plt.show()


    def _plot_codependence_scatters(self,dataDic,xlabel,ylabel="",name=""):
        self.pyplot_memcheck()
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
                self._return_scatter(dataDic[key1],dataDic[key2],name+" "+"%s vs %s" % (key1,key2),j+1,numCols,numRows,xlabel.replace("%","%s"%(key1)) if "%" in xlabel else xlabel,ylabel.replace("%","%s"%(key2)) if "%" in ylabel else ylabel) if ylabel != "" else return_scatter(dataDic[key1],dataDic[key2],"%s vs %s" % (key1,key2),j+1,numCols,numRows,xlabel.replace("%","%s"%(key1)) if "%" in xlabel else xlabel)
                j += 1
        self.save_all_figs()

    def _return_scatter(self,xdata,ydata,name,numberPlot=1,noOfPlotsW=1, noOfPlotsH=1,xlabel = "", ylabel="frequency/probability",legend=[], xticks=[], yticks=[]):
        ''' Plots a scatter plot showing any co-dependency between 2 variables. '''
        self.pyplot_memcheck(max([numberPlot,noOfPlotsW*noOfPlotsH]))
        if numberPlot==1:
            fig = plt.figure(figsize=(10, 6))
            fig.canvas.set_window_title(name)
            fig.canvas.figure.set_label(name)
        plt.subplot(noOfPlotsH,noOfPlotsW,numberPlot)
        x = np.linspace(min(xdata), max(xdata), 100)
        plt.scatter(x=xdata,y=ydata)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if(len(xticks)>0):
            plt.xticks(xticks)
        if(len(yticks)>0):
            plt.yticks(yticks)
        char_width = int(60/noOfPlotsW)-5
        plt.title('\n'.join([name[i:i+char_width] for i in range(0, len(name), char_width)]))
        if len(legend) > 0:
            plt.legend(legend,loc='best')
        plt.grid(True)

    def _return_scatter_multdependencies(self,xdata,arrydata,name,numberPlot=1,noOfPlotsW=1, noOfPlotsH=1,xlabel = "", ylabel="frequency/probability",legend=[], xticks=[], yticks=[]):
        ''' Plots a scatter plot showing any co-dependency between 2 variables. '''
        self.pyplot_memcheck(max([numberPlot,noOfPlotsW*noOfPlotsH]))
        if numberPlot==1:
            fig = plt.figure(figsize=(10, 6))
            fig.canvas.set_window_title(name)
            fig.canvas.figure.set_label(name)
        plt.subplot(noOfPlotsW,noOfPlotsH,numberPlot)
        x = np.linspace(min(xdata), max(xdata), 100)
        for y in arrydata:
            plt.scatter(x=xdata,y=y)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if(len(xticks)>0):
            plt.xticks(xticks)
        if(len(yticks)>0):
            plt.yticks(yticks)
        char_width = int(60/noOfPlotsW)-5
        plt.title('\n'.join([name[i:i+char_width] for i in range(0, len(name), char_width)]))
        if len(legend) > 0:
            plt.legend(legend,loc='best')
        plt.grid(True)
        if(numberPlot == noOfPlotsH * noOfPlotsW):
            self.save_all_figs()

    def _Plot_Converging_Averages(self,ArrOfArrays,baseName):
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
        self.pyplot_memcheck(ln)
        for j in range(0,ln):
            self._Plot_Converging_Average(CumAverage(np.asarray([item[j] for item in ArrOfArrays],dtype=np.float)),"%s %dth to default" % (baseName,j+1),j+1,numCols,numRows)
        self.save_all_figs()

    def _Plot_Converging_Average(self,data,name,numberPlot=1,noOfPlotsW=1, noOfPlotsH=1):
        ''' Plots a line plot showing the convergence against iterations of the average. '''
        self.pyplot_memcheck(max([numberPlot,noOfPlotsW*noOfPlotsH]))
        if numberPlot==1:
            fig = plt.figure(figsize=(10, 6))
            fig.canvas.set_window_title(name)
            fig.canvas.figure.set_label(name)
        plt.subplot(noOfPlotsW,noOfPlotsH,numberPlot)
        x = np.linspace(0, len(data), 100)
        plt.xlabel("Number of iterations")
        plt.ylabel('Average')
        char_width = int(60/noOfPlotsW)-5
        plt.title('\n'.join([name[i:i+char_width] for i in range(0, len(name), char_width)]))
        plt.grid(True)
        y = self.dN(x, np.mean(data), np.std(data))
        plt.plot(x, y, linewidth=2)


    def _return_Densitys(self,datatable,name,legendArray,noOfLines):
        ''' Plots Normal PDFs on an array of returns. '''
        self.pyplot_memcheck()
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
            y = self.dN(x, np.mean(dti), np.std(dti))
            plt.plot(x, y,color=colorPicker(i,True), linewidth=2)
            lines.append(mlines.Line2D([],[],color=colorPicker(i,True),label=legendArray[i]))
            i += 1
        plt.xlabel('Discounted Payoff')
        plt.ylabel('frequency/probability')
        plt.legend(bbox_to_anchor=(0.65, 1), loc=2, borderaxespad=0.,handles=lines)
        ' \n '.join()
        plt.title(name[:70] + '\n' + name[70:])
        plt.subplot(1,1,1)
        plt.grid(True)
        #plt.tight_layout()#rect=[0, 0, 0.7, 1]

    

    def _QQPlot(self,rv,name):
        ''' QQ plot to check normaility of random variables.
        Parameters
        ==========
        rvs : float[]
            rvs to be tested for normality
        '''
        self.pyplot_memcheck()
        rvs = sorted(rv)
        scaled_rvs = (np.array(rvs)-mean(rvs))/sd(rvs)
        n = len(rvs)
        ntiles = np.arange(1,n+1)/(n+1)
        normLn = norm.ppf(ntiles)
        #measurements = np.random.normal(loc = 20, scale = 5, size=100)
        fig = plt.figure(figsize=(10, 6))
        fig.canvas.set_window_title(name)
        fig.canvas.figure.set_label(name)
        plt.subplot(1,1,1)
        plt.title(name[:70] + '\n' + name[70:])
        plt.plot(normLn, ntiles) 
        plt.plot(scaled_rvs, ntiles)
        plt.legend(['normal',name], loc='best')
        plt.grid(True)
        #stats.probplot(measurements, dist="norm", plot=plt)

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

    

    def _dExpForPiecewiselambda(self,rates, tenors):
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
        #plt.tight_layout()
        return y

    # histogram of annualized daily log returns
    def _return_histogram(self,data,name,numberPlot=1,noOfPlotsW=1, noOfPlotsH=1,xlabel = "", length = 1, f = None, displayMiddlePercentile=100, figSize = (10,6)):
        ''' Plots a histogram of the returns. '''
        self.pyplot_memcheck(max([numberPlot,noOfPlotsW*noOfPlotsH,length]))

        c = int((100 - min(max(0,displayMiddlePercentile),100))/2)
        xMin = np.percentile(data,c)
        xMax = np.percentile(data,100-c)
        x = np.linspace(xMin, xMax, 100)

        noOfBins = 50
        bw=(xMax-xMin)/noOfBins
        name+="_BandWidth_{0}".format(bw)
        if f == None:
            fig = plt.figure(figsize=figSize)
            fig.canvas.set_window_title(name)
            fig.canvas.figure.set_label(name)
        else:
            fig = f

        plt.subplot(noOfPlotsW,noOfPlotsH,numberPlot)
        plt.hist(np.array(data), bins=noOfBins, normed=True)
        lw = 0.5 if length > 8 else 2.0
        y = self.dN(x, np.mean(data), np.std(data))
        plt.plot(x, y, linewidth=lw)
        if min(data) >= 0 and not np.mean(data) == 0:
            w = self.dExp(x, 1 / np.mean(data))
            plt.plot(x, w, linewidth=lw)
    
        plt.xlabel(xlabel)
        plt.ylabel('Frequency')
        char_width = min([int(60/noOfPlotsW)-5,30])
        plt.title('\n'.join([name[i:i+char_width] for i in range(0, len(name), char_width)]))
        plt.grid(True)
        #if numberPlot==length:
        #    #fig.tight_layout()
        #    fig.subplots_adjust(hspace=1.0,
        #                wspace=0.20)
        return fig

    def _plot_histogram_array(self,dataDic,xlabel, displayMiddlePercentile=100, outPercentiles=[1, 5, 10, 25, 75, 90, 95, 99],name=""):
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
        self.pyplot_memcheck(nPlt)
        for j in range(0,nPlt):
            key = keys[j]
            f = self._return_histogram(dataDic[key],name+" "+key,j+1,numRows,numCols,xlabel, nPlt, f, displayMiddlePercentile, figSize)
            result[key] = (mean(dataDic[key]), sd(dataDic[key]), np.fromiter(map(lambda p: np.percentile(dataDic[key],p),outPercentiles),dtype=np.float))
        ##plt.tight_layout()
        plt.subplots_adjust(hspace=1.0,wspace=0.4)
        self.save_all_figs()
        return result


    def _FittedValuesLinear(self,x,y,IsInterpolationOrRegression="Interpolation",PowerOfRegression=3,name="",numberOfAdditionalPointsToReturn=100):
        self.pyplot_memcheck()
        fig = plt.figure(figsize=(10, 6))
        fig.canvas.set_window_title(name)
        fig.canvas.figure.set_label(name)
        plt.subplot(1,1,1)
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

    def _return_lineChart(self,x,arrLines,name,numberPlot=1,noOfPlotsW=1, noOfPlotsH=1,xlabel = "",ylabel="",legend=[], xticks=[], yticks=[], trimTrailingZeros=False):
        ''' Plots all lines on the same chart against x.'''
        self.pyplot_memcheck(max([numberPlot,noOfPlotsW*noOfPlotsH]))
        if numberPlot==1:
            fig = plt.figure(figsize=(10, 6))
            fig.canvas.set_window_title(name)
            fig.canvas.figure.set_label(name)
            if noOfPlotsW*noOfPlotsH>1:
                plt.subplots_adjust(hspace=1.0,wspace=0.4)
        plt.subplot(noOfPlotsW,noOfPlotsH,numberPlot)
        xLin = np.linspace(min(x), max(x), 100)
        #y = dN(x, np.mean(x), np.std(x))
        for y in arrLines:
            if trimTrailingZeros:
                y_trim = np.trim_zeros(y, 'b')
                x_trim = x[0:len(y_trim)]
                plt.plot(x_trim, y_trim, linewidth=2)
            else:
                plt.plot(x, y, linewidth=2)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if(len(xticks)>0):
            plt.xticks(xticks)
        if(len(yticks)>0):
            plt.yticks(yticks)
        char_width = int(60/noOfPlotsW)-5
        plt.title('\n'.join([name[i:i+char_width] for i in range(0, len(name), char_width)]))
        if len(legend) > 0:
            plt.legend(legend,loc='upper right',fontsize="xx-small")
        ##plt.tight_layout()
        plt.grid(True)
        if(numberPlot == noOfPlotsH * noOfPlotsW):
            self.save_all_figs()


    def _return_lineChart_dates(self,x,arrLines,name,numberPlot=1,noOfPlotsW=1, noOfPlotsH=1,xlabel = "",ylabel="",legend=[], yticks=[]):
        ''' Plots all lines on the same chart against x.'''
        self.pyplot_memcheck(max([numberPlot,noOfPlotsW*noOfPlotsH]))
        if numberPlot==1:
            fig = plt.figure(figsize=(10, 6))
            fig.canvas.set_window_title(name)
            fig.canvas.figure.set_label(name)
            if noOfPlotsW*noOfPlotsH>1:
                plt.subplots_adjust(hspace=1.0,wspace=0.4)
        plt.subplot(noOfPlotsW,noOfPlotsH,numberPlot)
        xLin = np.linspace(pd.Timestamp(x[0]).value, pd.Timestamp(x[-1]).value, 100)
        #y = dN(x, np.mean(x), np.std(x))
        for y in arrLines:
            plt.plot(x, y, linewidth=2)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if(len(yticks)>0):
            plt.yticks(yticks)
        char_width = int(60/noOfPlotsW)-5
        plt.title('\n'.join([name[i:i+char_width] for i in range(0, len(name), char_width)]))
        if len(legend) > 0:
            plt.legend(legend, loc='upper right',fontsize="xx-small")
        plt.grid(True)
        if(numberPlot == noOfPlotsH * noOfPlotsW):
            self.save_all_figs()

    

    def _return_barchart_old(self,categories,dataDic,name="",xlabel="",ylabel=""):
        #  create the figure
        self.pyplot_memcheck()
        fig, ax1 = plt.subplots(figsize=(9, 7))
        fig.subplots_adjust(left=0.115, right=0.88)
        fig.canvas.set_window_title(name)
        fig.subplot(1,1,1)
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

        #fig.tight_layout()



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

    def _return_barchart(self,categories,dataDic,name="",xlabel="",ylabel="", ScalingAmount=1.0):

        self.pyplot_memcheck()
        fig = plt.figure(figsize=(10, 6))
        fig.subplots_adjust(left=0.115, right=0.88)
        fig.canvas.set_window_title(name)
        fig.canvas.figure.set_label(name)
        plt.subplot(1,1,1)
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

    def _PlotSemiParametricFitResults(self,c1,u,r1c,r2c,r3c,r4c, bwArr_Cdf,r1p,r2p,r3p,r4p, bwArr_pdf, plotvsc1=False,name="Semi-Parametric Fit",xlabel="",ylabel=""):
        '''
        Wrapper to plot the results of SemiParametricCDFFit
        returns void
        '''
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
            fits = None
            while fits == None:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    try:
                        fits = genpareto.fit(exceedances)
                    except Warning as e:
                        print('error found:', e)
                    warnings.filterwarnings('default')
            internals = np.array(internals).reshape((len(internals),1))
            #c1s = np.array(c1).reshape((len(c1),1))
            #cdf_smoother = kde_statsmodels_m_cdf(internals,x,bandwidth=0.2)
            #pdf_smoother = kde_statsmodels_m_pdf(internals,x,bandwidth=0.2)
            #plt.subplot(2,len(us),i)
            #plt.plot(x, HybridNormalGPDCDF(x,u,mean(c1),sd(c1),fits[0],loc=fits[1],scale=fits[2]), linewidth=2)
            #plt.plot(x, norm.cdf(x,mean(c1),sd(c1)), linewidth=2)
            #plt.plot(x, HybridNormalGPDPDF(x,u,mean(c1),sd(c1),fits[0],loc=fits[1],scale=fits[2]), linewidth=2)
            #plt.plot(x, norm.pdf(x,mean(c1),sd(c1)), linewidth=2)
            #plt.hist(np.array(c1), bins=15, normed=True)
            #plt.xlabel(xlabel)
            #plt.ylabel(ylabel)
            #plt.title("Generalised Pareto Tails on Gaussian Fitted Center")
            #plt.legend(["Fitted_HybridCDF", "Fitted_Normal_CDF", "Fitted_HybridPDF", "Fitted_Normal_PDF", "Data Histogram"],loc='best')
            plt.subplot(2,len(us),i)
        
            emp = pd.Series(r1c).apply(Empirical_StepWise_CDF(sorted(c1)))
            r1s,r2cs = DualSortByL1(r1c,r2c)
            plt.plot(r3c, r4c, linewidth=2)
            plt.plot(r1c, emp, linewidth=2)
            plt.plot(r1s, r2cs, linewidth=2)
            #plt.plot(r1, norm.cdf(r1,mean(c1),sd(c1)), linewidth=2)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title("Semi Parametric CDF with BandWidth {0}".format(bwArr_Cdf).replace('[ ',"list_").replace(']',"_"))
            #plt.legend(["Fitted_HybridCDF", "ECDF Comparison", "CDF_Smoother", "Fitted_NormalCDF", "Fitted_HybridPDF", "PDF_Smoother", "Fitted_Normal_PDF", "Student_T Hist"],loc='best')
            plt.legend(["Fitted_HybridCDF", "ECDF Comparison", "CDF_Smoother"],loc='best')

            plt.subplot(2,len(us),i+1)
    
            r1s,r2ps = DualSortByL1(r1p,r2p)
            plt.plot(r3p, r4p, linewidth=2)
            plt.plot(r1s, r2ps, linewidth=2)
            #plt.plot(r1, norm.pdf(r1,mean(c1),sd(c1)), linewidth=2)
            plt.hist(np.array(c1), bins=15, normed=True)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title("Semi Parametric PDF with BandWidth {0}".format(bwArr_pdf).replace('[ ',"list_").replace(']',"_"))
            plt.legend(["Fitted_HybridPDF", "PDF_Smoother", "Data Histogram"],loc='best')

            result['%.10f'%(u)] = (r2c,r2p)
            i += 3

            plt.subplots_adjust(hspace=0.48)


def DualSortByL1(L1,L2):
    return (list(t) for t in zip(*sorted(zip(L1,L2))))