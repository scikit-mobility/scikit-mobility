#!/usr/bin/env python

# ciacia_stats.py
# Filippo Simini
# Created: 20131021

import sys,os
import math
import random
#import mpmath
import operator
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pyl
import itertools
import scipy as sp
from scipy import stats
from scipy import optimize
from scipy.integrate import quad
#import scikits.statsmodels as sm


def segno(x):
    """
    Input:  x, a number
    Return:  1.0  if x>0,
            -1.0  if x<0,
             0.0  if x==0
    """
    if   x  > 0.0: return 1.0
    elif x  < 0.0: return -1.0
    elif x == 0.0: return 0.0


def standard_dev(list):
    ll = len(list)
    m = 1.0 * sum(list)/ll
    return ( sum([(elem-m)**2.0 for elem in list]) / ll )**0.5


def list_check(lista):
    """
    If a list has only one element, return that element. Otherwise return the whole list.
    """
    try:
        e2 = lista[1]
        return lista
    except IndexError:
        return lista[0]


        #______________________________________#
        #                                      #
        #      Probability distributions       #
        #______________________________________#
        #                                      #

def pdf(binsize, input, out='no', normalize=True, include_zeros=False, vmin='NA', vmax='NA',start_from='NA', closing_bin=False):
    """
    Return the probability density function of "input"
    using linear bins of size "binsize"

    Input format: one column of numbers

    Example:
    ---------
      a, m = 0.5, 1.
      data = np.random.pareto(a, 1000) + m
      xy = pdf(10.0, data)
    """
    # Detect input type:
    if input == sys.stdin:
    # the data come form standard input
        d = [ list_check(map(float,l.split())) for l in sys.stdin ]
    #         d = [ float(l) for l in sys.stdin if l.strip() ]
    elif isinstance(input, str):
        # the data come from a file
        d = [ list_check(map(float,l.split())) for l in open(input,'r') ]
    #         d = [ float(w) for w in open(input,'r') if w.split()]
    else:
        # the data are in a list
        try:
            iterator = iter(input)
            d = list(input)
        except TypeError:
            print("The input is not iterable.")

    bin = 1.0*binsize
    d.sort()
    lend = len(d)
    hist = []
    if out != 'no' and out != 'stdout': f = open(out,'wb')

    j = 0
    if not isinstance(start_from, str):
        i = int(start_from / bin)+ 1.0 * segno(start_from)
    else:
        i = int(d[j] / bin) + 1.0 *segno(d[j])

    while True:
        cont = 0
        average = 0.0
        if i>=0: ii = i-1
        else:   ii = i
        # To include the lower end in the previous bin, substitute "<" with "<="
        while d[j] < bin*(ii+1):
            cont += 1.0
            average += 1.0
            j += 1
            if j == lend: break
        if cont > 0 or include_zeros==True:
            if normalize == True and i != 0:
                hist += [[ bin*(ii)+bin/2.0 , average/(lend*bin) ]]
            elif i != 0:
                hist += [[ bin*(ii)+bin/2.0 , average/bin ]]
        if j == lend: break
        i += 1
    if closing_bin:
        # Add the "closing" bin
        hist += [[ hist[-1][0]+bin , 0.0 ]]
    if out == 'stdout':
        for l in hist:
            print("%s %s" % (l[0],l[1]))
    elif out != 'no':
        for l in hist:
            f.write("%s %s\n" % (l[0],l[1]))
        f.close()
    if out == 'no': return hist


def lbpdf(binsize, input, out='no'):
    """
    Return the probability density function of "input"
    using logarithmic bins of size "binsize"

    Input format: one column of numbers

    Example:
    ---------
      a, m = 0.5, 1.
      data = np.random.pareto(a, 1000) + m
      xy = lbpdf(1.5, data)
    """
    # Detect input type:
    if input == sys.stdin:
    # the data come form standard input
        d = [ list_check(map(float,l.split())) for l in sys.stdin ]
    #         d = [ float(l) for l in sys.stdin if l.strip() ]
    elif isinstance(input, str):
        # the data come from a file
        d = [ list_check(map(float,l.split())) for l in open(input,'r') ]
    #         d = [ float(w) for w in open(input,'r') if w.split()]
    else:
        # the data are in a list
        try:
            iterator = iter(input)
            d = list(input)
        except TypeError:
            print("The input is not iterable.")

    bin = 1.0*binsize
    d.sort()
    # The following removes elements too close to zero
    while d[0] < 1e-12:
        del(d[0])
    lend = len(d)
    tot = 0
    hist = []

    j = 0
    i = 1.0
    previous = min(d)

    while True:
        cont = 0
        average = 0.0
        next = previous * bin
        # To include the lower end in the previous bin, substitute "<" with "<="
        while d[j] < next:
            cont += 1.0
            average += 1.0
            j += 1
            if j == lend: break
        if cont > 0:
            hist += [[previous+(next-previous)/2.0 , average/(next-previous)]]
            tot += average
        if j == lend: break
        i += 1
        previous = next

    if out != 'no' and out != 'stdout': f = open(out,'wb')
    if out == 'stdout':
        for x,y in hist:
            print("%s %s" % (x,y/tot))
    elif out != 'no':
        f = open(out,'wb')
        for x,y in hist:
            f.write("%s %s\n" % (x,y/tot))
        f.close()
    if out == 'no': return [[x,y/tot] for x,y in hist]


    #______________________________________#
    #                                      #
    #          Least Squares Fit           #
    #______________________________________#
    #                                      #

class Parameter:
    def __init__(self, value):
        self.value = value

    def set(self, value):
        self.value = value

    def __call__(self):
        return self.value

def LSfit(function, parameters, y, x):
    """
    *** ATTENTION! ***
    *** _x_ and _y_ MUST be NUMPY arrays !!! ***
    *** and use NUMPY FUNCTIONS, e.g. np.exp() and not math.exp() ***

    _function_    ->   Used to calculate the sum of the squares:
                         min   sum( (y - function(x, parameters))**2 )
                       {params}

    _parameters_  ->   List of elements of the Class "Parameter"
    _y_           ->   List of observations:  [ y0, y1, ... ]
    _x_           ->   List of variables:     [ [x0,z0], [x1,z1], ... ]

    Then _function_ must be function of xi=x[0] and zi=x[1]:
        def f(x): return x[0] *  x[1] / mu()

        # Gaussian
            np.exp( -(x-mu())**2.0/sigma()**2.0/2.0)/(2.0*sigma()**2.0*np.pi)**0.5
        # Lognormal
            np.exp( -(np.log(x)-mu())**2.0/sigma()**2.0/2.0)/(2.0*sigma()**2.0*np.pi)**0.5/x

    Example:
        x=[np.random.normal() for i in range(1000)]
        variables,data = map(np.array,zip(*pdf(0.4,x)))

        # giving INITIAL _PARAMETERS_:
        mu     = Parameter(7)
        sigma  = Parameter(3)

        # define your _FUNCTION_:
        def function(x): return np.exp( -(x-mu())**2.0/sigma()**2.0/2.0)/(2.0*sigma()**2.0*np.pi)**0.5

        ######################################################################################
        USA QUESTE FORMULE
        #Gaussian formula
        #np.exp( -(x-mu())**2.0/sigma()**2.0/2.0)/(2.0*np.pi)**0.5/sigma()
        # Lognormal formula
        #np.exp( -(np.log(x)-mu())**2.0/sigma()**2.0/2.0)/(2.0*np.pi)**0.5/x/sigma()
        ######################################################################################


        np.exp( -(x-mu())**2.0/sigma()**2.0/2.0)/(2.0*np.pi)**0.5/sigma()

        # fit! (given that data is an array with the data to fit)
        popt,cov,infodict,mesg,ier,pcov,chi2 = LSfit(function, [mu, sigma], data, variables)
    """
    x = np.array(x)
    y = np.array(y)

    def f(params):
        i = 0
        for p in parameters:
            p.set(params[i])
            i += 1
        return y - function(x)

    p = [param() for param in parameters]
    popt,cov,infodict,mesg,ier = optimize.leastsq(f, p, maxfev=10000, full_output=1) #, warning=True)   #, args=(x, y))

    if (len(y) > len(p)) and cov is not None:
        #s_sq = (f(popt)**2).sum()/(len(y)-len(p))
        s_sq = (infodict['fvec']**2).sum()/(len(y)-len(p))
        pcov = cov * s_sq
    else:
        pcov = float('Inf')

    R2 = 1.0 - (infodict['fvec']**2.0).sum() / standard_dev(y)**2.0 / len(y)

    # Detailed Output: p,cov,infodict,mesg,ier,pcov,R2
    return popt,cov,infodict,mesg,ier,pcov,R2



    #______________________________________#
    #                                      #
    #       Maximum Likelihood Fit         #
    #______________________________________#
    #                                      #

def maximum_likelihood(function, parameters, data, full_output=True, verbose=True):
    """
    function    ->  callable: Distribution from which data are drawn. Args: (parameters, x)
    parameters  ->  np.array: initial parameters
    data        ->  np.array: Data

    Example:

        m=0.5
        v=0.5
        parameters = numpy.array([m,v])

        data = [random.normalvariate(m,v**0.5) for i in range(1000)]

        def function(p,x): return numpy.exp(-(x-p[0])**2.0/2.0/p[1])/(2.0*numpy.pi*p[1])**0.5

        maximum_likelihood(function, parameters, data)


        # # Check that is consistent with Least Squares when "function" is a gaussian:
        # mm=Parameter(0.1)
        # vv=Parameter(0.1)
        # def func(x): return numpy.exp(-(x-mm())**2.0/2.0/vv())/(2.0*numpy.pi*vv())**0.5
        # x,y = zip(*pdf(0.1,data,out='no'))
        # popt,cov,infodict,mesg,ier,pcov,chi2 = LSfit(func, [mm,vv], y, x)
        # popt
        #
        # # And with the exact M-L values:
        # mm = sum(data)/len(data)
        # vv = standard_dev(data)
        # mm, vv**2.0
    """

    def MLprod(p, data, function):
        return -np.sum(np.array([np.log(function(p,x)) for x in data]))

    return optimize.fmin(MLprod, parameters, args=(data,function), full_output=full_output, disp=verbose)

