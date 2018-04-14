# cython: profile=True
# cython: linetrace=True

import math
import numpy as np
import platform

#!Note for paper, that usnig the empirican discrepancy from jeckel, we can see that for dimensions less than 15, we are using 5 dims, low discrpency number generation performs much better than pseudo random number generation.

class SobolNumbers(object):
    '''Class use: initialise the class Object, then call Generate each time you want a new vector or low discrepancy numbers from that class instance.
    1. First for each dimension k of x_0k we must draw DIRECTION INTS. This is done using the Prmitive Polynomial 
    where the first mdeg DIRECTION INTS are INITIALISED (FREELY) Where ONLY the l leftmost bits of vkl CAN be NON-ZERO ;
    and the remainging (b - mdeg) DIRECTION INTS are calculated from a BITWISE RECURSION using the primitive polynomial.
    This recursion (8.19 in jackel) is to first shift vk(l-mdeg) by mdeg bits RIGHT and then XOR'd with some direction ints. 
    2. Additionally, for each draw n of xnk, we need is a new GENERATION INT: GRAY CODE or N gamma(n)
    3. Then we can use these to calculate each dimension of xnk using the XOR sum of the directionalInts for the kth dim on an indicator function if the jth bit of gamma(n) is set.
    OR if we use Gray Code, then all we need is 8.23 xnk = x(n-1)k XOR initVjk
    Note that the same directional ints are used for each of the n steps. So we only have to calculate them once at the beginning.
    . 
    params: d is the number of dimensions for each vector of sobol numbers
    '''
    def __init__(self, d):
        self.b = 31#63 if platform.architecture()[0] == '64bit' else 31
        self.mdeg = [0,1,2,3,3,4,4,5,5,5,5,5,5,6,6,6,6,6,6]
        self.initP = [0,1,3,7,11,13,19,25,37,41,47] #these are the coefficients of the primitive polynomial which is respective of the number of dimensions we are dealing with.
        self.fac = 0
        self.MAXDIM = 10
        self.Dimensions = d
        self.iGAMMAN = 0
        #self.initV = [0,1,1,1,1,1,1,3,1,3,3,1,1,5,7,7,3,3,5,15,11,5,15,13,9,11,17]
        self.ix = np.zeros(min(self.Dimensions,self.MAXDIM)+1,dtype=float)
        iV= list(np.zeros(self.MAXDIM*(self.b) + 1,dtype=int))
        #todo: First row is a row going from dimension 1 to max = 10 representing the vkl for bit 1, the next row l=2 represents bit 2. ie iV[1:10] is row 1, iV[11:20] is row2
        for l in range(1,self.b+1): #!l represents the direction for the lth bit
            for k in range(1,self.MAXDIM + 1):#!k represents the index dimension
                if l <= self.mdeg[k]:
                    lt = self.MAXDIM * (l-1)
                    while iV[k + (lt)] % 2 == 0:
                        iV[k + (lt)] = round(np.random.uniform() * math.pow(2,(l-1)))    
        self.initV = iV
        self.initDirectionals()

    def initDirectionals(self):
        iU = np.zeros(shape=(self.b+1,self.MAXDIM),dtype=int)
        self.ix = [0] + list(np.zeros(self.MAXDIM,dtype=int))
        if self.initV[1] != 1: 
            return
        self.fac = 1.0 / (1 << self.b)
        #!Init iU which is the vector of length b
        for j, k in zip(range(1,self.b+1), range(1,self.MAXDIM*(self.b-1)+1,self.MAXDIM)): 
            iU[j] = self.initV[k:k+self.MAXDIM] #todo: this needs to be assigned by memory in the algorithms

        for k in range(0,self.MAXDIM): 
            for j in range(1,self.mdeg[k+1]+1):  
                iU[j][k] <<= (self.b - j)
            for j in range(self.mdeg[k+1]+1,self.b+1):
                ipp = self.initP[k]
                i = iU[j - self.mdeg[k+1]][k]
                i ^= (i >> self.mdeg[k+1])
                for l in range(self.mdeg[k+1]-1,0,-1):
                    if ipp & 1: i ^= iU[j-l][k]
                    ipp >>= 1
                iU[j][k] = i
        for j, k in zip(range(1,self.b+1), range(1,self.MAXDIM*(self.b-1)+1,self.MAXDIM)): 
            self.initV[k:k+self.MAXDIM] = iU[j]
        return


    def Generate(self):
        self.iGAMMAN = self.iGAMMAN+1
        im = self.iGAMMAN#todo: GRAYCODE Here would use n XOR [n/2]
        #!find the index of jth bit from the RIGHT in iGAMMAN which is not set (1)
        j = 0
        for l in range(1,self.b+1):
            if not im & 1: break
            im >>= 1
            j = l
        if j > self.b:
           IndexError("Max number of bits not big enough for j")
        im = (j-1) * self.MAXDIM #!move along initV j-1 sections as jth bit was 0. Then we will add k below to get to the specific dimension that we want.
        x = np.zeros(min(self.Dimensions,self.MAXDIM),dtype=float)
        for k in range(1,min(self.Dimensions,self.MAXDIM)+1): 
            self.ix[k] ^= self.initV[im + k] #!So initV is a vector f all directional numbers for each dimension [v00,v11,v21,...,vd1,v12,...,vd2,...,vdb]
            #todo fix length of initV
            x[k-1] = self.ix[k]*self.fac
        return x

#def SobolNumberGen(n, d, GeneratingIntegerType="GrayCode", initializationType="RegularityBreaking"):
#    #todo: all the variables below need to be setup as class vars on a Sobol class so that we can store them outside of the method... we can then have another static method that inits the class and runs the required methods in succession.
#    b = 63 if platform.architecture()[0] == '64bit' else 31
#    mdeg = [0,1,2,3,3,4,4,5,5,5,5,5,5,6,6,6,6,6,6]
#    initP = [0,1,3,7,11,13,19,25,37,41,47] #these are the coefficients of the primitive polynomial which is respective of the number of dimensions we are dealing with.
#    initV = [0,1,1,1,1,1,1,3,1,3,3,1,1,5,7,7,3,3,5,15,11,5,15,13,9,11,17]
#    #! Jaeckel suggestion is to use a pseudo-random number generator to generate ukl, wkl = int(ukl * 2^(l-1)), vkl = wkl*2^(b-l).
#    rands = np.random.uniform(size=(b - d + 1))
#    iV= [0]
#    for l in range(1,mdeg[d]+1):
#        iV[l] = 0
#        while iV[l] % 2 == 0:
#            iV[l] = rands[l-1] << (l-1)
#        iV[l] <<= (b - l)
#    #!initV = iV
#    iU = []
#    ix = []
#    fac = 0
#    iGAMMAN = 0 #GENERATION INT

#    '''1. First for each dimension k of x_0k we must draw DIRECTION INTS. This is done using the Prmitive Polynomial 
#    where the first mdeg DIRECTION INTS are INITIALISED (FREELY) Where ONLY the l leftmost bits of vkl CAN be NON-ZERO ;
#    and the remainging (b - mdeg) DIRECTION INTS are calculated from a BITWISE RECURSION using the primitive polynomial.
#    This recursion (8.19 in jackel) is to first shift vk(l-mdeg) by mdeg bits RIGHT and then XOR'd with some direction ints. 
#    2. Additionally, for each draw n of xnk, we need is a new GENERATION INT: GRAY CODE or N gamma(n)
#    3. Then we can use these to calculate each dimension of xnk using the XOR sum of the directionalInts for the kth dim on an indicator function if the jth bit of gamma(n) is set.
#    OR if we use Gray Code, then all we need is 8.23 xnk = x(n-1)k XOR initVjk
#    Note that the same directional ints are used for each of the n steps. So we only have to calculate them once at the beginning.
#    . 
#    '''
#    if n < 0:
#        ix = [0] + list(np.zeros(size=d))
#        if initV != 1: 
#            return
#        fac = 1.0 / (1L << b)
#        #!Init iU which is the vector of length b
#        for j, k in zip(range(1,b), range(0,d*b,d)): iU[j] = initV[k][:] #todo: this needs to be assigned by memory in the algorithms

#        for k in range(1,d+1): #start at 1 and skip the 0 draw.
#            # initialise the init values.
#            for j in range(1,mdeg[k]+1):  iU[j][k] <<= (b - j)
#            # use the init values to recurse the other direction values 
#            for j in range(mdeg[k]+1,b+1):
#                ipp = initP[k]
#                i = iU[j - mdeg[k]][k]
#                i ^= (i >> mdeg[k])
#                for l in range(mdeg[k]-1,0,-1):
#                    if ipp & 1: i ^= iU[j-l][k]
#                    ipp >>= 1
#                iU[j][k] = i
#        for j, k in zip(range(1,b), range(0,d*b,d)): initV[k][:] = iU[j]
#        return
#    else:
#        im = iGAMMAN++ #todo: GRAYCODE Here would use n XOR [n/2]
#        #!find the index of jth bit from the RIGHT in iGAMMAN which is not set (1)
#        j = 0
#        for l in range(1,b+1):
#            if not im & 1: break
#            im >>= 1
#            j = l
#        if j > b:
#           IndexError("Max number of bits not big enough for j")
#        im = (j-1) * d #!move along initV j-1 sections as jth bit was 0. Then we will add k below to get to the specific dimension that we want.
#        for k in range(1,min(n,d)): 
#            ix[k] ^= initV[im + k] #!So initV is a vector f all directional numbers for each dimension [v11,v21,...,vd1,v12,...,vd2,...,vdb]
#            x[k] = ix[k]*fac
#        return x
#    #Unit,RegularityBreaking,
#    #n,GrayCode