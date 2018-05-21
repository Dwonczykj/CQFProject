import math
import numpy as np
import platform

#!Note for paper, that usnig the empirican discrepancy from jeckel, we can see that for dimensions less than 15, we are using 5 dims, low discrpency number generation performs much better than pseudo random number generation.

    # '''Class use: initialise the class Object, then call Generate each time you want a new vector or low discrepancy numbers from that class instance.
    # 1. First for each dimension k of x_0k we must draw DIRECTION INTS. This is done using the Prmitive Polynomial 
    # where the first mdeg DIRECTION INTS are INITIALISED (FREELY) Where ONLY the l leftmost bits of vkl CAN be NON-ZERO ;
    # and the remainging (b - mdeg) DIRECTION INTS are calculated from a BITWISE RECURSION using the primitive polynomial.
    # This recursion (8.19 in jackel) is to first shift vk(l-mdeg) by mdeg bits RIGHT and then XOR'd with some direction ints. 
    # 2. Additionally, for each draw n of xnk, we need is a new GENERATION INT: GRAY CODE or N gamma(n)
    # 3. Then we can use these to calculate each dimension of xnk using the XOR sum of the directionalInts for the kth dim on an indicator function if the jth bit of gamma(n) is set.
    # OR if we use Gray Code, then all we need is 8.23 xnk = x(n-1)k XOR initVjk
    # Note that the same directional ints are used for each of the n steps. So we only have to calculate them once at the beginning.
    # . 
    # params: d is the number of dimensions for each vector of sobol numbers
    # '''
cimport numpy as np
# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.int
FDTYPE = np.float
cdef class SobolNumbers:
    cpdef int initialise(self,int d):
        #Generation of sobol nos is initially carried out on a set of ints in [1,2^b-1], where b is no of bits in UInt in min(program compiler, computer).
        self.b = 30# As Cython compiler is 32bit atm. #62 if platform.architecture()[0] == '64bit' else 30
        self.mdeg = np.asarray([0,1,2,3,3,4,4,5,5,5,5,5,5,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8],dtype=DTYPE)
        self.initP = np.asarray([ \
            1,   3,  7, 11, 13, 19, 25, 37, 59, 47, \
            61, 55, 41, 67, 97, 91, 109, 103, 115, 131, \
            193, 137, 145, 143, 241, 157, 185, 167, 229, 171, \
            213, 191, 253, 203, 211, 239, 247, 285, 369, 299 ],dtype=DTYPE) #these are the coefficients of the primitive polynomial which is respective of the number of dimensions we are dealing with.
        self.fac = 0.0
        self.MAXDIM = 40
        self.Dimensions = d
        self.iGAMMAN = 0
        self.ix = np.zeros(min(self.Dimensions,self.MAXDIM)+1,dtype=DTYPE)
        cdef np.ndarray[DTYPE_t,ndim=1] iV
        # It is very important to type ALL your variables. You do not get any
        # warnings if not, only much slower code (they are implicitly typed as
        # Python objects). 
        cdef int l,k,i,lt
        # for each dim d, the basis of number generation is given by a SET OF direction integers.
        # There is one direction integer for each bit b of the draws binary representation.
        #it is conducive for the following to view all of the dirrection integers as b-wide bit fields.
        iV = np.zeros(self.MAXDIM*(self.b) + 1,dtype=DTYPE)
        cdef DTYPE_t value
        #todo: First row is a row going from dimension 1 to max = 10 representing the vkl for bit 1, the next row l=2 represents bit 2. ie iV[1:10] is row 1, iV[11:20] is row2
        for l in range(1,self.b+1): #!l represents the direction for the lth bit
            for k in range(1,self.MAXDIM + 1):#!k represents the index dimension
                if l <= self.mdeg[k]: #assign  freely for l < gk as in text between 8.18 & 8.19 in Jaeckel
                    lt = self.MAXDIM * (l-1)
                    while iV[k + (lt)] % 2 == 0:
                        value = round(np.random.uniform() << (l-1)) #(8.25) Jaeckel    
                        iV[k + (lt)] = value << (self.b - l) #(8.26) Jaeckel
        self.initV = iV
        self.initDirectionals()
        for i in range(0,500):
            self.Generate()
        return 0    
    cdef int initDirectionals(self):
        cdef int j,k,l
        cdef int i,ipp
        cdef np.ndarray[DTYPE_t,ndim=2] iU = np.zeros(shape=(self.b+1,self.MAXDIM),dtype=DTYPE)
        self.ix = np.asarray([0] + list(np.zeros(self.MAXDIM,dtype=DTYPE)),dtype=DTYPE)
        if self.initV[1] != 1: 
            return -1
        self.fac = 1.0 / (1 << self.b)
        #!Init iU which is the vector of length b
        for j, k in zip(range(1,self.b+1), range(1,self.MAXDIM*(self.b-1)+1,self.MAXDIM)): 
            iU[j] = id(self.initV[k:k+self.MAXDIM]) #iu is an array (length=maxBit+1) of pointers to longs.
                                                    #Hence iu will contain the same values in the same location in memory as iv. 
                                                    #Each memory location iu refererences a pointer to the jth segment of size k of iv. 
                                                    #Therefore we can assign to the values in these segments by using iu or iv. 
                                                    #They are interchangable. We use iu to visualise the array in 2D.

                                                    #the c implementation in numerical recipes grabs the memory address of segments of size 
                                                    #MAXDIM from initV and then ITERATES THE POINTER 1 -> MAXDIM times to bit move each value in initV. 
                                                    #Therefore initv[0] is untouched, in c this would mean that iU[j,6] = iU[j+1,0] as we dont use the 0 value from the pointer. 
                                                    #We can not do this in python so we use initV directly rather than using 2D addresses.
            #we use the last for loop in the method to reassign the values to initV.
        for k in range(0,self.MAXDIM): 
            for j in range(1,self.mdeg[k+1]+1):  
                self.initV[(j-1)*self.MAXDIM+k] <<= (self.b - j)
            for j in range(self.mdeg[k+1]+1,self.b+1):
                ipp = self.initP[k+1]
                i = self.initV[(j-self.mdeg[k+1]-1)*self.MAXDIM+(k+1)]#iU[j - self.mdeg[k+1]][k+1]
                i ^= (i >> self.mdeg[k+1]) # LHOperation in (8.19) Jaeckel
                for l in range(self.mdeg[k+1]-1,0,-1):
                    if ipp & 1: i ^= self.initV[(j-l-1)*self.MAXDIM+(k+1)]#iU[j-l][k+1]
                    ipp >>= 1
                #iU[j][k+1] = i
                self.initV[(j-1)*self.MAXDIM + (k+1)] = i
        #for j, k in zip(range(1,self.b+1), range(1,self.MAXDIM*(self.b-1)+1,self.MAXDIM)): 
            #id(self.initV[k:k+self.MAXDIM])  = iU[j]
        return 0

    cpdef np.ndarray[FDTYPE_t,ndim=1] Generate(self):
        self.iGAMMAN = self.iGAMMAN+1
        cdef int im,j,l,k
        im = self.iGAMMAN#todo: GRAYCODE Here would use n XOR [n/2]
        #!find the index of jth bit from the RIGHT in iGAMMAN which is not set (1)
        j = 0
        for l in range(1,self.b+1):
            j = l
            if not im & 1: break
            im >>= 1
        if j > self.b:
           ValueError("Max number of bits not big enough for j")
        im = (j-1) * self.MAXDIM #!move along initV j-1 sections as jth bit was 0. Then we will add k below to get to the specific dimension that we want.
        cdef np.ndarray[FDTYPE_t,ndim=1] x = np.zeros(min(self.Dimensions,self.MAXDIM),dtype=float)
        for k in range(1,min(self.Dimensions,self.MAXDIM)+1): 
            #!So initV is a vector f all directional numbers for each dimension [v00,v11,v21,...,vd1,v12,...,vd2,...,vdb]
            self.ix[k] ^= self.initV[im + k] #lhpart of XorSum of (8.20) Jaeckel
            #todo fix length of initV
            x[k-1] = self.ix[k]*self.fac
        return x