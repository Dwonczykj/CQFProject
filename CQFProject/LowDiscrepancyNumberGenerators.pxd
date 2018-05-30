import numpy as np
cimport numpy as np
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.int_t DTYPE_t
ctypedef np.float_t FDTYPE_t
cdef class SobolNumbers:
    cdef int b
    cdef np.ndarray mdeg
    cdef np.ndarray initP
    cdef float fac
    cdef int MAXDIM
    cdef int Dimensions
    cdef int iGAMMAN
    cdef np.ndarray ix 
    cdef np.ndarray iV
    cdef np.ndarray initV
    cdef np.ndarray initVt

    cpdef int initialise(self,int d)
    cdef int initDirectionals(self)
    cpdef np.ndarray Generate(self)