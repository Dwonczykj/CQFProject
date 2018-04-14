import numpy as np
from LowDiscrepancyNumberGenerators cimport SobolNumbers

cimport numpy as np
ctypedef np.float_t FDTYPE_t

cpdef np.ndarray UnifFromGaussCopula(np.ndarray LogRtnCorP, SobolNumbers NumbGen, int noIterations)

cpdef np.ndarray UnifFromTCopula(np.ndarray RankCorP, SobolNumbers NumbGen, int SeriesLength, int noIterations)