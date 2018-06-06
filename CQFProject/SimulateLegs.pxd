import numpy as np
from LowDiscrepancyNumberGenerators cimport SobolNumbers

cimport numpy as np
ctypedef np.float_t FDTYPE_t
ctypedef np.int_t DTYPE_t

cpdef np.ndarray UnifFromGaussCopula(np.ndarray LogRtnCorP, SobolNumbers NumbGen, int noIterations)

cpdef np.ndarray UnifFromTCopula(np.ndarray RankCorP, SobolNumbers NumbGen, int T_df, int noIterations)

cdef np.ndarray _TCopula_DF_MLE(np.ndarray U_hist_t, np.ndarray corM)

cpdef np.ndarray TCopula_DF_MLE(np.ndarray U_hist_t, np.ndarray corM)