import numpy as np
from LowDiscrepancyNumberGenerators cimport SobolNumbers

cimport numpy as np
ctypedef np.float_t FDTYPE_t

cpdef np.ndarray MultVarGaussianCopula(P,SobolNumbers LowDiscNumbers)
cpdef np.ndarray MultVarTDistnCopula(P,int df,SobolNumbers LowDiscNumbers)
cpdef FDTYPE_t TCopulaDensity(np.ndarray U, int df)