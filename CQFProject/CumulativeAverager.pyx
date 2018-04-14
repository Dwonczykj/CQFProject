import numpy as np
cimport numpy as np

def CumAverage(np.ndarray arr):
	cdef int i
	#Remember that arr is a copy as not passed in by reference as its a value type. So need to manually init a copy.
	for i in range(1,len(arr)):
		arr[i] = ((arr[i-1]*(i)+arr[i])/(i+1))
	return arr