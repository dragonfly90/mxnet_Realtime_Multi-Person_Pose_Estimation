cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t

def putGaussianMaps(np.ndarray[DTYPE_t, ndim = 2] entry, DTYPE_t rows, 
		    DTYPE_t cols, DTYPE_t center_x, DTYPE_t center_y,  DTYPE_t stride, 
		    int grid_x, int grid_y, DTYPE_t sigma):
    cdef DTYPE_t start = stride / 2.0 - 0.5
    cdef DTYPE_t x, y, d2

    for g_y in range(grid_y):
        for g_x in range(grid_x):
            x = start + g_x * stride
            y = start + g_y * stride
            d2 = (x - center_x) * (x - center_x) + (y - center_y) * (y - center_y)
            exponent = d2 / 2.0 / sigma / sigma
            if (exponent > 4.6052):
                continue
            entry[g_y, g_x] += np.exp(-exponent)
            if (entry[g_y, g_x] > 1):
                entry[g_y, g_x] = 1
    return entry
