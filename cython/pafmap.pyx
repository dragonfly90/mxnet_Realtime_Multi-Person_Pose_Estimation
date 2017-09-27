cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t

def putVecMaps(np.ndarray[DTYPE_t, ndim = 2] entryX, np.ndarray[DTYPE_t, ndim = 2]  entryY, 
               np.ndarray[DTYPE_t, ndim = 2] count, DTYPE_t center1_x, DTYPE_t center1_y,
               DTYPE_t center2_x, DTYPE_t center2_y, DTYPE_t stride, DTYPE_t grid_x, DTYPE_t grid_y, 
               DTYPE_t sigma, DTYPE_t thre):
    
    cdef DTYPE_t centerA_x = 0.125 * center1_x
    cdef DTYPE_t centerA_y = 0.125 * center1_y

    cdef DTYPE_t centerB_x = 0.125 * center2_x
    cdef DTYPE_t centerB_y = 0.125 * center2_y
    
    cdef DTYPE_t bc_x = centerB_x - centerA_x
    cdef DTYPE_t bc_y = centerB_y - centerA_y
    
    cdef int min_x = max(int(round(min(centerA_x, centerB_x)) - thre), 0)
    cdef int max_x = min(int(round(max(centerA_x, centerB_x))) + thre, grid_x)
    cdef int min_y = max(int(round(min(centerA_y, centerB_y) - thre)), 0)
    cdef int max_y = min(int(round(max(centerA_y, centerB_y) + thre)), grid_y)

    cdef DTYPE_t norm_bc = np.sqrt(bc_x * bc_x + bc_y * bc_y)
    if norm_bc == 0:
        return 
    bc_x = bc_x/norm_bc
    bc_y = bc_y/norm_bc
    
    cdef DTYPE_t ba_x, ba_y
    cdef DTYPE_t dist
    cdef DTYPE_t cnt

    for g_y in range(min_y, max_y):
        for g_x in range(min_x, max_x):
            ba_x = g_x - centerA_x
            ba_y = g_y - centerA_y
            dist = np.absolute(ba_x * bc_y - ba_y * bc_x)

            if (dist <= thre):
                cnt = count[g_y, g_x]
                if (cnt == 0):
                    entryX[g_y, g_x] = bc_x
                    entryY[g_y, g_x] = bc_y
                else:
                    entryX[g_y, g_x] = (entryX[g_y, g_x] * cnt + bc_x) / (cnt + 1)
                    entryY[g_y, g_x] = (entryY[g_y, g_x] * cnt + bc_y) / (cnt + 1)
                    count[g_y, g_x] = cnt + 1
