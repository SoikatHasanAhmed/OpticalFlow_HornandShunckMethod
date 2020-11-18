from scipy.ndimage.filters import convolve as filter2
import numpy as np
from derivatives import computeDerivatives

HSKERN = np.array([[1/12, 1/6, 1/12],
                   [1/6,    0, 1/6],
                   [1/12, 1/6, 1/12]], float)

def HS_Algorithm(im1, im2, *, alpha = 0.001, Niter = 8):

    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)

    # set up initial velocities
    uInitial = np.zeros([im1.shape[0], im1.shape[1]])
    vInitial = np.zeros([im1.shape[0], im1.shape[1]])

    U = uInitial
    V = vInitial

    # Estimate derivatives
    [fx, fy, ft] = computeDerivatives(im1, im2)

    for _ in range(Niter):

        uAvg = filter2(U, HSKERN)
        vAvg = filter2(V, HSKERN)

        der = (fx*uAvg + fy*vAvg + ft) / (alpha**2 + fx**2 + fy**2)
        U = uAvg - fx * der
        V = vAvg - fy * der

    return U, V