from scipy.ndimage.filters import convolve as filter2
import numpy as np


# kernel for x nd y and t derivative
kernelX = np.array([[-1, 1],
                    [-1, 1]]) * .25  # kernel for computing d/dx

kernelY = np.array([[-1, -1],
                    [1, 1]]) * .25  # kernel for computing d/dy

kernelT = np.ones((2, 2))*.25

def computeDerivatives(im1, im2) :

    fx = filter2(im1, kernelX) + filter2(im2, kernelX)
    fy = filter2(im1, kernelY) + filter2(im2, kernelY)
    ft = filter2(im1, kernelT) + filter2(im2, -kernelT)

    return fx, fy, ft