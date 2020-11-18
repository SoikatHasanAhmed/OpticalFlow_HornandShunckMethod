import cv2
import numpy as np
from HS import HS_Algorithm

if __name__ == '__main__':
    path = './caltrain/caltrain'

    for i in range (1,32):
        # read two frames in grayscale to input this in the algorithm
        image1 = cv2.imread('{}{:03}.png'.format(path,i), 0)
        image2 = cv2.imread('{}{:03}.png'.format(path,i+1), 0)
         # calculating the U and V
        U, V = HS_Algorithm(image1,image2, alpha=1.0, Niter=100)

        # making ht a empty image where we store the color coded optical flow
        hsv = np.zeros((400,512,3), dtype=np.uint8)
        hsv[..., 1] = 255

        mag, ang = cv2.cartToPolar(U,V)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        # images for visualization and store
        temp1 = np.dstack((image1,image1,image1))
        temp2 = np.dstack((image2, image2, image2))

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        out =  np.hstack([temp1,temp2,bgr])
        cv2.imshow("colored flow", out)
        # storing the output along with the first ( left) and second frame (midle)
        cv2.imwrite('./outputs/optical_flow_between_{}_and_{}_frames.png'.format(i,i+1),out)
        cv2.waitKey(1)
