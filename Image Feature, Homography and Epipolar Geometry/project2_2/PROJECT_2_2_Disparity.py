


import numpy as np
import cv2 
from matplotlib import pyplot as plt

img1 = cv2.imread('/Users/krishna/Downloads/data/tsucuba_left.png',0)   # left image
img2 = cv2.imread('/Users/krishna/Downloads/data/tsucuba_right.png',0)  # right image
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(img1, img2)
plt.imsave("disparity.png", disparity)

