
import cv2
import numpy as np
UBIT='ksehgal';
from copy import deepcopy
import numpy as np
np.random.seed(sum([ord(c) for c in UBIT]))

##Refernce
## https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html
##https://docs.opencv.org/3.2.0/da/de9/tutorial_py_epipolar_geometry.html
sift = cv2.xfeatures2d.SIFT_create()

def convert(img): 
	gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
	kp = sift.detect(gray,None)
	cv2.drawKeypoints(gray,kp,img)




image1 = cv2.imread("/Users/krishna/Downloads/data/mountain1.jpg")          
image2 = cv2.imread("/Users/krishna/Downloads/data/mountain2.jpg") 

image3 = cv2.imread('/Users/krishna/Downloads/data/tsucuba_left.png')  
image4 = cv2.imread('/Users/krishna/Downloads/data/tsucuba_right.png') 


task1_sift_1 = convert(image1)
cv2.imwrite('task1_sift1.jpg',image1)

task1_sift_2 = convert(image2)
cv2.imwrite('task1_sift2.jpg',image2)

task2_sift_1 = convert(image3)
cv2.imwrite('task2_sift1.jpg',image3)

task2_sift_2 = convert(image4)
cv2.imwrite('task2_sift2.jpg',image4)

