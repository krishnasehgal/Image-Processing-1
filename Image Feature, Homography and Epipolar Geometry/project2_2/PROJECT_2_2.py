import numpy as np
import cv2 
from matplotlib import pyplot as plt
img1 = cv2.imread('/Users/krishna/Downloads/data/tsucuba_left.png',0)   # left image
img2 = cv2.imread('/Users/krishna/Downloads/data/tsucuba_right.png',0)  # right image
sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)


# Used BruteForce matcher
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
good = []
pts1 = []
pts2 = []

#enumerate using Good value.
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

img7 = cv2.drawMatches(img1,kp1,img2,kp2,good,None)


cv2.imwrite('task2_matches_knn.jpg', img7)

points = []

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)


F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.RANSAC)

print(F)

# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

def drawlines(img1,img2,lines,pts1,pts2):

    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image

line = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
line = line.reshape(-1,3)
randomline=line[:10,]
point1=pts1[:10,]
point2=pts2[:10,]
img5,img6 = drawlines(img1,img2,randomline,point1,point2)
# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
rand_lines2=lines2[:10,]
rand_pts3=pts1[:10,]
rand_pts4=pts2[:10,]
img3,img4 = drawlines(img2,img1,rand_lines2,rand_pts3,rand_pts4)
cv2.imwrite('task2_epi_left.jpg' , img5)
cv2.imwrite('task2_epi_right.jpg' , img3)
