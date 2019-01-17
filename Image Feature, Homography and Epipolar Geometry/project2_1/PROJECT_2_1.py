import numpy as np
import cv2
from matplotlib import pyplot as plt

def warp(img1, img2, h):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    lp1 = np.float32([[0,0], [0,rows1], [cols1, rows1], [cols1,0]]).reshape(-1,1,2)
    temp = np.float32([[0,0], [0,rows2], [cols2, rows2], [cols2,0]]).reshape(-1,1,2)

    lp2 = cv2.perspectiveTransform(temp, M)
    lp = np.concatenate((lp1, lp2), axis=0)

    [x_min, y_min] = np.int32(lp.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(lp.max(axis=0).ravel() + 0.5)

    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0,0,1]])

    result = cv2.warpPerspective(img1, H_translation.dot(M), (x_max - x_min, y_max - y_min))
    result[translation_dist[1]:rows1+translation_dist[1],translation_dist[0]:cols1+translation_dist[0]] = img2
    return result

MIN_MATCH_COUNT = 10


img1 = cv2.imread("/Users/krishna/Downloads/data/mountain1.jpg",0)  # queryImage
img2 = cv2.imread("/Users/krishna/Downloads/data/mountain2.jpg",0) # trainImage

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

#Using BruteForce to match the points
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)


# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append(m)
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    points = []
    for i in range(len(matchesMask)):
        if matchesMask[i] == 1:
            points.append(i)
    points = points[:10]

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,flags=2)

    plt.imsave('task1_matches.png', img3)

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

tmp=warp(img1, img2, h)

cv2.imwrite("task1_pano.jpg)", tmp)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

plt.imshow(img3, 'gray'),plt.show()




