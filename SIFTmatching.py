#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 16:01:49 2017

@author: bingfei
"""

import cv2
import numpy as np
import random


img1 = cv2.imread('/Users/bingfei/project/Matching/HongKong/image/1.jpg',0)  #input left image
img2 = cv2.imread('/Users/bingfei/project/Matching/HongKong/image/2.jpg',0) #input right image
img3 = cv2.imread('/Users/bingfei/project/Matching/HongKong/image/1.jpg') #input left image
img4 = cv2.imread('/Users/bingfei/project/Matching/HongKong/image/2.jpg') #input right image
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

good = []
pts1 = []
pts2 = []

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

pts1 = np.float32(pts1)
pts2 = np.float32(pts2)

F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC)

np.save('lpoint.npy',pts1)
np.save('rpoint.npy',pts2)
np.save('F.npy',F)

for i in range(len(pts1)):
    r=random.randint(0,255)
    g=random.randint(0,255)
    b=random.randint(0,255)
    
    cv2.circle(img3, (pts1[i][0],pts1[i][1]), 10, (r,g,b),-1)
    cv2.circle(img4, (pts2[i][0],pts2[i][1]), 10, (r,g,b),-1)
#cv2.imshow("Image1", img1)   
#cv2.waitKey(0)
#cv2.destoryAllWindows()  
  
cv2.imwrite('/Users/bingfei/project/Matching/HongKong/image/11.jpg',img3)   
cv2.imwrite('/Users/bingfei/project/Matching/HongKong/image/21.jpg',img4)   