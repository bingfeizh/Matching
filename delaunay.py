#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 04:33:46 2017

@author: bingfei
"""
from scipy.spatial import Delaunay
import numpy as np
import cv2

img1 = cv2.imread('/Users/bingfei/project/Matching/HongKong/image/1.jpg')  # input left image
img2 = cv2.imread('/Users/bingfei/project/Matching/HongKong/image/2.jpg') # input right image

lpoint=np.load('lpoint.npy') #load matched points by sift matching
rpoint=np.load('rpoint.npy') #load matched points by sift matching

tri = Delaunay(lpoint)

triangle=tri.simplices

for i in range(len(triangle)):
    a=[lpoint[triangle[i][0]][0],lpoint[triangle[i][0]][1]]
    b=[lpoint[triangle[i][1]][0],lpoint[triangle[i][1]][1]]
    c=[lpoint[triangle[i][2]][0],lpoint[triangle[i][2]][1]]
    pts=np.int32([a,b,c])
    pts = pts.reshape((-1,1,2))
    cv2.polylines(img1, [pts],True,(0,0,255),2,1)
    
    a=[rpoint[triangle[i][0]][0],rpoint[triangle[i][0]][1]]
    b=[rpoint[triangle[i][1]][0],rpoint[triangle[i][1]][1]]
    c=[rpoint[triangle[i][2]][0],rpoint[triangle[i][2]][1]]
    pts=np.int32([a,b,c])
    pts = pts.reshape((-1,1,2))
    cv2.polylines(img2, [pts],True,(0,0,255),2,1)
    

#cv2.imwrite('/Users/bingfei/project/ltri.jpg',img1) 
#cv2.imwrite('/Users/bingfei/project/rtri.jpg',img2)  

np.save('triangle.npy',triangle)