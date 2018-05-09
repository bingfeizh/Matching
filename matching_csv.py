    
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np
from scipy.spatial import Delaunay
from munkres import Munkres
import csv

def sign(p1, p2, p3):
    return (p1[0] - p3[0][0]) * (p2[0][1] - p3[0][1]) - (p2[0][0] - p3[0][0]) * (p1[1] - p3[0][1])

def PointInTriangle(pt, v1, v2, v3):
    b1 = sign(pt, v1, v2) <= 0
    b2 = sign(pt, v2, v3) <= 0
    b3 = sign(pt, v3, v1) <= 0
    return ((b1 == b2) and (b2 == b3))

def triangle_area(a, b, c):
    return abs(0.5*(a[0]*(b[1]-c[1])+b[0]*(c[1]-a[1])+c[0]*(a[1]-b[1])))

def TriangleCoordinates(p,a,b,c):
    #print [p,a,b,c]
    #print triangle_area(a, b, c)
    tria=triangle_area(p, b, c)/triangle_area(a, b, c)
    trib=triangle_area(a, p, c)/triangle_area(a, b, c)
    tric=triangle_area(a, b, p)/triangle_area(a, b, c)
    return  [tria,trib,tric]

def distance(a,b):
    dis=np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)
    return dis

def bbx2centroid(boundingbox):
    centroid=[]
    for i in range(len(boundingbox)):
        bbx=boundingbox[i]
        x=bbx[0]+0.5*(bbx[2]-bbx[0])
        y=bbx[1]+0.5*(bbx[3]-bbx[1])
        centroid.append([x,y])
    return centroid
    
def removesamenumber(pts1,pts2):
    pts_1=[]
    pts_2=[]

    for i in range(len(pts1)):
        a=list(pts1[i]) in pts_1
        b=list(pts2[i]) in pts_2
        if a or b:
            continue
        else:
            pts_1.append(list(pts1[i]))
            pts_2.append(list(pts2[i]))
    return np.array(pts_1),np.array(pts_2)
            

def Matching(img1,img2,bbx_1,bbx_2):
    
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    sift = cv2.SURF()

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

    pts1 = np.float64(pts1)
    pts2 = np.float64(pts2)
    
    pts1,pts2=removesamenumber(pts1,pts2)
    #print len(pts1)
    #print pts1
    #print pts2

    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC)

    tri = Delaunay(pts1)

    triangle=tri.simplices

    lpoint=pts1
    rpoint=pts2

    # load bounding boxes
    centroid_1=bbx2centroid(bbx_1)
    centroid_2=bbx2centroid(bbx_2)    
    #print centroid_1
    #print centroid_2
    
    # Find triangles with controids
    triangle_1=[]
    triangle_2=[]
    
    for i in range(len(centroid_1)):
        p=centroid_1[i]
        for j in range(len(triangle)):
            lvertexa=[lpoint[triangle[j][0]]]
            lvertexb=[lpoint[triangle[j][1]]]
            lvertexc=[lpoint[triangle[j][2]]]
            rvertexa=[rpoint[triangle[j][0]]]
            rvertexb=[rpoint[triangle[j][1]]]
            rvertexc=[rpoint[triangle[j][2]]]
            if PointInTriangle(p, lvertexa, lvertexb, lvertexc):
                triangle_1.append([lvertexa, lvertexb, lvertexc])
                triangle_2.append([rvertexa, rvertexb, rvertexc])
        if not triangle_1:
            triangle_1.append([lvertexa, lvertexb, lvertexc])
            triangle_2.append([rvertexa, rvertexb, rvertexc])
    # Calculate triangle coordiantes
    
    ltc=[]
    rtc=[]
    for jj in range(len(triangle_1)):
        ltrico=[]
        rtrico=[]
    
        for lii in range(len(centroid_1)):
            ltricoordinate=TriangleCoordinates(centroid_1[lii],(np.float32(triangle_1[jj][0][0][0]),np.float32(triangle_1[jj][0][0][1])),\
                                               (np.float32(triangle_1[jj][1][0][0]),np.float32(triangle_1[jj][1][0][1])),\
                                               (np.float32(triangle_1[jj][2][0][0]),np.float32(triangle_1[jj][2][0][1])))
            ltrico.append(ltricoordinate)

        for rii in range(len(centroid_2)):   
            rtricoordinate=TriangleCoordinates(centroid_2[rii],(np.float32(triangle_2[jj][0][0][0]),np.float32(triangle_2[jj][0][0][1])),\
                                               (np.float32(triangle_2[jj][1][0][0]),np.float32(triangle_2[jj][1][0][1])),\
                                               (np.float32(triangle_2[jj][2][0][0]),np.float32(triangle_2[jj][2][0][1])))
            rtrico.append(rtricoordinate)
        ltc.append(ltrico)
        rtc.append(rtrico)
    #print ltc
    #print rtc
        
    # Generater cost matrix
    dis2=[]
    for dii in range(len(triangle_1)):
        dis1=[]
        for di in range(len(ltc[0])):
            dis=[]
            for d in range(len(rtc[0])):
                dis.append(distance(ltc[dii][di],rtc[dii][d]))
            dis1.append(dis)
        dis2.append(dis1)
    #print dis2

    #Combinatorial optimization
    m=Munkres()
    cost=np.zeros([len(ltrico),len(rtrico)])
    for mx in range(len(dis2)):                
        matrix=dis2[mx]
        indexes = m.compute(matrix)
        for row, column in indexes:
            cost[row][column]=cost[row][column]+1


    cost_matrix = []
    for row in cost:
        cost_row = []
        for col in row:
            cost_row += [1000 - col]
        cost_matrix += [cost_row]
    index=m.compute(cost_matrix)
    
    for row, column in index:
        print '(%d, %d)' % (row, column)

    return index
        



