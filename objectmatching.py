#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 04:39:12 2017

@author: bingfei
"""

import numpy as np
from munkres import Munkres
import cv2
import random

# load points and triangles
lpoint=np.load('lpoint.npy') #input matched points
rpoint=np.load('rpoint.npy') #input matched points
triangle=np.load('triangle.npy') #input triangulation result

# load bounding boxes
rcentroid=[[2469,696],
           [2250,732],
           [2400,696],
           [2103,738],
           [2061,711],
           [1995,654],
           [1878,657],
           [1890,624],
           [1758,645]] #input coordinates of target objects in the right image
lcentroid=[[1242,1407],
           [1500,1212],
           [2082,1266],
           [1929,1047],
           [1971,855],
           [2067,852],
           [2334,882],
           [2076,783],
           [2196,792],
           [2043,705],
           [2160,717],
           [1518,621]] #input coordinates of target objects in the left image
# Find triangles with controids
def sign(p1, p2, p3):
  return (p1[0] - p3[0][0]) * (p2[0][1] - p3[0][1]) - (p2[0][0] - p3[0][0]) * (p1[1] - p3[0][1])

def PointInTriangle(pt, v1, v2, v3):
  b1 = sign(pt, v1, v2) <= 0
  b2 = sign(pt, v2, v3) <= 0
  b3 = sign(pt, v3, v1) <= 0
  return ((b1 == b2) and (b2 == b3))

ltriangle=[]
rtriangle=[]
for i in range(len(lcentroid)):
    p=lcentroid[i]
    for j in range(len(triangle)):
        lvertexa=[lpoint[triangle[j][0]]]
        lvertexb=[lpoint[triangle[j][1]]]
        lvertexc=[lpoint[triangle[j][2]]]
        rvertexa=[rpoint[triangle[j][0]]]
        rvertexb=[rpoint[triangle[j][1]]]
        rvertexc=[rpoint[triangle[j][2]]]
        if PointInTriangle(p, lvertexa, lvertexb, lvertexc):
            ltriangle.append([lvertexa, lvertexb, lvertexc])
            rtriangle.append([rvertexa, rvertexb, rvertexc])

# Calculate triangle coordiantes
def triangle_area(a, b, c):
    return abs(0.5*(a[0]*(b[1]-c[1])+b[0]*(c[1]-a[1])+c[0]*(a[1]-b[1])))

def TriangleCoordinates(p,a,b,c):
    tria=triangle_area(p, b, c)/triangle_area(a, b, c)
    trib=triangle_area(a, p, c)/triangle_area(a, b, c)
    tric=triangle_area(a, b, p)/triangle_area(a, b, c)
    return  [tria,trib,tric]

def distance(a,b):
    return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)

dis3=[]
for jj in range(len(ltriangle)):
    ltrico=[]
    rtrico=[]
    
    for lii in range(len(lcentroid)):
        ltricoordinate=TriangleCoordinates(lcentroid[lii],lpoint[triangle[jj][0]],lpoint[triangle[jj][1]],lpoint[triangle[jj][2]])
        ltrico.append(ltricoordinate)

    for rii in range(len(rcentroid)):   
        rtricoordinate=TriangleCoordinates(rcentroid[rii],rpoint[triangle[jj][0]],rpoint[triangle[jj][1]],rpoint[triangle[jj][2]])
        rtrico.append(rtricoordinate)
    
    # Generater cost matrix
    dis2=[]
    for liii in range(len(ltrico)):
        dis1=[]
        for riii in range(len(rtrico)):
            dis1.append(distance(ltrico[liii],rtrico[riii]))
        dis2.append(dis1)
    dis3.append(dis2)
    
#Combinatorial optimization
m=Munkres()

cost=np.zeros([len(ltrico),len(rtrico)])
for k in range(len(dis3)-1):
    matrix=dis3[k]
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

img1 = cv2.imread('/Users/bingfei/project/Matching/HongKong/image/1.jpg')  #input left image
img2 = cv2.imread('/Users/bingfei/project/Matching/HongKong/image/2.jpg')  #input right image

for r in range(len(index)):
    
    rr=random.randint(0,255)
    g=random.randint(0,255)
    b=random.randint(0,255)
    cv2.circle(img1,(lcentroid[index[r][0]][0],lcentroid[index[r][0]][1]), 25, (rr,g,b),5)    
    cv2.circle(img2,(rcentroid[index[r][1]][0],rcentroid[index[r][1]][1]), 25, (rr,g,b),5)
        
cv2.imwrite('/Users/bingfei/project/Matching/HongKong/image/111.jpg',img1)  
cv2.imwrite('/Users/bingfei/project/Matching/HongKong/image/222.jpg',img2)     
cv2.waitKey(0)  
cv2.destroyAllWindows() 
        
        
        
        
        
        
        