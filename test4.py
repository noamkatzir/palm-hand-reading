__author__ = 'noam'

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


lefthand = cv2.imread('noam_left_hand_30.5.15_02062015_0001.png')
lefthand_imgray = cv2.cvtColor(lefthand,cv2.COLOR_BGR2GRAY)

small = cv2.resize(lefthand_imgray, (0, 0), fx=0.5**6, fy=0.5**6)

small = cv2.resize(small, (lefthand_imgray.shape[1], lefthand.shape[0]),interpolation=cv2.INTER_CUBIC)
ret, threshold1 = cv2.threshold(small,245,255,cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(threshold1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros(threshold1.shape,np.uint8)


areas = []
for i in xrange(len(contours)):
    area = cv2.contourArea(contours[i])
    if(area > 300000):
        cv2.drawContours(mask, contours,i,255,-1)

ret, threshold2 = cv2.threshold(mask, 0, 256, cv2.THRESH_BINARY)
contours2, hierarchy2 = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
mask = mask.astype('uint8')
cv2.drawContours(lefthand,contours2,-1,(0,255,0),3)

plt.figure(1)
plt.imshow(mask, cmap='gray')
plt.figure(2)
plt.imshow(lefthand, cmap='gray')
plt.show()

