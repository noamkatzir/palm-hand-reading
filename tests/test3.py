__author__ = 'noam'

__author__ = 'noam'

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


lefthand = cv2.imread('noam_left_hand_30.5.15_02062015_0001.png', 0)
small = cv2.resize(lefthand, (0, 0), fx=0.5**6, fy=0.5**6)

# lefthand = cv2.imread('omer_left_hand_26.05.93_02062015.png', 0)

# ret, threshold = cv2.threshold(lefthand, 220, 255, cv2.THRESH_BINARY_INV)

# G = lefthand.copy()
# gpLefthand = [G]
# for i in xrange(6):
#     G = cv2.pyrDown(G)
#     gpLefthand.append(G)

# lpLefthand = [gpLefthand[5]]
# for i in xrange(5,0,-1):
#     GE = cv2.pyrUp(gpLefthand[i])
#     L = cv2.subtract(gpLefthand[i-1],GE)
#     lpLefthand.append(L)

# plt.figure(1)
# pyup = np.zeros((1,1))
# up = cv2.pyrUp(gpLefthand[6])
# for i in xrange(5):
#     up = cv2.pyrUp(up)
small = cv2.resize(small, (lefthand.shape[1], lefthand.shape[0]),interpolation=cv2.INTER_CUBIC)
ret, threshold1 = cv2.threshold(small,245,255,cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(threshold1, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros(threshold1.shape)


areas = []
for i in xrange(len(contours)):
    area = cv2.contourArea(contours[i])
    if(area > 300000):
        cv2.drawContours(mask, contours,i,255,-1)

# plt.figure(1)
# plt.plot(range(165), areas)
plt.figure(3)
plt.imshow(mask, cmap='gray')

# plt.figure(1)
# plt.imshow(threshold1, cmap='gray')
plt.figure(2)
plt.imshow(lefthand, cmap='gray')
plt.show()


# plt.figure(1)
# plt.hist(small.ravel(),256,[0,256])
# plt.figure(2)
# plt.imshow(small, cmap='gray')
# plt.show()
#
# ret, threshold1 = cv2.threshold(lefthand,220,255,cv2.THRESH_BINARY_INV)
# threshold1_old = threshold1.copy()
#
#
# contours, hierarchy = cv2.findContours(threshold1 ,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#
#
# # plt.figure(1)
# # plt.hist(lefthand.ravel(),256,[0,256])
# # plt.show()
#
# # plt.figure(1)
# # plt.imshow(threshold1, cmap='gray')
# # plt.figure(2)
# # plt.imshow(threshold2, cmap='gray')
# # plt.show()
#
# mask = np.uint8(np.zeros(lefthand.shape))
# cv2.drawContours(mask, contours,-1,255,-1)
# plt.figure(1)
# plt.imshow(mask, cmap='gray')
# plt.figure(2)
# plt.imshow(threshold1_old, cmap='gray')
# plt.figure(3)
# plt.imshow(mask - threshold1_old, cmap='gray')
# plt.show()
