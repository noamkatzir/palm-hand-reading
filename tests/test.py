__author__ = 'noam'

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


old_lefthand = cv2.imread('noam_left_hand_6.12.08_02062015.png', 0)
lefthand = cv2.imread('noam_left_hand_30.5.15_02062015_0001.png', 0)
omer_lefthand = cv2.imread('omer_left_hand_26.05.93_02062015.png', 0)


ret, threshold1 = cv2.threshold(old_lefthand,252,255,cv2.THRESH_BINARY_INV)
ret, threshold2 = cv2.threshold(lefthand,252,255,cv2.THRESH_BINARY_INV)
ret, threshold3 = cv2.threshold(omer_lefthand,252,255,cv2.THRESH_BINARY_INV)

plt.figure(1)
plt.hist(old_lefthand.ravel(),256,[0,256])
plt.figure(2)
plt.hist(lefthand.ravel(),256,[0,256])
plt.figure(3)
plt.hist(omer_lefthand.ravel(),256,[0,256])
plt.show()

plt.figure(1)
# plt.imshow(lefthand, cmap='gray')
plt.imshow(threshold1, cmap='gray')
plt.figure(2)
plt.imshow(threshold2, cmap='gray')
plt.figure(3)
plt.imshow(threshold3, cmap='gray')
plt.show()
raw_input()