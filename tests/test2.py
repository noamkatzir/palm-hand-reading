__author__ = 'noam'

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

lefthand = cv2.imread('omer_left_hand_26.05.93_02062015.png', 0)


ret, threshold1 = cv2.threshold(lefthand,220,255,cv2.THRESH_BINARY_INV)
ret, threshold2 = cv2.threshold(lefthand,225,255,cv2.THRESH_BINARY_INV)


# plt.figure(1)
# plt.hist(lefthand.ravel(),256,[0,256])
# plt.show()

plt.figure(1)
plt.imshow(threshold1, cmap='gray')
plt.figure(2)
plt.imshow(threshold2, cmap='gray')
plt.show()
