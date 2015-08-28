__author__ = 'noam'

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

def sortObjects(contours):
    areas = []
    for i in xrange(len(contours)):
        M = cv2.moments(contours[i])
        centroid_x = int(M['m10']/M['m00'])
        centroid_y = int(M['m01']/M['m00'])
        # cv2.circle(lefthand, (centroid_x, centroid_y), 10, (255, 0, 0),-1)

        areas.append([cv2.contourArea(contours[i]), i, centroid_x, centroid_y])

    arr= np.array(areas)
    return arr[arr[:, 0].argsort()[::-1]]

imagesPath = '../images/preprocessed/'
# lefthand = cv2.imread(imagesPath+'noam_left_hand_30.5.15_02062015_0001.png')
lefthand = cv2.imread(imagesPath+'noam_left_hand_6.12.08_02062015.png')

lefthand_imgray = cv2.cvtColor(lefthand,cv2.COLOR_BGR2GRAY)

small = cv2.resize(lefthand_imgray, (0, 0), fx=0.5**6, fy=0.5**6)

small = cv2.resize(small, (lefthand_imgray.shape[1], lefthand.shape[0]),interpolation=cv2.INTER_CUBIC)
ret, threshold1 = cv2.threshold(small,245,255,cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(threshold1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros(threshold1.shape,np.uint8)



for i in xrange(len(contours)):
    area = cv2.contourArea(contours[i])
    if(area > 70000): #in origin it was 300000, but it start ignoring fingers
        cv2.drawContours(mask, contours,i,255,-1)

ret, threshold2 = cv2.threshold(mask, 0, 256, cv2.THRESH_BINARY)
contours2, hierarchy2 = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
mask = mask.astype('uint8')
cv2.drawContours(lefthand,contours2,-1,(0,255,0),3)

objects = sortObjects(contours2)

center = np.array([objects[0][2], objects[0][3]])
for i in xrange(1,len(objects)):
    dist = np.linalg.norm(np.array([objects[i][2], objects[i][3]]) - center)
    if dist > 2500:
        continue

    cv2.line(lefthand, (int(objects[0][2]), int(objects[0][3])), (int(objects[i][2]), int(objects[i][3])), (0,255,0),5)

    cv2.putText(lefthand,'{}'.format(dist),(int(objects[i][2]), int(objects[i][3])), cv2.FONT_HERSHEY_SIMPLEX, 4, 1,20)
    print('distance between center and {} is {}'.format(i, dist))

plt.figure(1)
plt.imshow(mask, cmap='gray')
plt.figure(2)
plt.imshow(lefthand, cmap='gray')
plt.show()

