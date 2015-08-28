__author__ = 'noam'

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg
from operator import itemgetter, attrgetter, methodcaller

# return list of center sorted descending to the contour area
# and the min rect cordinates
def mapPalmAndFingers(contours, image):
    handElements = []
    for i in xrange(len(contours)):
        M = cv2.moments(contours[i])
        centroid_x = int(M['m10']/M['m00'])
        centroid_y = int(M['m01']/M['m00'])

        # cv2.circle(lefthand, (centroid_x, centroid_y), 10, (255, 0, 0),-1)

        element = {
            'center': (centroid_x, centroid_y),
            'contour': contours[i],
            'area': cv2.contourArea(contours[i]),
            'palm': False
        }

        handElements.append(element)

    # mapp the distance of hand elements from the palm
    palm = max(handElements, key=lambda x: x['area'])
    palm['palm'] = True
    palm['dist'] = 0

    for handElement in handElements:
        if handElement['palm'] is False:
            handElement['dist'] = linalg.norm(np.array(handElement['center']) - np.array(palm['center']))

    # sorting the hand helements by the distance from the palm
    # because this way we will find the end of the fingers
    handElements = sorted(handElements, key=itemgetter('dist'), reverse=True)

    # for handElement in handElements:
    #     cv2.fitLine(handElement['contour'])

    return handElements, palm

imagesPath = '../images/preprocessed/'
# lefthand = cv2.imread(imagesPath+'noam_left_hand_30.5.15_02062015_0001.png')
lefthand = cv2.imread(imagesPath+'noam_left_hand_6.12.08_02062015.png')
small_color = cv2.resize(lefthand, (0, 0), fx=0.5**5, fy=0.5**5)

lefthand_imgray = cv2.cvtColor(lefthand,cv2.COLOR_BGR2GRAY)

# I resize the image to very samll size and then get the samll image back to the size of the origin
# this way I remove most of the noise and keep the large objects
small = cv2.resize(lefthand_imgray, (0, 0), fx=0.5**5, fy=0.5**5)
small_orig = small.copy()
small = cv2.resize(small, (lefthand_imgray.shape[1], lefthand.shape[0]),interpolation=cv2.INTER_CUBIC)

# I put the threshold to get the hand elements binay image
ret, threshold1 = cv2.threshold(small_orig,245,255,cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(threshold1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros(threshold1.shape,np.uint8)

# I'm filling the contours of with large area to ensure I get a the objects with minimum noise
for i in xrange(len(contours)):
    area = cv2.contourArea(contours[i])

    # in origin it was 300000, but it start ignoring fingers
    if area > 10:
        cv2.drawContours(mask, contours,i,255,-1)

# converting again to binary image & finding the contours of the large objects
ret, threshold2 = cv2.threshold(mask, 0, 256, cv2.THRESH_BINARY)
contours2, hierarchy2 = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
mask = mask.astype('uint8')
cv2.drawContours(small_color, contours2, -1, (0, 255, 0), 1)


elements, palm = mapPalmAndFingers(contours2, lefthand)

for element in elements:

    cv2.line(small_color, palm['center'], element['center'], (0, 255, 0), 1)
    ellipse = cv2.fitEllipse(element['contour'])
    cv2.ellipse(small_color, ellipse, (255, 0, 0), 1) # can add -1 to the tickness to fill the ellipse

    #cv2.putText(lefthand, '{}'.format(dist), element['center'], cv2.FONT_HERSHEY_SIMPLEX, 4, 1, 20)

# largeCon = np.concatenate((elements[0]['contour'], elements[1]['contour']), axis=0)
# ellipse = cv2.fitEllipse(largeCon)
# cv2.ellipse(lefthand, ellipse, (255, 0, 0), 2)

plt.figure(1)
plt.imshow(mask, cmap='gray')
plt.figure(2)
plt.imshow(small_color, cmap='gray')
plt.show()

