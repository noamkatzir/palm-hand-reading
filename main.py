__author__ = 'noam'
import FindHand as fh
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

first_hand = fh.FindHand('./images/preprocessed/noam_left_hand_6.12.08_02062015.png')
# first_hand = fh.FindHand('./images/preprocessed/noam_right_hand_19.08.08_02062015.png')

first_hand.create_map_with_pyramid()
first_hand.find_hand_elements()

print len(first_hand.fingers)

small1 = first_hand.small_image.copy()

for finger in first_hand.fingers:
    finger_merged_contour = np.concatenate([element['contour'] for element in finger])
    ellipse = cv2.fitEllipse(finger_merged_contour)
    # cv2.drawContours(small1, [finger_merged_contour], 0, 1, 1)
    cv2.ellipse(small1, ellipse, (255, 0, 0), 1) # can add -1 to the tickness to fill the ellipse

rect = cv2.minAreaRect(first_hand.palm['contour'])
box = cv2.cv.BoxPoints(rect)
box = np.int0(box)
cv2.drawContours(small1, [box], 0, (0, 0, 255), 1)
first_hand.calculate_normalization_box_angle()
small1 = first_hand.rotateImage(small1, first_hand.palm['normalization_angle'])
# small1 = ndimage.rotate(small1, first_hand.get_normalization_box_angle())

second_hand = fh.FindHand('./images/preprocessed/noam_left_hand_30.5.15_02062015_0001.png')
# second_hand = fh.FindHand('./images/preprocessed/noam_right_hand_30.5.15_02062015.png')

second_hand.create_map_with_pyramid()
second_hand.find_hand_elements()

print len(second_hand.fingers)

small2 = second_hand.small_image.copy()

for finger in second_hand.fingers:
    finger_merged_contour = np.concatenate([element['contour'] for element in finger])
    ellipse = cv2.fitEllipse(finger_merged_contour)
    cv2.ellipse(small2, ellipse, (255, 0, 0), 1) # can add -1 to the tickness to fill the ellipse

rect = cv2.minAreaRect(second_hand.palm['contour'])
box = cv2.cv.BoxPoints(rect)
box = np.int0(box)
cv2.drawContours(small2, [box], 0, (0, 0, 255), 1)
second_hand.calculate_normalization_box_angle()
small2 = second_hand.rotateImage(small2, second_hand.palm['normalization_angle']+180)
# small2 = ndimage.rotate(small2, second_hand.get_normalization_box_angle())


plt.figure(1)
plt.subplot(231)
plt.imshow(small1, cmap='gray')
plt.title('2008')
plt.subplot(232)
plt.imshow(small2, cmap='gray')
plt.title('2015')
plt.show()