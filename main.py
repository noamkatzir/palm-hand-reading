__author__ = 'noam'
import FindHand as fh
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

first_hand = fh.FindHand('./images/preprocessed/noam_left_hand_6.12.08_02062015.png', 'left')
# first_hand = fh.FindHand('./images/preprocessed/noam_right_hand_19.08.08_02062015.png', 'right')

first_hand.create_map_with_pyramid()
first_hand.find_hand_elements()

print len(first_hand.fingers)

small1 = first_hand.small_image.copy()
small1_orig = first_hand.small_image.copy()

# for finger in first_hand.fingers:
#     finger_merged_contour = np.concatenate([element['contour'] for element in finger])
#     ellipse = cv2.fitEllipse(finger_merged_contour)
#     # cv2.drawContours(small1, [finger_merged_contour], 0, 1, 1)
#     cv2.ellipse(small1, ellipse, (255, 0, 0), 1) # can add -1 to the tickness to fill the ellipse

rect = cv2.minAreaRect(first_hand.palm['contour'])
box = cv2.cv.BoxPoints(rect)
box = np.int0(box)
cv2.drawContours(small1, [box], 0, (0, 0, 255), 1)
first_hand.calculate_normalization_box_angle()
first_hand.rotate_contours_according_palm_center(first_hand.palm['normalization_angle'])
first_hand.map_fingers_and_orientation()
first_hand.map_fingers_angles_from_palm()
full_normalization_angle1 = first_hand.palm['normalization_angle'] + first_hand.orientation_angle + (first_hand.fingers[2][0]['angleFromPalmCenter'] - first_hand.orientation_angle)
first_hand.rotate_contours_according_palm_center(full_normalization_angle1)
small1_normalized = first_hand.rotate_image(small1, first_hand.palm['normalization_angle'])
small1 = first_hand.rotate_image(small1, full_normalization_angle1)
cv2.circle(small1, (int(first_hand.palm['rotatedCenter'][0]), int(first_hand.palm['rotatedCenter'][1])), 3, (0, 0, 255), -1)

for i in xrange(len(first_hand.rotatedContours)):
    cv2.drawContours(small1, first_hand.rotatedContours, i, (0, 0, 255), 1)

for i in xrange(len(first_hand.fingers)):
    cv2.putText(small1,"{}".format(i), (int(first_hand.fingers[i][0]['rotatedCenter'][0]), int(first_hand.fingers[i][0]['rotatedCenter'][1])), cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 255)

# small1 = ndimage.rotate(small1, first_hand.get_normalization_box_angle())

second_hand = fh.FindHand('./images/preprocessed/noam_left_hand_30.5.15_02062015_0001.png', 'left')
# second_hand = fh.FindHand('./images/preprocessed/noam_right_hand_30.5.15_02062015.png', 'right')

second_hand.create_map_with_pyramid()
second_hand.find_hand_elements()

print len(second_hand.fingers)

small2 = second_hand.small_image.copy()
small2_orig = second_hand.small_image.copy()

# for finger in second_hand.fingers:
#     finger_merged_contour = np.concatenate([element['contour'] for element in finger])
#     ellipse = cv2.fitEllipse(finger_merged_contour)
#     cv2.ellipse(small2, ellipse, (255, 0, 0), 1) # can add -1 to the tickness to fill the ellipse

rect = cv2.minAreaRect(second_hand.palm['contour'])
box = cv2.cv.BoxPoints(rect)
box = np.int0(box)
cv2.drawContours(small2, [box], 0, (0, 0, 255), 1)
second_hand.calculate_normalization_box_angle()
second_hand.rotate_contours_according_palm_center(second_hand.palm['normalization_angle'])
second_hand.map_fingers_and_orientation()
second_hand.map_fingers_angles_from_palm()
full_normalization_angle2 = second_hand.palm['normalization_angle'] + second_hand.orientation_angle + (second_hand.fingers[2][0]['angleFromPalmCenter'] - second_hand.orientation_angle)
second_hand.rotate_contours_according_palm_center(full_normalization_angle2)
small2_normalized = second_hand.rotate_image(small2, second_hand.palm['normalization_angle'])
small2 = second_hand.rotate_image(small2, full_normalization_angle2)
cv2.circle(small2, (int(second_hand.palm['rotatedCenter'][0]), int(second_hand.palm['rotatedCenter'][1])), 3, (0, 0, 255), -1)

for i in xrange(len(second_hand.rotatedContours)):
    cv2.drawContours(small2, second_hand.rotatedContours, i, (0, 0, 255), 1)

for i in xrange(len(second_hand.fingers)):
    cv2.putText(small2,"{}".format(i), (int(second_hand.fingers[i][0]['rotatedCenter'][0]), int(second_hand.fingers[i][0]['rotatedCenter'][1])), cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 255)
# small2 = second_hand.rotate_image(small2, second_hand.palm['normalization_angle']+180)
# small2 = ndimage.rotate(small2, second_hand.get_normalization_box_angle())


plt.figure(1)
plt.subplot(231)
plt.imshow(small1_orig, cmap='gray')
plt.title('2008 origin')
plt.subplot(232)
plt.imshow(small1_normalized, cmap='gray')
plt.title('2008 normalized')
plt.subplot(233)
plt.imshow(small1, cmap='gray')
plt.title('2008')
plt.subplot(234)
plt.imshow(small2_orig, cmap='gray')
plt.title('2015 origin')
plt.subplot(235)
plt.imshow(small2_normalized, cmap='gray')
plt.title('2015 normalized')
plt.subplot(236)
plt.imshow(small2, cmap='gray')
plt.title('2015')
plt.show()