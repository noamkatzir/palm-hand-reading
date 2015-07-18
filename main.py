__author__ = 'noam'
import FindHand as fh
import cv2
import numpy as np
from matplotlib import pyplot as plt

# hand = fh.FindHand('./images/preprocessed/noam_left_hand_30.5.15_02062015_0001.png')
first_hand = fh.FindHand('./images/preprocessed/noam_left_hand_6.12.08_02062015.png')

first_hand.create_map_with_pyramid()
first_hand.find_hand_elements()

print len(first_hand.fingers)

small1 = first_hand.small_image.copy()

for finger in first_hand.fingers:
    finger_merged_contour = np.concatenate([element['contour'] for element in finger])
    ellipse = cv2.fitEllipse(finger_merged_contour)
    cv2.ellipse(small1, ellipse, (255, 0, 0), 1) # can add -1 to the tickness to fill the ellipse

second_hand = fh.FindHand('./images/preprocessed/noam_left_hand_30.5.15_02062015_0001.png')

second_hand.create_map_with_pyramid()
second_hand.find_hand_elements()

print len(second_hand.fingers)

small2 = second_hand.small_image.copy()

for finger in second_hand.fingers:
    finger_merged_contour = np.concatenate([element['contour'] for element in finger])
    ellipse = cv2.fitEllipse(finger_merged_contour)
    cv2.ellipse(small2, ellipse, (255, 0, 0), 1) # can add -1 to the tickness to fill the ellipse

plt.figure(1)
plt.imshow(small1, cmap='gray')
plt.figure(2)
plt.imshow(small2, cmap='gray')
plt.show()