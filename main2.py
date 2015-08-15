__author__ = 'noam'
import FindHand as fh
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

# first_hand = fh.FindHand('./images/preprocessed/noam_left_hand_6.12.08_02062015.png', 'left')
first_hand = fh.FindHand('./images/preprocessed/noam_right_hand_19.08.08_02062015.png', 'right')

first_hand.create_map_with_pyramid()
first_hand.find_hand_elements()

print len(first_hand.fingers)

small1 = first_hand.small_image.copy()

first_hand.normalize_image_and_map_fingers()

second_hand = fh.FindHand('./images/preprocessed/noam_left_hand_30.5.15_02062015_0001.png', 'left')
# second_hand = fh.FindHand('./images/preprocessed/noam_right_hand_30.5.15_02062015.png', 'right')

second_hand.create_map_with_pyramid()
second_hand.find_hand_elements()

print len(second_hand.fingers)

# small2_negative = 255 - second_hand.small_image.copy()


second_hand.normalize_image_and_map_fingers()

first_hand.rotate_to_the_other_hand(second_hand, True)
small1_rotated = first_hand.get_range_of_interest(True,True)
small2_negative = second_hand.get_range_of_interest(False,True)
registration_transform, small1_rotated_again = fh.FindHand.optimize_registration_transform(small1_rotated.copy(), small2_negative.copy(), second_hand.palm['center'])

small1_preview = np.zeros(small1_rotated_again.shape, np.uint8)
small1_preview[small1_rotated_again == 1] = 255

small2_preview = np.zeros(small1_rotated_again.shape, np.uint8)
small2_preview[small2_negative == 1] = 255

plt.figure(1)
plt.subplot(231)
plt.imshow(small1_preview[:, :, 0], cmap='gray')
plt.title('2008 rotated type is {}'.format(first_hand.type))
plt.subplot(232)
plt.imshow(small2_preview[:, :, 0], cmap='gray')
plt.title('2015 type is {}'.format(second_hand.type))
plt.subplot(233)
plt.imshow((small2_preview - small1_preview)[:, :, 0], cmap='gray')
plt.title('diff 2015 - 2008 type is {}'.format(first_hand.type))
plt.subplot(234)
plt.imshow((small1_preview - small2_preview)[:, :, 0], cmap='gray')
plt.title('diff 2008 - 2015')
plt.subplot(235)
plt.imshow(np.abs((small1_preview - small2_preview)[:, :, 0]), cmap='gray')
plt.title('absolute diff 2008 - 2015')
plt.show()