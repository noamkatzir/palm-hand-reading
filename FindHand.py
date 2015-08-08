__author__ = 'noam'
import cv2
import numpy as np
from numpy import linalg
import math
from operator import itemgetter, attrgetter, methodcaller
from matplotlib import pyplot as plt


# __author__ = 'katzirn'
#
# class MatchingData:
#     def __init__(self,objectsDir,notObjectsDir):
#         self.objectsDir = objectsDir
#         self.notObjectsDir = notObjectsDir
#
#     @property
#     def objectsDir(self):
#         return self.__objectsDir
#
#     @objectsDir.setter
#     def objectsDir(self,value):
#         self.__objectsDir = value
#
#     @property
#     def notObjectsDir(self):
#         return self.__notObjectsDir
#
#     @notObjectsDir.setter
#     def notObjectsDir(self,value):
#         self.__notObjectsDir = value


class FindHand:
    def __init__(self, imagePath,handType):
        self.image = cv2.imread(imagePath)
        self.small_image = np.zeros(1)
        self.imageGray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.imageMap = np.zeros(1)
        self.handElements = []
        self.fingers = []
        self.palm = {}
        self.rotatedContours = []
        self.type = handType
        self.orientation_angle = 0

    def create_map_with_pyramid(self, pyramid_level=5, area_threshold=10):
        self.small_image = cv2.resize(self.image, (0, 0), fx=0.5 ** pyramid_level, fy=0.5 ** pyramid_level)
        small = cv2.resize(self.imageGray, (0, 0), fx=0.5 ** pyramid_level, fy=0.5 ** pyramid_level)

        # I put the threshold to get the hand elements binay image
        ret, threshold = cv2.threshold(small, 245, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.imageMap = np.zeros(threshold.shape, np.uint8)

        # I'm filling the contours of with large area to ensure I get a the objects with minimum noise
        for i in xrange(len(contours)):
            area = cv2.contourArea(contours[i])

            # in origin it was 300000, but it start ignoring fingers
            if area > area_threshold:
                cv2.drawContours(self.imageMap, contours, i, 255, -1)

    def find_hand_elements(self):
        contours, hierarchy = cv2.findContours(self.imageMap.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        self.handElements = []
        for i in xrange(len(contours)):

            # cv2.circle(lefthand, (centroid_x, centroid_y), 10, (255, 0, 0),-1)

            element = {
                'center': self._get_contour_center(contours[i]),
                'contour': contours[i],
                'area': cv2.contourArea(contours[i]),
                'palm': False,
                'passed': False
            }

            self.handElements.append(element)

        # mapp the distance of hand elements from the palm
        self.palm = max(self.handElements, key=lambda x: x['area'])
        self.palm['palm'] = True
        self.palm['dist'] = 0

        for handElement in self.handElements:
            if handElement['palm'] is False:
                handElement['dist'] = linalg.norm(np.array(handElement['center']) - np.array(self.palm['center']))

        # sorting the hand elements by the distance from the palm
        # because this way we will find the end of the fingers
        # self.handElements = sorted(self.handElements, key=itemgetter('dist'), reverse=True)
        self.handElements.sort(key=itemgetter('dist'), reverse=True)

        for handElement in self.handElements:
            if handElement['palm'] is False and handElement['passed'] is False:
                # fetching all the fingers from the image
                self.fingers.append(self._collect_finger_elements(handElement))

    def _get_contour_center(self,contour):
        M = cv2.moments(contour)
        centroid_x = M['m10'] / M['m00']
        centroid_y = M['m01'] / M['m00']
        return (centroid_x, centroid_y)

    def _collect_finger_elements(self, finger_end):
        line_mask = np.zeros(self.imageMap.shape, np.uint8)
        rows, cols = self.imageMap.shape[:2]
        [vx, vy, x, y] = cv2.fitLine(finger_end['contour'], cv2.cv.CV_DIST_L2, 0, 0.01, 0.01)
        lefty = int((-x * vy / vx) + y)
        righty = int(((cols - x) * vy / vx) + y)
        cv2.line(line_mask, (cols - 1, righty), (0, lefty), 1, 2)
        hand_element_mask = np.zeros(self.imageMap.shape, np.uint8)

        # any finger element which already passed as the flag as true, make the detection easier
        finger_elements = [finger_end]
        finger_end['passed'] = True
        for handElement in self.handElements:
            if handElement['palm'] is False and handElement['passed'] is False:
                # set zeros in the mask
                hand_element_mask.fill(0)
                cv2.drawContours(hand_element_mask, [handElement['contour']], 0, 1, -1)

                # plt.figure(1)
                # plt.imshow(line_mask, cmap='gray')
                # plt.figure(2)
                # plt.imshow(hand_element_mask, cmap='gray')
                # plt.figure(3)
                # plt.imshow(np.multiply(hand_element_mask,line_mask), cmap='gray')
                # plt.show()

                # if the line and the handElement came a cross the max value will be larger than zero
                if np.multiply(hand_element_mask, line_mask).max() > 0:
                    finger_elements.append(handElement)
                    handElement['passed'] = True

        return finger_elements

    def calculate_normalization_box_angle(self):
        rect = cv2.minAreaRect(self.palm['contour'])
        box = cv2.cv.BoxPoints(rect)
        box = np.int0(box)
        ydiff = float(box[0][1] - box[1][1])
        xdiff = float(box[0][0] - box[1][0])

        angle = self._calculate_angle(xdiff, ydiff)

        self.palm['normalization_angle'] = angle

    def _calculate_angle(self, xdiff, ydiff):
        angle = np.pi / 2
        if xdiff is not 0:
            angle = math.atan(ydiff / xdiff)
        angle = (angle / np.pi) * 180
        return angle

    def rotate_contours_according_palm_center(self, angle, padding_size=-1):
        if padding_size is -1:
            padding_size = max(self.small_image.shape[0:2]) / 4

        # adding the padding to the center because when we padding the image the center is moving
        center = self.palm['center'][0] + padding_size, self.palm['center'][1] + padding_size
        M = cv2.getRotationMatrix2D(center, angle, 1)

        homogeneous_center = np.array([self.palm['center'][0], self.palm['center'][1], 1])
        rotatedContour = np.dot(M, homogeneous_center)
        self.palm['rotatedCenter'] = rotatedContour[0],rotatedContour[1]

        for handElement in self.handElements:
            homogeneous_representation = np.ones((handElement['contour'].shape[0], 3), np.uint8)
            contour = handElement['contour'].copy().reshape((handElement['contour'].shape[0],2))
            # adding the padding to the contour points too
            homogeneous_representation[:, :2] = contour + padding_size

            rotatedContour = np.dot(M, homogeneous_representation.transpose())

            handElement['rotatedContour'] = np.int32(rotatedContour.transpose().reshape((handElement['contour'].shape[0], 1, 2)))
            handElement['rotatedCenter'] = self._get_contour_center(handElement['rotatedContour'])
            self.rotatedContours.append(handElement['rotatedContour'])

    def map_fingers_and_orientation(self):
        x,y,w,h = cv2.boundingRect(self.palm['rotatedContour'])

        end_fingers_centers = np.array([finger[0]['rotatedCenter'] for finger in self.fingers])
        order = self.type is 'right'

        if np.sum(end_fingers_centers[:, 0] >= x+w) >= 4:
            angle = 0
            self.fingers.sort(key=lambda y: y[0]['rotatedCenter'][1], reverse=(not order))
        elif np.sum(end_fingers_centers[:, 1] >= y+h) >= 4:
            angle = 90
            self.fingers.sort(key=lambda x: x[0]['rotatedCenter'][0], reverse=order)
        elif np.sum(end_fingers_centers[:, 0] <= x) >= 4:
            angle = 180
            self.fingers.sort(key=lambda y: y[0]['rotatedCenter'][1], reverse=order)
        else:
            angle = 270
            self.fingers.sort(key=lambda x: x[0]['rotatedCenter'][0], reverse=(not order))

        self.orientation_angle = angle

    def map_fingers_angles_from_palm(self):
        center = np.array(self.palm['rotatedCenter'])

        for finger in self.fingers:
            pos = center - np.array(finger[0]['rotatedCenter'])

            finger[0]['angleFromPalmCenter'] = self._calculate_angle(pos[0], pos[1])

    def rotate_image(self, image, angle):
        # padding the image before rotation
        padding_size = max(image.shape[0:2]) / 4
        image = cv2.copyMakeBorder(image, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_CONSTANT,
                                   value=[255, 255, 255])
        rows, cols = image.shape[0:2]
        center = self.palm['center'][0] + padding_size, self.palm['center'][1] + padding_size

        M = cv2.getRotationMatrix2D(center, angle, 1)
        return cv2.warpAffine(image, M, (cols, rows))