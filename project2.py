#!/usr/bin/env python3

################################################################
# FILE : project2.py
# WRITER : Raphael Haehnel
# DESCRIPTION: This code is my implementation of the Document Scanner,
#   project proposed by Murtaza Hassan from Murtaza's Workshop
################################################################

import cv2
import numpy as np
import requests
from murtazaIntro import stackImages

CAMERA = True

KERNEL = np.ones((5, 5), np.uint8)
CAMERA_URL = "http://192.168.1.3:8080/shot.jpg"
X = 0
Y = 1
DEBUG = True


def getContours(img, imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    maxArea = 0
    biggest = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2. arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            objCor = len(approx)

            if area > maxArea and objCor == 4:
                maxArea = area
                biggest = approx

    if biggest is not None:

        if DEBUG:
            for i in range(4):
                cv2.line(imgContour, biggest[i, 0], biggest[(i+1)%4, 0], (0, 255, 0), thickness=3)
            for pt in biggest:
                cv2.circle(imgContour, (pt[0, 0], pt[0, 1]), 10, (0, 0, 255), 20)

        biggest = reorder(biggest)

    return biggest


def reorder(pts):
    pts = pts.reshape((4, 2))
    ptsNew = np.zeros((4,1,2), np.int32)
    add = pts.sum(1)
    ptsNew[0] = pts[np.argmin(add)]
    ptsNew[3] = pts[np.argmax(add)]
    diff = np.diff(pts, axis=1)
    ptsNew[1] = pts[np.argmin(diff)]
    ptsNew[2] = pts[np.argmax(diff)]
    return ptsNew


def warp_perspective(img, pts):
    """ Create an new image from the original image with the boundaries defines in "pts"
    :param img: An image
    :param pts: An array of 4 sets of points
    :return: An image with the new perspective
    """
    imgOutput = np.zeros_like(img)
    if pts is not None:
        width, height = 400, 600

        pts1 = np.float32(pts)

        # pts1 = np.float32(pts)
        pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgOutput = cv2.warpPerspective(img, matrix, (width, height))

    return imgOutput


def get_camera_android(url):
    """ Read the image from the url
    :param url: the link to the image
    :return: The image from the url
    """
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    return cv2.imdecode(img_arr, -1)


def empty():
    """ Function that does nothing and return None
    :return: None
    """
    pass


def init_trackbar():
    """ Initiate the trackbar window
    :return: None
    """
    cv2.namedWindow("TrackBars")
    cv2.resizeWindow("TrackBars", 640, 240)

    cv2.createTrackbar("Canny1", "TrackBars", 83, 500, empty)
    cv2.createTrackbar("Canny2", "TrackBars", 190, 500, empty)


def get_trackbar_parameters():
    """ By given an image and the parameters of the trackbar, apply a filter for the color range specified by the
    trackbar.
    :param img: The input image
    :return: The filtered image
    """

    n1 = cv2.getTrackbarPos("Canny1", "TrackBars")
    n2 = cv2.getTrackbarPos("Canny2", "TrackBars")
    return n1, n2


if __name__ == '__main__':

    init_trackbar()

    while True:
        img = get_camera_android(CAMERA_URL)

        # Rotate and resize the image
        dims = img.shape
        img = cv2.resize(img, (dims[1] // 2, dims[0] // 2))
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)

        n1, n2 = get_trackbar_parameters()
        imgCanny = cv2.Canny(imgBlur, n1, n1)

        # Increase the thickness of our edge
        imgDilation = cv2.dilate(imgCanny, KERNEL, iterations=1)

        # Make the lines thiner
        imgEroded = cv2.erode(imgDilation, KERNEL, iterations=1)

        # Create a black image
        imgBlank = np.zeros_like(img)

        imgContours = img.copy()
        pts = getContours(imgDilation, imgContours)
        imgDoc = warp_perspective(img, pts)

        imgDocGray = cv2.cvtColor(imgDoc, cv2.COLOR_BGR2GRAY)
        im_bw = cv2.threshold(imgDocGray, n2, 255, cv2.THRESH_BINARY)[1]

        imgStacked = stackImages(0.4, ([img, im_bw], [imgContours, imgDilation]))

        cv2.imshow("Video", imgStacked)
        cv2.imshow("Result", imgDoc)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



