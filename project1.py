#!/usr/bin/env python3

################################################################
# FILE : project1.py
# WRITER : Raphael Haehnel
# DESCRIPTION: This code is my implementation of the Virtual Paint,
#   project proposed by Murtaza Hassan from Murtaza's Workshop
################################################################

import cv2
import numpy as np
import requests

CAMERA_URL = "http://192.168.1.3:8080/shot.jpg"
DEBUG = True
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)


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

    cv2.createTrackbar("Hue Min", "TrackBars", 50, 179, empty)
    cv2.createTrackbar("Hue Max", "TrackBars", 105, 179, empty)
    cv2.createTrackbar("Sat Min", "TrackBars", 80, 255, empty)
    cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)
    cv2.createTrackbar("Val Min", "TrackBars", 0, 255, empty)
    cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)


def get_trackbar_parameters():
    """ By given an image and the parameters of the trackbar, apply a filter for the color range specified by the
    trackbar.
    :param img: The input image
    :return: The filtered image
    """

    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")

    return (h_min, h_max), (s_min, s_max), (v_min, v_max)


def apply_hsv_and(img, h, s, v):
    """ Convert the image to a hsv matrix, apply the threshold h,s,v and the bitwise AND to the new image and the
    original image
    :param img: an image
    :param h: hue threshold
    :param s: saturation threshold
    :param v: value threshold
    :return: The filtered image
    """
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([h[0], s[0], v[0]])
    upper = np.array([h[1], s[1], v[1]])
    mask = cv2.inRange(imgHSV, lower, upper)

    imgResult = cv2.bitwise_and(img, img, mask=mask)
    return imgResult


def detect_shape(img, drawer, color):
    """ Draw a circle on the image "drawer" according to the contours of the image "img"
    :param img: the image containing the contours
    :param drawer: the image on which we draw the circle
    :param color: the color of the circle
    :return: None
    """
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
    imgCanny = cv2.Canny(imgBlur, 50, 50)
    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        minLocal = min(contours, key=lambda item: item[0, 0, 1])
        minGlobal = min(minLocal, key=lambda item: item[0, 1])
        cv2.circle(drawer, (minGlobal[0, 0], minGlobal[0, 1]), 10, color, 20)


def get_camera(url):
    """ Read the image from the url
    :param url: the link to the image
    :return: The image from the url
    """
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    return cv2.imdecode(img_arr, -1)


if __name__ == '__main__':

    if DEBUG:
        # Initiate the trackbar by using global variables: Hue, Sat, Val (Min/Max)
        init_trackbar()

    # Get the first image
    img = get_camera(CAMERA_URL)

    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    dims = img.shape
    img = cv2.resize(img, (dims[1]//2, dims[0]//2))

    # Create a black image
    imgPaint = np.zeros_like(img)

    while True:

        # Get the image from Android
        img = get_camera(CAMERA_URL)

        # Rotate and resize the image
        dims = img.shape
        img = cv2.resize(img, (dims[1]//2, dims[0]//2))
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if DEBUG:
            # Apply the function inRange to the image according to the Trackbar
            h, s, v = get_trackbar_parameters()
            imgDebug = apply_hsv_and(img, h, s, v)
            cv2.imshow("Debug", imgDebug)

        # Create three images according to three specific sets of threshold
        imgb = apply_hsv_and(img, (108, 145), (130, 255), (0, 255))
        imgr = apply_hsv_and(img, (0, 16), (167, 255), (0, 255))
        imgg = apply_hsv_and(img, (70, 98), (80, 255), (0, 255))

        # Draw on the imgPaint according to the three images
        detect_shape(imgb, imgPaint, BLUE)
        detect_shape(imgg, imgPaint, GREEN)
        detect_shape(imgr, imgPaint, RED)

        imgResult = cv2.addWeighted(img, 1, imgPaint, 1, 0.0)

        cv2.imshow("Result", imgResult)
        # cv2.imshow("Paint", imgPaint)

        # Wait and exit de program is the key "q" is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
