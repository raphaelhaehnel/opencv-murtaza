#!/usr/bin/env python3

################################################################
# FILE : project2.py
# WRITER : Raphael Haehnel
# DESCRIPTION: This code is my implementation of the Document Scanner,
#   project proposed by Murtaza Hassan from Murtaza's Workshop
################################################################

import cv2
import numpy as np
from murtazaIntro import stackImages


CAMERA = True

def get_camera():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(3, 480)
    cap.set(10, 100)
    return cap


def getContours(img, imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2. arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            objCor = len(approx)


if __name__ == '__main__':

    if CAMERA:
        cap = get_camera()
        while True:
            success, img = cap.read()
            imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
            imgCanny = cv2.Canny(imgBlur, 50, 50)

            imgBlank = np.zeros_like(img)

            imgContours = img.copy()
            getContours(imgCanny, imgContours)

            imgStacked = stackImages(0.7, ([img, imgGray, imgCanny], [imgContours, imgBlank, imgBlank]))

            cv2.imshow("Video", imgStacked)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        img = cv2.imread("resources/lena.png")
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
        imgCanny = cv2.Canny(imgBlur, 50, 50)

        imgBlank = np.zeros_like(img)

        imgContours = img.copy()
        getContours(imgCanny, imgContours)

        imgStacked = stackImages(0.7, ([img, imgBlur, imgCanny], [imgContours, imgBlank, imgBlank]))

        cv2.imshow("Video", imgStacked)
        cv2.waitKey(0)

