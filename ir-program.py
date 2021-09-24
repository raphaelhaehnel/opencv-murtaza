import cv2
import numpy as np


height = 500
width = 800

img = np.zeros((height, width, 3), np.uint8)

for i in range(width):
    cv2.circle(img, (i, 50), 1, (0, 0, 0), 5)
    cv2.imshow("Image", img)
    cv2.waitKey(1)

# for i in range(width):
#     cv2.circle(img, (20, i), 1, (255, 0, 0), 5)
#     cv2.imshow("Image", img)
#     cv2.waitKey(1)
#
# for i in range(width):
#     cv2.circle(img, (i, i), 1, (0, 255, 0), 5)
#     cv2.imshow("Image", img)
#     cv2.waitKey(1)
#
# for i in range(width):
#     cv2.circle(img, (i, i), 1, (255, 255, 255), 5)
#     cv2.imshow("Image", img)
#     cv2.waitKey(1)

cv2.imshow("Image", img)
cv2.waitKey(0)