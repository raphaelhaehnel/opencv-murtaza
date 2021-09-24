import cv2
import numpy as np


def webcam_detection():
    faceCascade = cv2.CascadeClassifier("resources/haarcascades/haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)

    # Define the width which is index 3 at 640
    cap.set(3, 640)

    # Define the height which is index 4 at 480
    cap.set(3, 480)

    # Define the brightness which is index 10 at 100
    cap.set(10, 100)

    while True:
        success, img = cap.read()

        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Video", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def face_detection():
    img = cv2.imread("resources/lena.png")
    faceCascade = cv2.CascadeClassifier("resources/haarcascades/haarcascade_frontalface_default.xml")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 2)

    cv2.imshow("Output", img)
    cv2.waitKey(0)


def getContours(img, imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2. arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            objCor = len(approx)

            # Now we create a bounding box around our detected object
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x, y), (x+w, y+h), (0,255,0), 2)

            if objCor == 3:
                objectType = "Tri"
            elif objCor == 4:
                aspRatio = w/h
                if w/h > 0.95 and w/h < 1.05:
                    objectType = "Square"
                else:
                    objectType = "Rec"
            elif objCor > 4:
                objectType = "Circle"
            else:
                objectType = " "
            cv2.putText(imgContour, objectType, (x+(w//2)-10, y+(h//2)-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 100, 20), 2)


def detect_shape():
    img = cv2.imread("resources/shapes.png")
    imgContour = img.copy()

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7,7), 1)
    imgCanny = cv2.Canny(imgBlur, 50, 50)
    getContours(imgCanny, imgContour)

    imgBlank = np.zeros_like(img)
    imgStack = stackImages(0.6, ([img, imgBlur, imgCanny], [imgContour, imgBlank, imgBlank]))
    cv2.imshow("Output", imgStack)
    cv2.waitKey(0)


def empty(a):
    pass


def detect_color():
    cv2.namedWindow("TrackBars")
    cv2.resizeWindow("TrackBars", 640, 240)

    cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
    cv2.createTrackbar("Hue Max", "TrackBars", 19, 179, empty)
    cv2.createTrackbar("Sat Min", "TrackBars", 110, 255, empty)
    cv2.createTrackbar("Sat Max", "TrackBars", 240, 255, empty)
    cv2.createTrackbar("Val Min", "TrackBars", 153, 255, empty)
    cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)

    while True:
        h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
        h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
        s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
        s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
        v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
        v_max = cv2.getTrackbarPos("Val Max", "TrackBars")

        img = cv2.imread("resources/lambo.png")
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(imgHSV, lower, upper)

        imgResult = cv2.bitwise_and(img, img, mask=mask)

        imgStack = stackImages(0.6, ([img, imgHSV], [mask, imgResult]))
        cv2.imshow("Stacked images", imgStack)
        cv2.waitKey(1)


def stack_images():
    img = cv2.imread("resources/lena.png")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgStack = stackImages(0.5, ([img, imgGray, img], [img, img, img]))
    cv2.imshow("output", imgStack)
    cv2.waitKey(0)


def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


def join_images():
    img = cv2.imread("resources/lena.png")

    # Horizontal stack function
    imgHor = np.hstack((img, img))

    # Vertical stack function
    imgVer = np.vstack((imgHor, imgHor))
    cv2.imshow("output", imgVer)
    cv2.waitKey(0)


def warp_perspective():
    img = cv2.imread("resources/cards.jpg")
    width, height = 250, 350

    pts1 = np.float32([[111,219],[287,188],[154,482],[352,440]])
    pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (width, height))
    cv2.imshow("Image", img)
    cv2.imshow("Output", imgOutput)
    cv2.waitKey(0)


def draw_shapes():
    img = np.zeros((512, 512, 3), np.uint8)  # unsigned integer 258 levels

    img[200:202, 0:512] = (0, 150, 255)  # BGR
    cv2.line(img, (0,0), (img.shape[1], img.shape[0]),(0, 255, 0), thickness=3)
    cv2.rectangle(img, (0,0), (250,350), (100, 2, 100), 2)
    cv2.circle(img, (400, 50), 30, (255, 255, 0), 5)
    cv2.putText(img, "OpenCV", (300, 200), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 150, 0), 4)

    cv2.imshow("Image", img)
    print(img.shape)
    cv2.waitKey(0)


def resize_img():
    img = cv2.imread("resources/lambo.png")

    imgShape = img.shape  # (width, height)

    width = int(np.floor(imgShape[1]/1.5))
    height = int(np.floor(imgShape[0]/1.5))

    imgResize = cv2.resize(img, (width, height))

    imgCropped = imgResize[0:200, 200:500]

    cv2.imshow("Resized", imgResize)
    cv2.imshow("Cropped", imgCropped)
    cv2.waitKey(0)


def img_to_blurred():
    img = cv2.imread("resources/lena.png")
    kernel = np.ones((5,5), np.uint8)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # The kernel has to be odd numbers
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)

    # The edge detector is known as Canny detector
    imgCanny = cv2.Canny(img, 150, 150)

    # Increase the thickness of our edge
    imgDialation = cv2.dilate(imgCanny, kernel, iterations=1)

    # Make the lines thiner
    imgEroded = cv2.erode(imgDialation, kernel, iterations=1)

    cv2.imshow("Gray image", imgGray)
    cv2.imshow("Blur image", imgBlur)
    cv2.imshow("Canny image", imgCanny)
    cv2.imshow("Dialation image", imgDialation)
    cv2.imshow("Erored image", imgEroded)
    cv2.waitKey(0)


def img_to_grayscale():
    img = cv2.imread("resources/lena.png")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray image", imgGray)
    cv2.waitKey(0)


def use_webcam():
    cap = cv2.VideoCapture(0)

    # Define the width which is index 3 at 640
    cap.set(3, 640)

    # Define the height which is index 4 at 480
    cap.set(3, 480)

    # Define the brightness which is index 10 at 100
    cap.set(10, 100)

    while True:
        success, img = cap.read()
        cv2.imshow("Video", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def import_video():
    cap = cv2.VideoCapture("resources/vid.mp4")
    while True:
        success, img = cap.read()
        cv2.imshow("Video", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def import_image():
    """
    Import an image and show him
    :return: None
    """
    img = cv2.imread("resources/lena.png")

    cv2.imshow("Output", img)
    cv2.waitKey(0)


if __name__ == '__main__':
    print("Package imported ")
    detect_shape()
