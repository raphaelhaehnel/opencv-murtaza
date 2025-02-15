# PySnake

- This code is my implementation of the Virtual Paint and the Document Scanner, two project proposed by Murtaza Hassan from Murtaza's Workshop.
- [Link](https://www.youtube.com/watch?v=WQeoO7MI0Bs) the the Youtube video.
- For these projects, you'll need the application "IP Webcam" to get a camera stream from your phone.

### Project 1 - Virtual Paint
- This project implements a Virtual Paint system that captures video from an IP camera and applies color detection to draw on the captured images. It is designed to allow users to create virtual paintings using specific colors detected in real-time.
- The program captures an image, applies color filters using HSV (Hue, Saturation, Value), and draws shapes on the detected colors, mimicking the effect of painting.

### Project 2 - Document Scanner
- This project implements a real-time document scanner using a webcam or IP camera. It captures images, processes them to find contours, applies perspective transformation to align the document, and converts the scanned document into a clean, thresholded black-and-white image.
- The system uses OpenCV for image processing and can scan documents from live video input, making it suitable for creating digital copies of physical documents.
