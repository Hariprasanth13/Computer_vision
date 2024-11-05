import cv2
import numpy as np
import imutils
import os

image = cv2.imread("C:\\Projects\\ArrowDetection\\images\\page1.jpg")
r = 800.0 / image.shape[0]
dim = (int(image.shape[1] * r), 800)
# perform the resizing
resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

cv2.imshow("image", resized)
cv2.waitKey(0)