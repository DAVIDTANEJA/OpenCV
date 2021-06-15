# Image has : foreground(width, height, edge, corners) , background  and we have to detect these features.
# methods :  cornerHarris() detection  ,  shi-tomasi corner detection  "goodFeaturesToTrack()"

# 1.cornerHarris() detection :  parameters are :
# 'img' Input image should be grayscale and float32 type.
# blockSize - It is the size of neighbourhood considered for corner detection
# ksize - Aperture parameter of Sobel derivative used.
# k - Harris detector free parameter in the equation.

import numpy as np
import cv2
from matplotlib import pyplot as plt

# img = cv2.imread('shapes.jpg')

# cv2.imshow('img', img)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = np.float32(gray)

# res= cv2.cornerHarris(gray, 2, 3, 0.04)
# res = cv2.dilate(res, None)  # if want to dilate

# img[res > 0.01 * res.max()] = [0, 0, 255]  # marked color of corner values

# cv2.imshow('dst', img)
# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()

# ---------------------------------------------------------
# 2. shi-tomasi corner detection  "goodFeaturesToTrack()"
# In this we limit the number of corners and corners quality.
# it requires int data

img = cv2.imread('shapes.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# (img, no.of corner, quality_level, min_distance between corner)
# quality_level - it will drop corners from this quality level if they are not forming accurately.
corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 20)   # 10 -find 10 corners only at 20 distance, we can increase no.
corners = np.int64(corners)

for i in corners:
    x,y = i.ravel()   # ravel() - convert multi-dim. array into 1-D
    cv2.circle(img,(x,y),3,255,-1)
cv2.imshow("res==",img)

cv2.waitKey(0)
cv2.destroyAllWindows()
