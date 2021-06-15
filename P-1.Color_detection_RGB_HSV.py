# 1. Get the RGB color values from Trackbar 
# 2. HSV - Hue saturation Value : detetct the color from images, by this we can find color intensity for which RGB not capable.

import cv2
import numpy as np

# 1. Get the RGB color values from Trackbar 
# Note : image shows color in  "RGB" format but OpenCV takes color in  "BGR" format.
# And note down RGB  values from Trackbar.
def cross(x):
    pass

img = np.zeros((200,500,3),np.uint8) # empty black image

cv2.namedWindow("Color Picker")    # Remember window name and Trackbar name should be same
cv2.createTrackbar("R","Color Picker",0,255,cross)
cv2.createTrackbar("G","Color Picker",0,255,cross)
cv2.createTrackbar("B","Color Picker",0,255,cross)

while True:
    cv2.imshow("Color Picker",img)
    k = cv2.waitKey(1) & 0xFF
    if k==27: #for exit
        break
    
    r = cv2.getTrackbarPos("R","Color Picker")
    g = cv2.getTrackbarPos("G","Color Picker")
    b = cv2.getTrackbarPos("B","Color Picker")
   
    img[:] = [r,g,b]

cv2.destroyAllWindows()

# -----------------------------------------------------------------------------
# 2. HSV - Hue saturation Value : detetct the color from images
# check - "2.py"  - using Trackbar find HSV values
# mask image : shows the black-white image 
# result 'bitwise_and' image : shows the color need image
# How to Use Trackbar : 1st use lowest H - 2nd upper H , similarly lowet S - upper S , similarly for V

frame = cv2.imread('abc.png')
frame = cv2.resize(frame,(600,400))

# Binding Color Trackbars with image
def nothing(x):
    pass

cv2.namedWindow("Color Adjustments")
cv2.createTrackbar("Lower_H", "Color Adjustments", 0, 255, nothing)
cv2.createTrackbar("Lower_S", "Color Adjustments", 0, 255, nothing)
cv2.createTrackbar("Lower_V", "Color Adjustments", 0, 255, nothing)
cv2.createTrackbar("Upper_H", "Color Adjustments", 255, 255, nothing)
cv2.createTrackbar("Upper_S", "Color Adjustments", 255, 255, nothing)
cv2.createTrackbar("Upper_V", "Color Adjustments", 255, 255, nothing)

while True:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    l_h = cv2.getTrackbarPos("Lower_H", "Color Adjustments")
    l_s = cv2.getTrackbarPos("Lower_S", "Color Adjustments")
    l_v = cv2.getTrackbarPos("Lower_V", "Color Adjustments")

    u_h = cv2.getTrackbarPos("Upper_H", "Color Adjustments")
    u_s = cv2.getTrackbarPos("Upper_S", "Color Adjustments")
    u_v = cv2.getTrackbarPos("Upper_V", "Color Adjustments")
    
    lower_bound = np.array([l_h, l_s, l_v])
    upper_bound = np.array([u_h, u_s, u_v])
    
    mask = cv2.inRange(hsv, lower_bound, upper_bound)    # mask image

    res = cv2.bitwise_and(frame, frame, mask=mask)    # result image - "bitwise_and()"

    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("res", res)

    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()

# ------------------------------------------------------------
# we can also detect colors using  "Webcam" / "Video"
# same code 
cap = cv2.VideoCapture(0)    # can pass any video

while True:
    _,frame = cap.read()
    frame = cv2.resize(frame,(400,400))

    # same from here like above
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
