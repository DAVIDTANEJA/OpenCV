#  Color detection and paint with it.
import cv2
import numpy as np

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10,150)

# colors in list which we want to detect, these are the HSV min.-max. values which get from TrackBar, 
# how we get : make the required color white in mask and the rest is black 
# [hue min, sat min, val min, hue max, sat max, val max]
# for new color add values in both "myColors"  and  "myColorValues"
myColors = [[5,107,0,19,255,255],     # orange
            [133,56,0,159,156,255],   # purple
            [57,76,0,100,255,255],    # green
            [90,48,0,118,255,255]]
# BGR : these are the color values for detected color like orange detected then it work as orange, similarly for green
# colorId
myColorValues = [[51,153,255],    # BGR - orange
                 [255,0,255],     # purple
                 [0,255,0],       # green
                 [255,0,0]]

myPoints =  []  # [x, y, colorId] # colorId - will get particular color and draw color at x,y point

# detect color
def findColor(img, myColors, myColorValues):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)   # convert img into HSV
    count = 0    # count colors 
    newPoints=[]
    for color in myColors:            # color - will take 1st list from 'myColors' and then loop throught all lists.
        lower = np.array(color[0:3])  # [0:3] - 1st 3 values from 1st list 'myColors'
        upper = np.array(color[3:6])  # [3:6] - values from 'myColors' 1st list
        mask = cv2.inRange(imgHSV,lower,upper)

        x,y=getContours(mask)    # get contours of color detected into mask image
        cv2.circle(imgResult, (x,y), 15, myColorValues[count], cv2.FILLED)  # and draw circle around it contours
        if x!=0 and y!=0:
            newPoints.append([x,y,count])
        count +=1
        #cv2.imshow(str(color[0]),mask)  # it show different mask windows for different colors
    return newPoints

def getContours(img):
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    x,y,w,h = 0,0,0,0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>500:
            #cv2.drawContours(imgResult, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            x, y, w, h = cv2.boundingRect(approx)
    return x+w//2,y     # get the color from tip of pen , not form center

def drawOnCanvas(myPoints,myColorValues):
    for point in myPoints:     # 'point' has (x,y) values 
        cv2.circle(imgResult, (point[0], point[1]), 10, myColorValues[point[2]], cv2.FILLED)


while True:
    success, img = cap.read()
    imgResult = img.copy()    # this will have final information on this image
    newPoints = findColor(img, myColors, myColorValues)
    if len(newPoints)!=0:
        for newP in newPoints:
            myPoints.append(newP)
    if len(myPoints)!=0:
        drawOnCanvas(myPoints,myColorValues)


    cv2.imshow("Result", imgResult)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




# -------------------------------------------------------------
# Using Webcam To detetct color for sketch pen - to paint 

# import cv2
# import numpy as np

# frameWidth = 240
# frameHeight = 240
# cap = cv2.VideoCapture(0)
# cap.set(3, frameWidth)
# cap.set(4, frameHeight)
# cap.set(10,150)

# def empty(a):
#     pass

# # create new window , both name should be same of "window and resize"
# cv2.namedWindow("TrackBars")
# cv2.resizeWindow("TrackBars", 640,240)
# cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
# cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, empty)
# cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, empty)
# cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)
# cv2.createTrackbar("Val Min", "TrackBars", 0, 255, empty)
# cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)

# while True:
#     _, img = cap.read()
#     hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
#     h_min = cv2.getTrackbarPos("Hue Min","TrackBars")  # spelling should be same (min, to which trackbar window does it belong)
#     h_max = cv2.getTrackbarPos("Hue Max","TrackBars")
#     s_min = cv2.getTrackbarPos("Sat Min","TrackBars")
#     s_max = cv2.getTrackbarPos("Sat Max","TrackBars")
#     v_min = cv2.getTrackbarPos("Val Min","TrackBars")
#     v_max = cv2.getTrackbarPos("Val Max","TrackBars")
#     # print(h_min, h_max)
#     lower = np.array([h_min, s_min, v_min])
#     upper = np.array([h_max, s_max, v_max])
#     mask = cv2.inRange(hsv_img, lower, upper)   # filter out image color

#     result_img = cv2.bitwise_and(img, img, mask=mask)
#     mask2 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

#     stack_img = np.hstack([img, mask2, result_img])  # scale, array of images 1st row, 2nd row
#     cv2.imshow('Stack Images', stack_img)           # display image


#     cv2.waitKey(1)

# Note : when run program we have :   1.img - original image shown , hsv_img, 
# 3.mask : in which we convert white-black color acc. to color ,  4.result_img : in which we see original color needed.
# we can also use  "np.hstack()"  instead of  "stackImages()" function.

