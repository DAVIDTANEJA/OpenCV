# Detect contours / corners / shapes of image  and put text on it.
import cv2
import numpy as np

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


# Detect contour / shapes  and put text
def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)    # (image, better in find outer details, get all contours)
    for cnt in contours:
        area = cv2.contourArea(cnt)    # 1st find area for each contour , To get area call the function like : getContours(canny_img)
        print(area)
        if area>500:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)    # draw the contours/shapes, (image, contour, index, blue color, thickness) 
            peri = cv2.arcLength(cnt,True)    # help get the curve/corners of image
            # print(peri)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)    # detect how many corners points / each shape corner points
            print(len(approx))    # like: 3-Tri, 4- square / rect. , 8-circle

            objCor = len(approx)    # create bounded box around detected object
            x, y, w, h = cv2.boundingRect(approx)
            if objCor ==3:
                objectType ="Tri"
            elif objCor == 4:
                aspRatio = w/float(h)    # width, height are same of square
                if aspRatio >1.0 and aspRatio <1.5:    # so define a range if its i b/w then its "Square"
                    objectType= "Square"
                else:
                    objectType="Rectangle"      # otherwise "rectangle"
            elif objCor>4:
                objectType= "Circle"
            else:
                objectType="None"

            cv2.rectangle(imgContour, (x,y), (x+w,y+h), (0,255,0), 2)    # create bounding box around each image
            # put text on image in center , (image, objecttype defined, center point -10 shift from center, text font, font scale, color, line ) 
            cv2.putText(imgContour, objectType, (x+(w//2)-10,y+(h//2)-10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,0), 2)    


img = cv2.imread('shapes.jpg')
imgContour = img.copy()        # copy of image for contours in function, to put the drawing on this image 

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(img, (7,7), 1)  # kernel size, sigma - higher sigma value will get more blur image

# Canny() - detect corners
canny_img = cv2.Canny(blur, 50,50)   # 50- threshold

getContours(canny_img)  # call the function

blank_img = np.zeros_like(img)       # black/blank image just created for array
# Now we will use  "stack images()" function, to work in real time
stack_img = stackImages(0.5, ([img, gray, blur], [canny_img, imgContour, blank_img]))   # (scale, array) , scale -to resize image

cv2.imshow("Stack", stack_img)
cv2.waitKey(0)