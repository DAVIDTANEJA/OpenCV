import cv2
import numpy as np
from matplotlib import pyplot as plt


# Morphological Transformations : are some simple operations based on the image shape.
# It is normally performed on binary images. two inputs : 1. original image, 2. structuring element(kernel).
# Morphological Transformations : 1. Erosion , 2. Dilation , 3.Opening (performs 1st Erosion then 2nd Dilation), 4.Closing(1st Dilation then 2nd Erosion)
# Threshold : we define a range acc. to color we required, like : (150, 255) , so b/w 150-255  its convert into white pixels and 150 below it coverts into black pixels.
# 'Thresholding' is like 'Masking' - e convert required part of image/pixels into white color and not rquired into black color.

# 1.Erosion :
# it erodes away the boundaries of foreground object
#kernal slides through all the image and all the pixel , from the original image consider 1 only if kernal's pixel is 1
img = cv2.imread('Data\\col_balls.jpg',0)
_,mask= cv2.threshold(img,230,255,cv2.THRESH_BINARY_INV)
kernel = np.ones((2,2),np.uint8)# 5x5 kernel with full of ones. 
e = cv2.erode(mask,kernel) #optional parameters   iterations = 2
cv2.imshow("img",img) 
cv2.imshow("ker=",kernel)
cv2.imshow("mask==",mask)
cv2.imshow("erosion==",e)
 
# 2.Dilation : 
# It is just opposite of erosion. Here, a pixel element is ‘1’ if atleast one pixel under the kernel is ‘1’
# So it include the white region in the image or size of foreground object in.
# Normally, in cases like noise removal, erosion is followed by dilation. Because, erosion removes white noises, but it also shrinks our object. 
kernel = np.ones((3,3),np.uint8)# 5x5 kernel with full of ones.  
d = cv2.dilate(mask,kernel) #iterations = 2 (optional parameters) iterations = 2
cv2.imshow("dilate==",d)

# 3.Opening :
# Opening perform both first Erosion then Dilation
img = cv2.imread('Data\\col_balls.jpg',0)
_,mask= cv2.threshold(img,230,255,cv2.THRESH_BINARY_INV)
kernel = np.ones((3,3),np.uint8)# 5x5 kernel with full of ones. 
o = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)  # optional parameters 'iterations = 2'
cv2.imshow("img",img) 
cv2.imshow("ker=",kernel)
cv2.imshow("mask==",mask)
cv2.imshow("opening==",o)

# 4.Closing :
# It is opposite of opening , Closing also performs both but first Dilation then Erosion
kernel = np.ones((3,3),np.uint8)# 5x5 kernel with full of ones. 
c= cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # optional parameters 'iterations = 2'
cv2.imshow("closing",c)

# using matplotlib to display images /  "stack images()" in "2.py"
titles = ["img","mask","erosion","dilation"]
images = [img,mask,e,d]
for i in range(4):
    plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')    # 2x2 - (2 rows, 2 columns) 
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()


# -----------------------------------
# All Morphological Transformations :
img = cv2.imread('Data\\girl.jpg',0)
img = cv2.resize(img,(300,300))
_,mask= cv2.threshold(img,220,255,cv2.THRESH_BINARY_INV)
kernel = np.ones((2,2),np.uint8)# 5x5 kernel with full of ones. 

# operations
e = cv2.erode(mask,kernel) 
d = cv2.dilate(mask,kernel)
o = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
c = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
x1 = cv2.morphologyEx(mask,cv2.MORPH_TOPHAT,kernel)    # cv2.MORPH_TOPHAT : contains diff b/w mask and opening
x2 = cv2.morphologyEx(mask,cv2.MORPH_GRADIENT,kernel)  # cv2.MORPH_GRADIENT : contains diff b/w dilation and erosion
x3 = cv2.morphologyEx(mask,cv2.MORPH_BLACKHAT,kernel)  # cv2.MORPH_BLACKHAT : contains diff. b/w 'Closing of input image' and 'input image'.

cv2.imshow("img",img) 
cv2.imshow("mask==",mask)
cv2.imshow("erosion==",e)
cv2.imshow("dilate==",d)
cv2.imshow("opening==",o)
cv2.imshow("closing",c) 
cv2.imshow("x1",x1) 
cv2.imshow("x2",x2) 
cv2.imshow("x3",x3) 

# plot it
titles = ['image', 'mask', 'erosion', 'dilation', 'opening', 'closing', 'x1', 'x2',"x3"]
images = [img,mask,e,d,o,c,x1,x2,x3]
for i in range(9):
    plt.subplot(3, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

# --------------------------------------------------------------------------------------------
# Image Smoothening :
# for blurring image , noise remove.
# Types : 1.LOW Pass Filter (LPS) : remove Noise from images , 2.High Pass Filter : edge detection.
# filters : homogeneous, blur(averaging), homogeneous, median, bilateral.

img = cv2.imread("noisy.jpg")    # noisy image distorted pixels
img = cv2.resize(img,(400,400))

cv2.imshow("original==",img)
kernel = np.ones((5,5),np.float32)/25 # define a kernel for homogeneous function

# 1st filter : homogenous / 2D convolution matrix
# Its work like: each output pixel is the mean of its kernel neigbours , in this all pixel contribute with equal weight.
# kernel is a small shape or matrix which we apply on image. in "kernel is : [(1/kernal(h,w))*kernal] ".
h_filter = cv2.filter2D(img, -1, kernel) # -1 / -2 is desired depth
cv2.imshow("homogeneous==",h_filter)

# 2. blur method (or averaging) filter :
# takes the avg of all the pixels under kernel area and replaces the central element with this average..
blur = cv2.blur(img,(8,8))  # here we have image and kernel as parameter
cv2.imshow("blur==",blur)

# 3.Gaussian Filter : it uses different weight kernel, in row as well as in column.
# It means side values are small then centre. It manage distance b/w value of pixels.
gau= cv2.GaussianBlur(img,(5,5),0) # 0 -is sigma x value
cv2.imshow("gau blur=",gau)

# 4.Median Filter : 
# computes the median of all the pixels under the kernel window and the central pixel is replaced with this median value.
# This is highly effective in removing 'salt-and-pepper noise' (white dots in black image). Here kernal size must be odd except one
med = cv2.medianBlur(img,5)
cv2.imshow("median filter",med)

# 5.Bilateral filter : is highly effective at noise removal while preserving edges.
# It work like gaussian filter but more focus on edges , it is slow as compare with other filters
bi_f = cv2.bilateralFilter(img, 9, 75, 75)    # (img, neigbour_pixel_diameter, sigma_color, sigma_space)
cv2.imshow("bi_f",bi_f)

# Now plot all the images on graph
titles = ["original_image","homo","blur","gauss","med","bi_f"]
images = [img,h_filter,blur,gau,med,bi_f]
for i in range(6):
    plt.subplot(2, 3, i+1), 
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()


# -----------------------------------------------------------------------------------------------------------------------------------
# Image Gradient - Laplacian Derivatives , SobelX and SobelY  :  detetct edges
# It is a directional change in the color or intensity in an image, use to find inormation from image

# load image into gray scale
img = cv2.imread("Data\\page.jpg")
img = cv2.resize(img,(400,300))
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# 1.Laplacian Derivative : It calculate laplace derivate
lap = cv2.Laplacian(img_gray, cv2.CV_64F, ksize=3)  # (img, data_type for -ve val, ksize) # kernel size should be odd
lap = np.uint8(np.absolute(lap))   # it will remove -ve values

# 2.Sobel operation :  is a joint Gausssian smoothing plus differentiation operation, 
# so it is more  resistant to noise, This is used for 'X' and 'Y' both
# Sobel X : focus on vertical lines
# Sobel y : focus on horizontal lines
# (img, type for -ve val, x=1, y=0, ksize)
sobelX = cv2.Sobel(img_gray,cv2.CV_64F, 1, 0, ksize = 3) # (x,y) here (1,0) means 1 for - x direction
sobelX = np.uint8(np.absolute(sobelX))  # absolute : remove -ve values 

sobelY = cv2.Sobel(img_gray,cv2.CV_64F, 0, 1, ksize = 3) # here (0,1) 1: means - y direction
sobelY = np.uint8(np.absolute(sobelY))

# Now combine 'sobelX' and 'sobelY' together
sobelcombine = cv2.bitwise_or(sobelX,sobelY)

cv2.imshow("original==",img)
cv2.imshow("gray====",img_gray)
cv2.imshow("Laplacian==",lap)
cv2.imshow("SobelX===",sobelX)
cv2.imshow("SobelY==",sobelY)
cv2.imshow("COmbined image==",sobelcombine)

# Now plot all the images on graph
titles = ["original","gray","laplacian","sobelX","sobelY","combined"]
images = [img,img_gray,lap,sobelX,sobelY,sobelcombine]
for i in range(6):
    plt.subplot(3,2, i+1), 
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()


# ----------------------------------------------------------------------------------------------------
# "Canny" Edge Detection
# It uses  multi-stage algorithm to detect a edges. 5 steps :
# 1.NOise reduction(gauss) , 2. Gradient calculation , 3. Non-maximum suppresson, 4. Double Threshold , 5. Edge Tracking by Hysteresis 

img = cv2.imread("Data\\building.jpg")
img = cv2.resize(img,(600,700))
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

canny = cv2.Canny(img_gray,20,150)   # (img, thresh1, thres2)  -thresh 1 and thresh2 at different lvl
cv2.imshow("original==",img)
cv2.imshow("gray====",img_gray)
cv2.imshow("canny==",canny)

# Using  Trackbar with canny edge
def nothing(x):
    pass

cv2.namedWindow("Canny")
cv2.createTrackbar("Threshold", "Canny", 0, 255, nothing)
while True:
    a= cv2.getTrackbarPos('Threshold','Canny')  # gets the value for 'a'
    # print(a)
    res = cv2.Canny(img_gray,a,255)
    cv2.imshow("Canny",res)
    k=cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

# ----------------------------------------------------------------------------------------------------
# Image Pyramid
# sometimes we work on same imge but different resolution. e.g. searching face, eye in an image
# and it vary image to image, so in this case, we create a set of images with different resolution which is called pyramid.
# We also use  'pyramids' to blends the images. WE can also make small to bigger images and vice-versa.
# Types :  1. Gaussian Pyramid  and  2. Laplacian Pyramids

img = cv2.imread("Data\\avengers.jpg")
img = cv2.resize(img,(700,700))

# Gaussian Pyramid : 2 functions : 1. cv2.pyrUp() , 2. cv2.pyrDown()
pd1 = cv2.pyrDown(img)  # pyrdown() : reduce the resolution 1/4
pd2 = cv2.pyrDown(pd1)

# if we pyrup any pyrdown image both are not equal
pu1 = cv2.pyrUp(pd2)    # pyrup() : increase resolutio 4 times -  as opposite to pyrdown

cv2.imshow("original==",img)
cv2.imshow("pd1==",pd1)
cv2.imshow("pd2==",pd2)
cv2.imshow("pu1==",pu1)

# using loop to generate pyramid
img = cv2.imread("Data\\avengers.jpg")
img = cv2.resize(img,(700,700))
img1 = img.copy()
data = []
# pyrDown()
for i in range(4):
    img1 = cv2.pyrDown(img1)
    data.append(img1)
    cv2.imshow("res"+str(i), img1)


# ----------------------------------------------------------------------------------------------------
# Image Contours 
# Contours - a curve joining all the continuous points (along the boundary), having same color or intensity. 
# The contours are a useful tool for shape analysis and object detection and recognition
# For better accuracy, use binary images and also apply edge detection before finding countours.
# findCountour function manipulate original imge so copy it before proceeding. findContour is like finding white object from black background.
# so you must turn image in white and background is black.
# 1st always set threshold acc. to contours, less no. of contour accurate image detection.

# Some methods to remove Noise from image : 
# we also use  "mediaBlur()" to remove noise from image. Also dilate()
# Note : Threshold : we define a range acc. to color we required, like : (150, 255) , so b/w 150-255  its convert into white pixels and 150 below it coverts into black pixels.
# 'Thresholding' is like 'Masking' - we convert required part of image/pixels into white color and not rquired into black color.

img = cv2.imread("Data\\logo.jpg")
img1 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(img1,127,255,0)   # threshold is must to find contours

# find contour
cnts,hier = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  # (img, contour_retrival_mode, method)
# cnts : is a list of contours. And each contour is an array with x, y cordinates
# hier : variable called hierarchy and it contain image information.
print("Number of contour==",cnts,"\ntotal contour==",len(cnts))
print("Hierarchy==\n",hier)

# draw contours, -1 draws all contours we can also pass inplace of it particular no. of contour
img = cv2.drawContours(img, cnts, -1, (176,10,15), 4)  # (img, cnts, id of contour, color, thickness)

cv2.imshow("original===",img)
cv2.imshow("gray==",img1)
cv2.imshow("thresh==",thresh)

# --------------------
# Contours Functions : 
# 1.Moment : an image moment is a certain particular weighted average (moment) of the image pixels, or find the center / wighted avg. of contour. finds : Area, peri., center
# 2.Approximation : it is used to approximate shape with less number of vertices
# 3.ConvexHull : used to provide proper contours convexity. 'returnPoints=False' - it will return convex hull cordinates 

img = cv2.imread("shapes.png")
img = cv2.resize(img, (600,700))
img1 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(img1,127,255, cv2.THRESH_BINARY_INV)   # threshold is must to find contours

# find contour
cnts,hier = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)    # (img, contour_retrival_mode, method)
print("Number of contour==",cnts,"\ntotal contour==",len(cnts))
print("Hierarchy==\n",hier)

# # 1.Moment and draw contours :
# for c in cnts:
#     # compute the center of the contour
#     M = cv2.moments(c)
#     # print("M==",M)
#     cX = int(M["m10"] / M["m00"])    # find the center 
#     cY = int(M["m01"] / M["m00"])
#     # draw the contour and center of the shape on the image
#     cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
#     cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1)  # 7-radius
#     cv2.putText(img, "center", (cX - 20, cY - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# # display images/contours
# cv2.imshow("original===",img)
# cv2.imshow("gray==",img1)
# cv2.imshow("thresh==",thresh)


# 2.Approximation  and  3.Convex hull, draw contours , also using Moment
area1 = []
for c in cnts:
    # Moment : an image moment is a certain particular weighted average (moment) of the image pixels
    M = cv2.moments(c)
    #print("M==",M)
    cX = int(M["m10"] / M["m00"])    # find the center
    cY = int(M["m01"] / M["m00"])

    #find area of contour
    area = cv2.contourArea(c)
    area1.append(area)
    
    if area < 10000:
        # Approximation : it is used to approximate shape with less number of vertices
        epsilon = 0.1*cv2.arcLength(c,True) #arc lenght take contour and return its perimeter
        data= cv2.approxPolyDP(c,epsilon,True)

        # Convexhull : is used to provide proper contours convexity. it captures all image contours into it.
        hull = cv2.convexHull(data)
        
        x,y,w,h = cv2.boundingRect(hull)
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(125,10,20),5)

    # draw the contours and center of the shape on the image
    cv2.drawContours(img, [c], -1, (50, 100, 50), 2)
    cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1)
    cv2.putText(img, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# display images/contours
cv2.imshow("original===",img)
cv2.imshow("gray==",img1)
cv2.imshow("thresh==",thresh)

# -------------------
# Hand contour Detection : 1st always set threshold acc. to contours, less no. of contour accurate image detection.
# Convex hull : it surrounds all contours in detected image. , 'returnPoints=False' - it will return convex hull cordinates 
# Convexity defect : basically shows points where contours/line break , returns an array which contain values:  [start_point, end_point, farthest_point, approximate_distance to farthest point]
# Extreme points (end points) : It means topmost, bottommost, rightmost and leftmost points of the object.

img = cv2.imread("Data\\hand.jpg")
img = cv2.resize(img,(600,700))
img1 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

blur = cv2.medianBlur(img1,11)
ret,thresh = cv2.threshold(blur,240,255,cv2.THRESH_BINARY_INV)
# dilata = cv2.dilate(thresh,(1,1),iterations = 6)

# find contours
cnts,hier = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

for c in cnts:
    epsilon = 0.0001*cv2.arcLength(c,True)
    data= cv2.approxPolyDP(c,epsilon,True)    
    hull = cv2.convexHull(data)
    cv2.drawContours(img, [c], -1, (50, 50, 150), 2)
    cv2.drawContours(img, [hull], -1, (0, 255, 0), 2)

# Convexity defect : returns an array which contain value  [start_point, end_point, farthest_point, approximate_distance to farthest point]
hull2 = cv2.convexHull(cnts[0],returnPoints = False)
defect = cv2.convexityDefects(cnts[0],hull2)              # convexity defect
# draw defects
for i in range(defect.shape[0]):
    s,e,f,d = defect[i,0]
    print(s,e,f,d)
    start = tuple(c[s][0])
    end = tuple(c[e][0])
    far = tuple(c[f][0])
    #cv2.line(img,start,end,[255,0,0],2)
    cv2.circle(img,far,5,[0,0,255],-1)


# Extreme Points (end points) : It means topmost, bottommost, rightmost and leftmost points of the object.
c_max = max(cnts, key=cv2.contourArea)
# determine the most extreme points along the contour
extLeft = tuple(c_max[c_max[:, :, 0].argmin()][0])
extRight = tuple(c_max[c_max[:, :, 0].argmax()][0])
extTop = tuple(c_max[c_max[:, :, 1].argmin()][0])
extBot = tuple(c_max[c_max[:, :, 1].argmax()][0])

# draw the outline of the object, then draw each of the extreme points, 
# where : left-most - red , right-most - green , top-most - blue , bottom-most - teal
cv2.circle(img, extLeft, 8, (255, 0, 255), -1)   # pink
cv2.circle(img, extRight, 8, (0, 125, 255), -1)  # brown
cv2.circle(img, extTop, 8, (255, 10, 0), -1)     # blue
cv2.circle(img, extBot, 8, (19, 152, 152), -1)   # green

cv2.imshow("original===",img)
cv2.imshow("gray==",img1)
cv2.imshow("thresh==",thresh)

# -------------------------------------------
# Contours Detection using Webcam

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

def nothing(x):
    pass

cv2.namedWindow("Color Adjustments",cv2.WINDOW_NORMAL)
cv2.resizeWindow("Color Adjustments", (300, 300)) 
# Trackbar - detect color, Threshold , HSV
cv2.createTrackbar("Thresh", "Color Adjustments", 0, 255, nothing)  # Threshold
cv2.createTrackbar("Lower_H", "Color Adjustments", 0, 255, nothing)  # then all for HSV
cv2.createTrackbar("Lower_S", "Color Adjustments", 0, 255, nothing)
cv2.createTrackbar("Lower_V", "Color Adjustments", 0, 255, nothing)
cv2.createTrackbar("Upper_H", "Color Adjustments", 255, 255, nothing)
cv2.createTrackbar("Upper_S", "Color Adjustments", 255, 255, nothing)
cv2.createTrackbar("Upper_V", "Color Adjustments", 255, 255, nothing)


while True:
    _,frame = cap.read()
    frame = cv2.resize(frame,(400,400))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("Lower_H", "Color Adjustments")
    l_s = cv2.getTrackbarPos("Lower_S", "Color Adjustments")
    l_v = cv2.getTrackbarPos("Lower_V", "Color Adjustments")

    u_h = cv2.getTrackbarPos("Upper_H", "Color Adjustments")
    u_s = cv2.getTrackbarPos("Upper_S", "Color Adjustments")
    u_v = cv2.getTrackbarPos("Upper_V", "Color Adjustments")
    
    lower_bound = np.array([l_h, l_s, l_v])
    upper_bound = np.array([u_h, u_s, u_v])
    
    mask = cv2.inRange(hsv, lower_bound, upper_bound)    # mask
    filtr = cv2.bitwise_and(frame, frame, mask=mask)     # filter mask with image

    mask1  = cv2.bitwise_not(mask)
    m_g = cv2.getTrackbarPos("Thresh", "Color Adjustments")  # getting track bar value
    ret,thresh = cv2.threshold(mask1, m_g, 255, cv2.THRESH_BINARY)
    dilata = cv2.dilate(thresh,(1,1),iterations = 6)    # if image is distorted so dilate.

    # find contour(img, contour_retrival_mode, method)
    cnts,hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 'cnts' -list of contours and each contour is an array with x, y cordinate., 'hier' -contain image information.
    # print("Number of contour==",cnts,"\ntotal contour==",len(cnts))
    # print("Hierarchy==\n",hier)

    # frame = cv2.drawContours(frame,cnts,-1,(176,10,15),4)    # Draw the contours

    # loop over the contours
    for c in cnts:
        epsilon = 0.0001*cv2.arcLength(c,True)
        data= cv2.approxPolyDP(c,epsilon,True)
    
        hull = cv2.convexHull(data)
        cv2.drawContours(frame, [c], -1, (50, 50, 150), 2)
        cv2.drawContours(frame, [hull], -1, (0, 255, 0), 2)

        # hull = cv2.convexHull(data, returnPoints = False)
        # defect = cv2.convexityDefects(data[0],hull)
        # print("defect==",defect)

    cv2.imshow("Thresh", thresh)
    cv2.imshow("mask==",mask)
    cv2.imshow("filter==",filtr)
    cv2.imshow("Result", frame)

    key = cv2.waitKey(25) &0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

# ----------------------------------------------------------------------------------------------------
# Image Histogram
# It gives you an overall idea about the intensity distribution of an image. 
# It distribute data along x and y axis. x-axis contain range of color vlaues. y-axis contain numbers of pixels in an image.
# With histogram to extract information about contrast, brigthness and intensity etc.
# Plot : shows / goes from 0 to 255 (black to white) in 'x' and shows highest no. of pixels in 'y'

# plotting with calhist() method
img = np.zeros((200,200), np.uint8)
cv2.rectangle(img, (0, 100), (200, 200), (255), -1)
cv2.rectangle(img, (0, 50), (50, 100), (127), -1)

hist = cv2.calcHist([img], [0], None, [256], [0, 256])    # ([img], [channel], mask, [histsize], range[0-255]).

plt.plot(hist)
plt.show()

cv2.imshow("res",img)

# with image
img = cv2.imread("Data\\thor.jpg")
img = cv2.resize(img,(500,650))
b, g, r = cv2.split(img)
cv2.imshow("img", img)
cv2.imshow("b", b)
cv2.imshow("g", g)
cv2.imshow("r", r)
#Plotting different channel with hist
plt.hist(b.ravel(), 256, [0, 256])
plt.hist(g.ravel(), 256, [0, 256])
plt.hist(r.ravel(), 256, [0, 256])
plt.title("ColorFull Image")
plt.show()

#cal
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.title("ColorFull Image")
plt.plot(hist)
plt.show()

#Gray scale
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
plt.plot(hist)
plt.title("Gray Image")
plt.show()

# Histogram equalization is good when  of the image is confined to a particular region. It accept gray scale image
equ = cv2.equalizeHist(img_gray)
res = np.hstack((img_gray,equ)) #stacking images side-by-side
cv2.imshow("equ",res)
hist1 = cv2.calcHist([equ], [0], None, [256], [0, 256])
plt.plot(hist1)
plt.title("Equalization")
plt.show()

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
# It is used to enchance image and also handle noise froom image region.
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img_gray)
cv2.imshow('clahe',cl1)
hist2 = cv2.calcHist([cl1], [0], None, [256], [0, 256])
plt.plot(hist2)
plt.title("CLAHE")
plt.show()

# ----------------------------------------------------------------------------------------------------
# Image Back Projection : remove background from image - using HSV.
original_image = cv2.imread("Data\\green.jpg")             # 1st we take a image
original_image = cv2.resize(original_image,(600,650))
hsv_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

roi = cv2.imread("Data\\g.jpg")    # we take part of image which we want to remove from 1st image
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# 1.create Histogram of 2nd hsv_roi image , then create mask using - "calcBackProject()" it help in removing bg from image.
# (img, color space, mask None, Hist. size 180 to 256, Hist. range x '0-180' y '0-256')
roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
mask = cv2.calcBackProject([hsv_original], [0, 1], roi_hist, [0, 180, 0, 256], 1)  # original_img hsv , roi_hist - by this hist. it will remove bg

# Filtering remove noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.filter2D(mask, -1, kernel)
_, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

mask = cv2.merge((mask, mask, mask))
result = cv2.bitwise_or(original_image, mask)

cv2.imshow("Mask", mask)
cv2.imshow("Original image", original_image)
cv2.imshow("Result", result)
cv2.imshow("Roi", hsv_original)

# ----------------------------------------------------------------------------------------------------
# Hough Transformation : 
# detect shapes on basis of lines, circles. Also completes distorted images.
# Methods : 1.cv2.HoughLines() , 2.cv2.HoughLinesP()  P-probability , 3.HoughCircles() , 4.Using Webcam
# And lines are expressed for Hough Transform by 2 methods :
# 1.Cartesian CS(cordinate system) --> y= mx+c  and  2.Polar CS --> xcos0+ysin0

img = cv2.imread('Data\\chess.jpg')
img = cv2.resize(img,(400,400))
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,10,200,apertureSize = 3)  # 50,150

# function accept parameter , 'rho' -distance b/w resolution of pixels , 'thetha' -angle resolution b/w pixels
lines = cv2.HoughLines(edges, 1, np.pi/180,200)  # (img, rho, theta)
# we need 1st element of 'lines [0]' which contains - 'rho', 'theta'

# 1.method line threshold
for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))   # 1000 pixels backward
    y1 = int(y0 + 1000*(a))    # 1000 pixels forward
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img,(x1,y1),(x2,y2),(255,0,255),2)


# # 2.method : HoughLinesP() - probability
# lines = cv2.HoughLinesP(edges, 1, np.pi/180,100, minLineLength=8, maxLineGap=100)
# for line in lines:
#     x1,y1,x2,y2 = line[0]
#     cv2.line(img,(x1,y1),(x2,y2),(100,200,125),2)


# 3.method : HoughCircles()
img = cv2.imread('Data\\col_balls.jpg')
img2= img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5)

#(img, circle_method, dp-resolution distance per pixel, min.distance circles, parm1, parm2 [p1>p2],)
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
data = np.uint16(np.around(circles))
for (x, y ,r) in data[0, :]:
    cv2.circle(img2, (x, y), r, (50, 10, 50), 3)   # outer circle
    cv2.circle(img2, (x, y), 2, (0, 255, 100), -1) # center of circle
cv2.imshow('Result',img2)


# 4.Using Webcam : detect ball and create circle around it and center ccircle in it.
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

while True:
    _,img = cap.read()
    img2 = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 10, param1=50, param2=30, minRadius=0, maxRadius=0)
    if circles is not None:
        data = np.uint16(np.around(circles))
        for (x, y ,r) in data[0, :]:
            cv2.circle(img2, (x, y), r, (50, 10, 50), 3)    # outer circle
            cv2.circle(img2, (x, y), 2, (0, 255, 100), -1)  # center

    cv2.imshow("res",img2)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break


cv2.imshow("edge",edges)
cv2.imshow("lines",img)

# ----------------------------------------------------------------------------------------------------

cv2.waitKey(0)
cv2.destroyAllWindows()