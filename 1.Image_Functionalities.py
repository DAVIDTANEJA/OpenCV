import cv2
import numpy as np


#---------------------------------------------------------------------
# Image Functionaities
#----------------------
# read the image   # 0 / cv2.IMREAD_GRAYSCALE :grayscale, -1 / cv2.IMREAD_COLOR :color , 1 / cv2.IMREAD_UNCHANGED
img = cv2.imread('abc.png', -1)
img1 = cv2.imread('abc.png', 0)
img2 = cv2.imread('abc.png', 1)

# change color of image grayscale , blur image , corner of img , 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
average = cv2.blur(img, (3,3))
blur = cv2.GaussianBlur(img, (7,7), 1)    # (7,7) - kernel size it should be odd no. (3,3) (5,5)
median = cv2.medianBlur(img, 3)
bilateral = cv2.bilateralFilter(img, 10, 35, 25)   # img, diameter, sigmacolor, sigmaspace 

# smoothness of image
smooth = cv2.edgePreservingFilter(img1, cv2.RECURS_FILTER, 200, 0.5)

# sketch effect - pencil effect , colored skecth effect
pencil, colored = cv2.pencilSketch(img1, 200, 0.1, shade_factor=0.1)


# detect corner of image
canny = cv2.Canny(img, 100, 100)       # threshold can be chnaged 150, 200
# dilation - sometime images are joined so it can not detect as proper line, so increase thickness of edges
kernel = np.ones((5,5), np.uint8)  # 5x5 matrix, type of the object , its all values are 1
dilate = cv2.dilate(canny, kernel, iterations=1)  # iterations -kernel move around image and create thickness needed, increase acc.
# erosion - opposite of dilation, increase edge of image
erode = cv2.erode(dilate, kernel, iterations=1)

# display image
cv2.imshow('gray', gray)  # cv2.imshow('blur', blur) , cv2.imshow('canny', canny) , cv2.imshow('dilation', dilate) , cv2.imshow('erode', erode)

print(img.shape)      # (rows, columns, 3d array/channels/pixels) / (height, width, channels/pixels)  channels- RGB color
img = cv2.resize(img, (200,200))  # resize  , (img, (0,0), fx=0.5, fy=0.5) -it make half of size if don't use pixel
img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)    # rotate 
img = cv2.flip(img,0)    # flip the image : 0, 1, -1

# Concatenate images : horizonal , vertical
hor = cv2.hconcat([img1, img2])
ver = cv2.vconcat([img1, img2])

# display using matplotlib
import matplotlib.pyplot as plt
img = cv2.imread('abc.png')
plt.imshow(img)
plt.show()

# ----------------
# Translation : movement of image in x-y direction
def translate(img, x, y):
    transMat = np.float32([[1,0,x],[0,1,y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv2.warpAffine(img, transMat, dimensions)
# -x = Left  ,  -y = Up  ,  x = Right  ,  y = Down
img = cv2.imread('abc.png')
translated = translate(img, -100, 100)
cv2.imshow('translated', translated)

# --------------
# create blank image 'black' color 
blank = np.zeros((400,400,3), dtype='uint8')    
cv2.imshow('black', blank)
blank[:] = 0,225,125        # can change color all over / part of it ->  blank[200:300, 300:400] = 0,225,125 
cv2.imshow('image',blank)


img = cv2.arrowedLine(img, (0,125), (255,255), (255, 0, 125), 10)

img = cv2.ellipse(img,(400,600),(100,50),0,0,180,155,5)

pts = np.array([[100,150],[200,30],[170,20],[50,10]], np.int32)
pts = pts.reshape((-1,1,2))
img = cv2.polylines(img,[pts],True,(0,255,155))


cv2.imwrite('new_abc.jpg', img)       # save / write the image

cv2.waitKey(0)    # 0 -means it will wait infinite time until we press any key and it disappear.  10 - for 10 mili sec.
cv2.destroyAllWindows()    # in the end to close opencv and all yhe windows

# --------------------------
# ROI(region of Interest) / crop image - take the image part and paste into original image some location, we can also use this in video capture.
# 1. slicing method
img = cv2.imread('abc.png', -1)
crop = img[80:400, 300:520]       # [height, width]  [rows,columns]  # take the image part
img[80:400, 100:320] = crop        # pixels must be equal what we take part of it , pasting into image here 'tag' part
cv2.imshow('Image', img)           # display image

# 2.ROI/crop image - opencv method
r = cv2.selectROI(img1)                     # select the roi part when image shows and press 'space' button
# print(r)
cropped = img1[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]  # it will take 4 points and pass here and it will crop it.
cv2.imshow('cropped', cropped)

# ---------------------------
# Draw : line, rectangle, circle, Text
img = np.zeros((512,512,3), np.uint8)   # create image 'black'
# # img[:] = 255,0,0                      # it will 'blue' image
# img[200:300 , 100:300] = 255,0,0        # [height, width] part of image colored

height = img.shape[0]
width = img.shape[1]

# line , (image, stating, ending point (width,height), color of line, width of line)
img = cv2.line(img, (0,0), (width, height), (255,0,0), 10)    # from left upper corner to right bottom    
img = cv2.rectangle(img, (100,200), (200, 300), (0,0,255), -1)  # instead of thickness if we use : '-1' / cv2.FILLED : this will fill the area with that color
img = cv2.circle(img, (300,300), 60, (0,0,255), -1)  # (image, centre point, radius, color, width of line / fill with that color if use '-1')

font = cv2.FONT_HERSHEY_SIMPLEX
img = cv2.putText(img,'David Taneja', (200, height - 10), font, 1, (0,0,255), 5, cv2.LINE_AA)  # (image, text, text place, font, font scale/thickness, color, line width)  # height-10 '-10' for padding

# warp perspective - capture particular part of image
# if want to find points,  open image in "Paint" and move cursor it will show below (height,width)points
img = cv2.imread('abc.png')    # take any card image
width, height = 250, 350    # 2.5 x 3.5 inches a card ratio
pts1 = np.float32([[111,219],[287,188],[154,482],[352,440]])   # and use points acc.
pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
output = cv2.warpPerspective(img, matrix, (width,height))
cv2.imshow("output", output)

# for particular cordinate : [125,125,0]  , 0-blue, 1-green, 2-red

# Split image
img = cv2.imread('abc.png')
b,g,r = cv2.split(img)
# cv2.imshow("blue",b)
# cv2.imshow("green",g)
# cv2.imshow("red",r)

# merge image
mr1 = cv2.merge((r,g,b))
cv2.imshow("rgb",mr1)
mr2 = cv2.merge((g,b,r))
cv2.imshow("gbr",mr2)

# create border
brdr = cv2.copyMakeBorder(img, 20, 10, 5, 5, cv2.BORDER_CONSTANT,value=(0,125,176))  # (image, top, bottom, left, right, bordertype, color value)

# Stack images / join - images hsould be of equal sizes
hor = np.hstack((img,img))    # numpy functions
ver = np.vstack((img,img))
cv2.imshow('Horizontal', hor)    # ver

# ---------------------------------------------------------------------------------------------------------------------
# Thresholding

img = cv2.imread("abc.png",0)
img = cv2.resize(img,(400,400))

# Simple Thresholding
# min : 0-50, max. 50-255
_, th1 = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)  # min. value - make it black, max. - white
_, th2 = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)  # inverse it.
_, th3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)      # truncate
_, th4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)  # black-white : for min.-max
_, th5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

cv2.imshow("1 - THRESH_BINARY",th1)
cv2.imshow("1 - THRESH_BINARY",th1)

# ---------
# Adaptive Thresholding
# it creates different no. of threshold pixel values
# cv2.ADAPTIVE_THRESH_MEAN_C : takes mean of neighbourhood area.
# cv2.ADAPTIVE_THRESH_GAUSSIAN_C : takes weighted sum of neighbourhood values where weights are a gaussian window.
_, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY) #simple 
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2) #adaptive
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2); #adaptive

cv2.imshow("data",img)
cv2.imshow("Adaptive thres mean",th1)
cv2.imshow("Adaptive Gaussian ", th2)

# ---------------------------------------------------------------------------------------------------------------------
# Bitwise operations : and , or , not , xor  , see  "truth table" for True-False condition of these operations

img1 = np.zeros((250, 500, 3), np.uint8)  # black image
img1 = cv2.rectangle(img1,(150, 100), (200, 250), (255, 255, 255), -1)  # rectangle created inside blacki mage 1

img2 = np.zeros((250, 500, 3), np.uint8)  # black image 2
img2 = cv2.rectangle(img2,(10, 10), (170, 190), (255, 255, 255), -1)  # rectangle inside image 2

cv2.imshow("img1", img1)
cv2.imshow("img2", img2)

# opertaions
bitAnd = cv2.bitwise_and(img2, img1)
bitOr = cv2.bitwise_or(img2, img1)
bitXor = cv2.bitwise_xor(img1, img2)
bitNot1 = cv2.bitwise_not(img1)
bitNot2 = cv2.bitwise_not(img2)

# cv2.imshow('bitAnd', bitAnd)
# cv2.imshow('bitOr', bitOr)
# cv2.imshow('bitXor', bitXor)
# cv2.imshow('bitNot1', bitNot1)
# cv2.imshow('bitNot2', bitNot2)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# here we put / crop 2nd image data into 1st image. Use case : roi, grayscale img, mask/ threshold, remove bg, put color into it, put roi into final image
# image 2 should be equal or less than from image 1 in pixels.

# Load two images
img1 = cv2.imread("soccer_practice.jpg")
img2 = cv2.imread("abc.png")

img1 = cv2.resize(img1,(1024,650))
img2 = cv2.resize(img2,(600,650))

# put img2 data into img1 , get the image 1 and roi part of image 2 equal
r,c,ch = img2.shape    # get image 2 - width, height
roi = img1[0:r,0:c]    # we get here part of image 1 acc. to 'r','c' width, height

img_gry = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)    # grayscale

_, mask = cv2.threshold(img_gry, 100, 125, cv2.THRESH_BINARY)   # threshold
#remove bg
mask_inv= cv2.bitwise_not(mask)
#put mask into roi
img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

# Take only region of figure from original  image.
img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
# Put logo in ROI and modify the main image
res = cv2.add(img1_bg,img2_fg)

final = img1
final[0:r,0:c]= res   #final output

# cv2.imshow('image 1', img1)
# cv2.imshow('image 2', img2)
# cv2.imshow('roi', roi)
cv2.imshow('image 1', final)

# ---------------------------------------------------------------------------------------------------------------------
# Corner detection : before passing to algorithms convert images into Grayscale so it works properly.

img = cv2.imread('abc.png')    # if image is large resize it, cv2.resize(img, (0,0), fx=0.75, fy=0.75)    # 0.50
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# algorithm : Shi-tomasi corner detector
corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)   # (img, 100 -no. of corners, min. quality 0 to 1, min euclidean distance b/w 2 corners)
# convert corners into integers
corners = np.int0(corners)

for corner in corners:
    x,y = corner.ravel()
    cv2.circle(img, (x,y), 5, (255,255,255), -1)

for i in range(len(corners)):
    for j in range(i+1, len(corners)):
        corner1 = tuple(corners[i][0])
        corner2 = tuple(corners[j][0])
        color = tuple(map(lambda x: int(x), np.random.randint(0, 255, size=3)))
        cv2.line(img, corner1, corner2, color, 1)

cv2.imshow('frame', img)

# ---------------------------------------------------------------------------------------------------------------------
# Template matching / Object detetction in image
img = cv2.resize(cv2.imread('soccer_practice.jpg', 0), (0, 0), fx=0.8, fy=0.8)
template = cv2.resize(cv2.imread('shoe.png', 0), (0, 0), fx=0.8, fy=0.8)
h, w = template.shape

# different methods for detection, go through every method which gives better result use it.
methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

for method in methods:
    img2 = img.copy()

    result = cv2.matchTemplate(img2, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        location = min_loc
    else:
        location = max_loc

    bottom_right = (location[0] + w, location[1] + h)    
    cv2.rectangle(img2, location, bottom_right, 255, 5)
    cv2.imshow('Match', img2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# -----------------------------------------------------------------------------------------------------------
# Blending Image
import cv2
import numpy as np

# Blending : addition of two images using opencv
# if images not equal resize it.
img1 = cv2.imread("shoe.png")
img1 = cv2.resize(img1, (400,400))
img2 = cv2.imread("ball.png")
img2 = cv2.resize(img2, (400,400))

# Now perform blending
result = img2+img1  #numpy addition in this we get module between value

# recommended to use cv2.add
result1 = cv2.add(img1,img2) #its your saturated oprn which means value to value

# by weight - we can choose which image should be shown more/less
# function cv2.addWeighted(img1,wt1,img2,wt2,gama_val)  , sum of both the weight  = w1+w2 = 1(max)
result2 = cv2.addWeighted(img1,0.7,img2,0.3,0)

# cv2.imshow("result==",result)
# cv2.imshow("result1==",result1)
# cv2.imshow("result2 = ",result2)

# ----------------
# 2.using trackbar
def blend(x):
    pass

img = np.zeros((400,400,3),np.uint8)
cv2.namedWindow('win') #create track bar windows
cv2.createTrackbar('alpha','win',1,100,blend)         # alpha value from 0 to 100
switch = '0 : OFF \n 1 : ON'  #create switch for invoke the trackbars
cv2.createTrackbar(switch,'win',0,1,blend)  #create track bar for switch

while(1):
    alpha = cv2.getTrackbarPos('alpha','win')
    s = cv2.getTrackbarPos(switch,'win')
    na = float(alpha/100)    # max value for trackbar which becomes '1'
    
    if s == 0:
        dst = img[:]
    else:
        dst = cv2.addWeighted(img1, 1-na, img2, na, 0)
        cv2.putText(dst, str(alpha), (20, 50), cv2.FONT_ITALIC, 2, (0, 125, 255), 2)
    cv2.imshow('dst',dst)

    k=cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    
cv2.waitKey(0)    
cv2.destroyAllWindows()


# -------------------------------------------------------------------------------
# Stauration , Brightness , contrast , sharpness
# saturation - in HSV we can change value of 's'

# Brightness , contrast
alpha = 1    # contrast (1.0 to 3.0)
beta = 10    # brightness (1 to 100)
adjusted = cv2.convertScaleAbs(img1, alpha=alpha, beta=beta)     # another method : cv2.addWeighted()

# sharpness
kernel = np.ones((3,3), np.uint8)  # 5x5 matrix, type of the object, all values are 1, Remember it should be odd 3x3, 5x5
sharpened = cv2.filter2D(img1, -1, kernel)


# using trackbar for : Brightness , Contrast
def blend(x):
    pass

img1 = cv2.imread('abc.png')
img2 = img1.copy()
# img = np.zeros((400,400,3),np.uint8)
cv2.namedWindow('win') #create track bar windows
cv2.createTrackbar('Brightness','win',10,100,blend)  # brightness value from 1.0 to 3.0 , in while loop convert into float
cv2.createTrackbar('Contrast','win', 1, 100,blend)  # contrast value from 1 to 100

while(1):
    alpha = cv2.getTrackbarPos('Brightness','win')
    beta = cv2.getTrackbarPos('Contrast', 'win')

    alpha = float(alpha/10)    # max value for trackbar which becomes '1'
    
    brightened = cv2.addWeighted(img1, alpha, img2, 0, beta)
    cv2.putText(brightened, f"Brightness : {alpha}", (10, 40), cv2.FONT_ITALIC, 1, (0, 125, 255), 2)
    cv2.putText(brightened, f"Contrast : {beta}", (10, 80), cv2.FONT_ITALIC, 1, (0, 125, 255), 2)
    cv2.imshow('Brightness',brightened)
    cv2.imshow("original", img1)

    k=cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    
cv2.waitKey(0)    
cv2.destroyAllWindows()
