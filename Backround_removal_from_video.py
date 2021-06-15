# Background Subtraction - to extract the moving foreground from static background.

import cv2

cap = cv2.VideoCapture('test2.mp4')

# old_algo = cv.bgsegm.createBackgroundSubtractorMOG()  # not available now.
algo1 = cv2.createBackgroundSubtractorMOG2(detectShadows=True) # algo1 
algo2 = cv2.createBackgroundSubtractorKNN(detectShadows=True)  # algo2

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame,(600,400))
    if frame is None:
        break

    res1 = algo1.apply(frame)  # apply algo to 'frame'
    res2 = algo2.apply(frame)

    cv2.imshow('original', frame)
    cv2.imshow('res1',res1)
    cv2.imshow('res2',res2)

    keyboard = cv2.waitKey(60)
    if keyboard == 'q' or keyboard == 27:
        break

cap.release()
cv2.destroyAllWindows()

# ---------------------------------------------------------------------------------
# GrabCut Algoritm :
# cutoff any foreground object from image/video. 
# It works like a rectangle portion mark on the image and area outise the rectangle is treat as a extra part so remove it.
# Gaussian Mixture model helps in this.

import	numpy  as  np
import	cv2

img  =	cv2.imread('abc.png')
img = cv2.resize(img,(800,800))
mask =	np.zeros(img.shape[:2],np.uint8)  # shape of img., to tell grabcut() which pixels wants to remove.

bgdModel =  np.zeros((1,65),np.float64)*255   # we place this image inplace of bg removal
fgdModel =  np.zeros((1,65),np.float64)*255   # and use this if image is distorted/noise fill with this.
# rectangle(x1,y1,x2,y2) , get cordinates from image open in Paint which you want to cut.
rect = (375,180,660,790)	# (134,150,660,730)  
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)  # (img,mask,rect,bgmodel,fgmodel,iter,method)
# To find area using -mask
mask2  =  np.where((mask==2)|(mask==0),0,1).astype('uint8')
img  =	img * mask2[:,:, np.newaxis]

cv2.imshow("res==",img)
cv2.waitKey(0)
cv2.destroyAllWindows()