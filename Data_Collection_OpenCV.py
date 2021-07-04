# Collection of data using camera, After collecting images , take those images of particular object 
# and save in folder 'positive' images and also take images of other obejcts in 'negative' folder. To classify images.
# After open "Cascade Trainer" and pass both folder : "positive" and "negative" into it. 
# And it wil create "cascade.xml" file and use acc. to classify obejct / detect object.


import cv2
import os
import time

#####################################################

myPath = 'data/images'   # create new folder in 'data/images' everytime we run this.
cameraNo = 1
cameraBrightness = 180
moduleVal = 10           # SAVE EVERY 10th FRAME TO AVOID REPETITION change acc.
minBlur = 500            # SMALLER VALUE MEANS MORE BLURRINESS , and it avoid blur images so it will take higher values
grayImage = False        # IMAGES SAVED COLORED OR GRAY
saveData = True          # SAVE DATA FLAG                     # can change acc.  True / False
showImage = True         # IMAGE DISPLAY FLAG
imgWidth = 180
imgHeight = 120


#####################################################

global countFolder
cap = cv2.VideoCapture(cameraNo)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10,cameraBrightness)

count = 0
countSave =0

# save data in folder
def saveDataFunc():
    global countFolder
    countFolder = 0
    while os.path.exists( myPath+ str(countFolder)):
        countFolder += 1
    os.makedirs(myPath + str(countFolder))

if saveData:saveDataFunc()


while True:

    success, img = cap.read()
    img = cv2.resize(img,(imgWidth,imgHeight))
    if grayImage:img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    if saveData:
        blur = cv2.Laplacian(img, cv2.CV_64F).var()
        if count % moduleVal ==0 and blur > minBlur:
            nowTime = time.time()
            cv2.imwrite(myPath + str(countFolder) +
                    '/' + str(countSave)+"_"+ str(int(blur))+"_"+str(nowTime)+".png", img)
            countSave+=1
        count += 1

    if showImage:
        cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



# ----------------------------------------------------------------------------
# Use this to detect object
import cv2

################################################################
path = 'haarcascades/haarcascade_frontalface_default.xml'  # PATH OF THE CASCADE created 
cameraNo = 1                       # CAMERA NUMBER
objectName = 'Arduino'       # OBJECT NAME TO DISPLAY
frameWidth= 640                     # DISPLAY WIDTH
frameHeight = 480                  # DISPLAY HEIGHT
color= (255,0,255)
#################################################################

cap = cv2.VideoCapture(cameraNo)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

def empty(a):
    pass

# CREATE TRACKBAR
cv2.namedWindow("Result")
cv2.resizeWindow("Result",frameWidth,frameHeight+100)
cv2.createTrackbar("Scale","Result",400,1000,empty)
cv2.createTrackbar("Neig","Result",8,50,empty)
cv2.createTrackbar("Min Area","Result",0,100000,empty)
cv2.createTrackbar("Brightness","Result",180,255,empty)

# LOAD THE CLASSIFIERS DOWNLOADED
cascade = cv2.CascadeClassifier(path)

while True:
    # SET CAMERA BRIGHTNESS FROM TRACKBAR VALUE
    cameraBrightness = cv2.getTrackbarPos("Brightness", "Result")
    cap.set(10, cameraBrightness)
    # GET CAMERA IMAGE AND CONVERT TO GRAYSCALE
    success, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # DETECT THE OBJECT USING THE CASCADE
    scaleVal =1 + (cv2.getTrackbarPos("Scale", "Result") /1000)
    neig=cv2.getTrackbarPos("Neig", "Result")
    objects = cascade.detectMultiScale(gray,scaleVal, neig)
    # DISPLAY THE DETECTED OBJECTS
    for (x,y,w,h) in objects:
        area = w*h
        minArea = cv2.getTrackbarPos("Min Area", "Result")
        if area >minArea:
            cv2.rectangle(img,(x,y),(x+w,y+h),color,3)
            cv2.putText(img,objectName,(x,y-5),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,color,2)
            roi_color = img[y:y+h, x:x+w]

    cv2.imshow("Result", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
         break