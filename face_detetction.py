# Face Detection : method proposed by "VIOLA & JONES" earliest method to real time obejct detection
# To detect Faces - collect lots of "Positive faces" and "Negative non faces" and train them and create a "Cascade" file to detect faces.
# Here we are using already pre-trained  "cascade file" , OpenCV provides many files for detection : Face, number plate, eyes, full body.
# When install Opencv : haarcascade files also installed :  copy into folder
# C:\Users\DavidTaneja\AppData\Local\Programs\Python\Python39\Lib\site-packages\cv2\data

# OpenCV - read the image and features of file and convert image into Numpy array (or convert color image into grayscale).
# detectMultiScale() : method to Search for the rows-columns values of the face numpy ndarray (face rectangle co-ordinates). 
# scalefactor : decrease the shape value by 5%, until face is found. Smaller this value, greater is accuracy.


# Note : we can also create trained  "Custom cascade file"


import cv2

# faceCascade = cv2.CascadeClassifier("haarcascade_frontal_face_default.xml")
# eyeCascade = cv2.CascadeClassifier("haarcascade_eye.xml")

# img = cv2.imread('soccer_practice.jpg')
# imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)    # (image, scale factor, min. neighbors)

# # create rectangle around faces
# for (x,y,w,h) in faces:
#     # draw rectangle around face
#     img=cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)  # (image, initial points, corner/diagonal points, color, thickness)

#     # # detect eyes : it will take ROI of face and then detecct eyes from it.
#     # roi_gray = imgGray[y:y+h, x:x+w]
#     # roi_color = img[y:y+h, x:x+w]
#     # eyes = eyeCascade.detectMultiScale(roi_gray,1.2,1)  # detect eyes
#     # for (ex,ey,ew,eh) in eyes:
#     #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)

# cv2.imshow("Result", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# -----------------------------------------
# Using Webcam
face=cv2.CascadeClassifier("haarcascade_frontal_face_default.xml")    # face
eye = cv2.CascadeClassifier('haarcascade_eye.xml')                   # eyes

def face_eye_detect(img):
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray,1.3,5)
    # rectangle around face
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(127,0,125),3)

        # take ROI of face and then detetct eyes
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye.detectMultiScale(roi_gray,1.3,3)
        # (ex,ey,ew,eh) cordinates for rectangle and for circle use only (ex, ey) - starting point of eyes so here we add some padding
        for (ex,ey,ew,eh) in eyes:
            cv2.circle(roi_color, (ex+27,ey+27), 20, (255,255,0), 2)    # cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,111),2)
    return img

# Webcam
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
while True:
    ret,frame =cap.read()
    frame = cv2.flip(frame,2)
    cv2.imshow("face dect",face_eye_detect(frame))

    if cv2.waitKey(1)==27:   # press enter to terminate
        break
    

cap.release()
cv2.destroyAllWindows()