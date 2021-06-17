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
import os
import numpy as np

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

# -----------------------------------------------------------------------------------------------
# Face Recognition - using 'haarcascade' file : 
# 2 steps : 1.train model , 2.use model file and recognise

# create folders for persons to recognise : 1 person 1 folder with name as many pictures 20-30.
people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']  # folders names with person names
haar_cascade = cv2.CascadeClassifier('haarcascade_frontal_face_default.xml')

# # ---------------------
# # 1.train model - loop through the folder and take pictures faces and label with it.
# DIR = "here comes the directory path where it contains all people folders"
# features = []
# labels = []
# def create_train():
#     for person in people:
#         path = os.path.join(DIR, person)
#         label = people.index(person)

#         for img in os.listdir(path):
#             img_path = os.path.join(path, img)
#             img_array = cv2.imread(img_path)
#             gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
#             # detect faces and crop out (x,y,w,h) and append to features and label it.
#             faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
#             for (x,y,w,h) in faces_rect:
#                 faces_roi = gray[y:y+h,x:x+w]
#                 features.append(faces_roi)
#                 labels.append(label)    # label -index no. of list

# create_train()
# print(f"no. of faces : {len(features)} and labels : {len(labels)}")
# # Train the recognizer on 'features list' and 'labels list', 1st convert features and labels into array befor train.
# features = np.array(features)
# labels = np.array(labels)
# face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# face_recognizer.train(features, labels)
# # save the file
# face_recognizer.save("face_trained.yml")     # trained model
# np.save('features.npy', features)
# np.save('labels.npy', labels)
# print("Training Done")


# -------------------
# 2.recognise using trained model 
# load/read the files
# features = np.load('features.npy', allow_pickle=True)    # use it / not
# labels = np.load('labels.npy')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

# Detect the face in the image
img = cv2.imread('1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Person', gray)
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h,x:x+w]

    label, confidence = face_recognizer.predict(faces_roi)
    # print(f'Label = {people[label]} with a confidence of {confidence}')     # people[label] - give the label of people/person

    cv2.putText(img, str(people[label]), (20,20), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)
 
cv2.imshow('Detected Face', img)
cv2.waitKey(0)



