import cv2
import numpy as np


# read the image   # 0 / cv2.IMREAD_GRAYSCALE :grayscale, -1 / cv2.IMREAD_COLOR :color , 1 / cv2.IMREAD_UNCHANGED
img = cv2.imread('abc.png', -1)

img = cv2.resize(img, (200,200))  # resize  , (img, (0,0), fx=0.5, fy=0.5) -it make half of size if don't use pixel
img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)    # rotate 

cv2.imwrite('new_abc.jpg', img)       # save / write the image
cv2.imshow('Image', img)           # display image
print(img.shape)      # (rows, columns, 3d array/channels/pixels) / (height, width, channels/pixels)  channels- RGB color


cv2.waitKey(0)    # 0 -means it will wait infinite time until we press any key and it disappear.  10 - for 10 mili sec.
cv2.destroyAllWindows()


# ---------------------------------------------------------
# take the image part and paste into image some location
img = cv2.imread('abc.png', -1)
tag = img[80:400, 300:520]       # [rows,columns]  # take the image part
img[80:400, 100:320] = tag        # pixels must be equal what we take part of it , pasting into image here 'tag' part

cv2.imshow('Image', img)           # display image

# --------------------------------------------------------
# Capture the video from webcam
cap = cv2.VideoCapture(0)    # 0 -webcam, 1,2: others camera , 'video.mp4' can pass here any video file

while True:
    ret, frame = cap.read()    # ret - True:to tell capture actually work properly, frame - numpy array image
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):      # when we press 'q' it will break loop/ stop it.
        break


# now instead of displaying frame, turn this frame into 4 separate images, put all together in quadrant
while True:
    ret, frame = cap.read()    # ret - True:to tell capture actually work properly, frame - numpy array image

    # now instead of displaying frame, turn this frame into 4 separate images, put all together in quadrant
    width = int(cap.get(3))    # 3 - default value for 'width' of frame
    height = int(cap.get(4))   # 4 - height of frame
    # 1st create blank canvas / numpy array, where we put images into this, it should have (shape, type of array)
    image = np.zeros(frame.shape, np.uint8)
    smaller_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)  # now shrink 'frame' half of x,y (width,height)
    # now put smaller_frame into blank 'image'
    image[:height//2, :width//2] = smaller_frame     # top left starts from 0:
    image[height//2 :, :width//2] = smaller_frame    # below it left side
    image[:height//2, width//2 :] = smaller_frame    # top right
    image[height//2 :, width//2 :] = smaller_frame   # below right

    cv2.imshow('frame', image)
    if cv2.waitKey(1) == ord('q'):      # when we press 'q' it will break loop/ stop it.
        break 


# ------------------------------------------------------
# Draw line, rectangle, circle, images, Text

while True:
    ret, frame = cap.read()    # ret - True:to tell capture actually work properly, frame - numpy array image
    width = int(cap.get(3))    # 3 - default value for 'width' of frame
    height = int(cap.get(4))   # 4 - height of frame
    
    # (image, stating, ending point, color of line, width of line)
    # line
    img = cv2.line(frame, (0,0), (width, height), (0,0,0), 10)    # from left upper corner to right bottom    
    img = cv2.line(frame, (0,0), (width, height), (0,0,0), 10)    # from top right to left corner

    # rectangle
    img = cv2.rectangle(frame, (100,200), (200, 300), (0,0,0), -1)   # '-1' if we use this will fill the area with that color

    # circle
    img = cv2.circle(frame, (300,300), 60, (0,0,255), -1)    # (image, centre point, radius, color, width of line / fill with that color if -1)

    # put text
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.putText(frame,'David Taneja', (200, height - 10), font, 2, (0,0,0), 5, cv2.LINE_AA)  # (image, text, text place, font, font scale, color, line width)  # height-10 '-10' for padding

    cv2.imshow('frame', img)

# ---------------------------------------------------------------------------------------------------------------------
# Color Detection / convert :  RGB : Red Green Blue , BGR : Blue Green Red , HSV : Hue Saturation & Lightness / Brightness
# extract color from 'HSV' from lower and upper bound, it will diplay that color image only which lies b/w upper-lower color image

while True:
    ret, frame = cap.read()    # ret - True:to tell capture actually work properly, frame - numpy array image
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90,50,0])                    # change acc. lower, upper
    upper_blue = np.array([130,255,255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('frame', result)

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


# -----------------------------------------
# Instead of guessing the millisecond sleep value of waitKey, you can calculate it from the FPS of the video: 
video = cv2.VideoCapture(video_path)
fps = video.get(cv2.CAP_PROP_FPS)
sleep_ms = int(numpy.round((1/fps)*1000))




cv2.waitKey(0)    # 0 -means it will wait infinite time until we press any key and it disappear.  10 - for 10 mili sec.
cap.release()                   # after using release the camera for other purpose use
cv2.destroyAllWindows()
