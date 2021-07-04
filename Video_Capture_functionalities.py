# Capture the video from webcam
import cv2
import numpy as np

cap = cv2.VideoCapture(0)    # 0 -webcam, 1,2: others camera , 'video.mp4' can pass here any video file

# print("for width===",cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# print("for height==",cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# we can also set the "width, height" of webcam , also brightness
# cap.set(3, 640)    # 3 -width  , 640 px width
# cap.set(4, 480)    # 4 -height , 480 px height
# cap.set(10, 100)   # 10 -default code for brightness , 100 value for brightness

# cv2.namedWindow("LIve_Recording",cv2.WINDOW_NORMAL)    # name the window
# cv2.resizeWindow("Live",(600,400))    # Resize the window

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

# --------------------------------------------------------------------------------------------
# rescale the frame
def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    frame_resized = rescaleFrame(frame, scale=0.20)
    cv2.imshow('Original', frame)
    cv2.imshow('resized', frame_resized)
    if cv2.waitKey(1) == 27 :
        break

cap.releaes()
cv2.destroyAllWindows()

# --------------------------------------------------------------------------------------------
# Draw line, rectangle, circle, images, Text - on  'webcam'

while True:
    ret, frame = cap.read()    # ret - True:to tell capture actually work properly, frame - numpy array image
    width = int(cap.get(3))    # 3 - default value for 'width' of frame
    height = int(cap.get(4))   # 4 - height of frame
    
    # line , (image, stating, ending point, color of line, width of line)
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


# -----------------------------------------------
#Capture  video from webcam and save it

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)   #Here parameter 0 is a path of any video use for webcam
print("check===",cap.isOpened())

#it is 4 byte code which is use to specify the video codec
#Various codec -- 
#DIVX, XVID, MJPG, X264, WMV1, WMV2
fourcc = cv2.VideoWriter_fourcc(*"XVID")  # *"XVID"
# It contain 4 parameter , name, codec,fps,resolution
output = cv2.VideoWriter("output.avi",fourcc,20.0,(640,480),0)

while(cap.isOpened()):
    ret, frame = cap.read()   #here read the frame
    
    if ret==True:
        gray  = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #here flip is used to flip the video at recording time
        frame = cv2.flip(frame,0)
        output.write(gray)
        
        # cv2.imshow("Gray Frame",gray)    # cv2.imshow('Colorframe',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):   #press to exit
            break
    else:
        break
 
# Release everything if job is finished
cap.release()
output.release()
cv2.destroyAllWindows()


# -------------------------------------------------------------
# Put text on webcam / date - time on live webcam
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
       text = ' Height: ' + str(cap.get(4))+' Width: '+ str(cap.get(3))
       date_data = "Date: "+str(datetime.datetime.now())

       font = cv2.FONT_HERSHEY_COMPLEX_SMALL
       frame = cv2.putText(frame, text, (10, 20), font, 1, (0, 155, 255), 1, cv2.LINE_AA)
       frame = cv2.putText(frame, date_data, (20, 50), font, 1, (100, 255, 255), 1, cv2.LINE_AA)

       cv2.imshow('frame', frame)
       if cv2.waitKey(1) & 0xFF == ord('q'):
         break
    else:
        break

# -------------------------------------------------------------
# Capture video from Youtube
import pafy      # pip install pafy youtube-dl
import cv2

url = "https://www.youtube.com/watch?v=SLD9xzJ4oeU"
data = pafy.new(url)
data = data.getbest(preftype="mp4")

cap = cv2.VideoCapture()   
cap.open(data.url)

fourcc = cv2.VideoWriter_fourcc(*"XVID")  # *"XVID"
output = cv2.VideoWriter("output.avi",fourcc,10.0,(640,480),0)

while(cap.isOpened()):
    ret, frame = cap.read()   #here read the frame
    
    if ret==True:
        gray  = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        #output.write(gray)
        #cv2.imshow("Gray Frame",gray)
        cv2.imshow('Colorframe',frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):   #press to exit
            break
    else:
        break
 
cap.release()
output.release()
cv2.destroyAllWindows()

# -------------------------------------------------------------
# Capture Multiple Images/frames and Store in a folder
vidcap = cv2.VideoCapture(0)
ret,image = vidcap.read()        # READ THE VIDEO / capture the 1st frame
count = 0

while True:
    if ret == True:
        cv2.imwrite("frames\\imgN_%d.jpg" % count, image)     # save frame/image, %d : change 'count' no.
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count**100))     # used to hold speed of frames
        ret,image = vidcap.read()                          # again read the video for frames
        cv2.imshow("res",image)
        print ('Read a new frame:',count ,ret)
        count += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
            cv2.destroyAllWindows()
        else:
            break

# -----------------------------------------
# Instead of guessing the millisecond sleep value of waitKey, you can calculate it from the FPS of the video: 
video = cv2.VideoCapture(video_path)
fps = video.get(cv2.CAP_PROP_FPS)
sleep_ms = int(numpy.round((1/fps)*1000))




cv2.waitKey(0)    # 0 -means it will wait infinite time until we press any key and it disappear.  10 - for 10 mili sec.
cap.release()                   # after using release the camera for other purpose use
cv2.destroyAllWindows()
