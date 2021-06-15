# download "IP Webcam" app in Phone , Open app : Start server -> IPv4 address -> copy this ip address in browser -> click 'Browser' /'javascript' - fullscreen , it opens camera 
# connect your laptop and android device with same network either wifi or hotspot
import cv2
import numpy as np

camera = "http://192.168.0.102:8080/video"     # ip address shown in app, video - just name given
cap = cv2.VideoCapture(0)    # cv2.CAP_DSHOW  - pass this parameter if any error shows

cap.open(camera)             # this will open the 'phone camera'  not webcam

# print("check===",cap.isOpened())

fourcc = cv2.VideoWriter_fourcc(*"XVID")  # *"XVID"
output = cv2.VideoWriter("output.avi",fourcc,20.0,(640,480),0)    # (file name, codec, fps, resolution)

while(cap.isOpened()):
    ret, frame = cap.read()   #here read the frame
    if ret == True:
        frame = cv2.resize(frame,(700,700))
        gray  = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.imshow("Gray Frame",gray)
        cv2.imshow('Colorframe',frame)

        # output.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):   #press to exit
            break
    
cap.release()
output.release()
cv2.destroyAllWindows()

# ----------------------------------------------------------------------------------------------
# import requests
# import numpy as np
# import cv2

# # In android mobile phone open Play store install "IP Webcam" app, 
# # open it and click on below 'Start server'.
# # It shows Ipv4 address : "http........"  , copy and paste  in "images". and  add  '/shot.jpg'
# while True:
#     images = requests.get("http://....../shot.jpg")                  # copy Ip address
#     video = np.array(bytearray(images.content), dtype=np.uint8)
#     render = cv2.imdecode(video, -1)
#     cv2.imshow('frame', render)

#     if (cv2.waitKey(1) and 0xff==ord('q')):
#         break
