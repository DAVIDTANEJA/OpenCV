# 1st we select object - drawing rectangle / box around it.  2nd we track that object.
# for trackers : install : opencv-contrib-python
# Select the ROI and press "enter" it will start tracking.


import cv2

# Tracker Types
# tracker = cv2.TrackerBoosting_create()
# tracker = cv2.TrackerMIL_create()
# tracker = cv2.TrackerKCF_create()
# tracker = cv2.TrackerTLD_create()
# tracker = cv2.TrackerMedianFlow_create()
tracker = cv2.TrackerCSRT_create()
# tracker = cv2.legacy.TrackerMOSSE_create()

cap = cv2.VideoCapture(0)

# TRACKER INITIALIZATION- it will take 1st frame and draw bbox around it and then start tracking in while loop
success, frame = cap.read()
bbox = cv2.selectROI("Tracking", frame, False)    # bbox for 1st frame
tracker.init(frame, bbox)     # initialization


def drawBox(img,bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x, y), ((x + w), (y + h)), (255, 0, 255), 3, 3 )
    cv2.putText(img, "Tracking", (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


while True:

    timer = cv2.getTickCount()
    success, img = cap.read()
    success, bbox = tracker.update(img)    # run tracker/update, bbox gives: (x,y,w,h)

    if success:
        drawBox(img, bbox)      # if getting object then draw box otherwise say 'Lost' it.
    else:
        cv2.putText(img, "Lost", (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.rectangle(img,(15,15),(200,90),(255,0,255),2)
    cv2.putText(img, "Fps:", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)
    cv2.putText(img, "Status:", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    # frame rate
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    if fps>60: myColor = (20,230,20)
    elif fps>20: myColor = (230,20,20)
    else: myColor = (20,20,230)
    cv2.putText(img, str(int(fps)), (75,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, myColor, 2)

    cv2.imshow("Tracking", img)
    
    if cv2.waitKey(1) & 0xff == ord('q'):
       break
