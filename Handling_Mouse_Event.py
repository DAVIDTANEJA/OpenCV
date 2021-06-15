import numpy as np
import cv2


# mouse callback function
def draw(event,x,y,flags,param):      # x,y - cordinates, flags - 1 for left click, 2 -rightclick, params execute operations
    if event == cv2.EVENT_LBUTTONDBLCLK:    # if left doubble click - it will draw circle on blank image
        cv2.circle(img,(x,y),100,(125,0,255),5)
        
    if event == cv2.EVENT_RBUTTONDBLCLK:    # if right double click - rectangle will draw
        cv2.rectangle(img,(x,y),(x+100,y+75),(125,125,255),2)

# Create a window of black image and bind the function to window
cv2.namedWindow(winname = "res")    
img = np.zeros((512,512,3), np.uint8)
cv2.setMouseCallback("res",draw)

while True:
    cv2.imshow("res",img)
    if cv2.waitKey(1) & 0xFF == 27:     # 27 - esc key
        break

cv2.destroyAllWindows()


# --------------------------------------------------------------------------------------------------------
# on black image/ any image : if left click - find cordinates , right click - gives color value of pixel

def mouse_event(event, x, y, flg, param):
    font = cv2.FONT_HERSHEY_PLAIN 
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,', ' ,y)
        cord = ". "+str(x) + ', '+ str(y)    # get the cordinates in var.
        cv2.putText(img, cord, (x, y), font, 1, (155,125 ,100), 2)    # then put text on x,y with cordinates

    if event == cv2.EVENT_RBUTTONDOWN:
        b= img[y, x, 0]   # for blue channel is 0
        g = img[y, x, 1]  # for green channel is 1
        r = img[y, x, 2]  # for red channel is 2
        
        color_bgr = ". "+str(b) + ', '+ str(g)+ ', '+ str(r)
        cv2.putText(img, color_bgr, (x, y), font, 1, (152, 255, 130), 2)


cv2.namedWindow(winname = "2nd Window")
# img = np.zeros((512, 512, 3), np.uint8)    # we can find on black image
img = cv2.imread('abc.png')                  # or any image
cv2.setMouseCallback('2nd Window', mouse_event)

while True:
    cv2.imshow("2nd Window",img)
    if cv2.waitKey(1) & 0xFF == 27:     # 27 - esc key
        break

cv2.waitKey(0)
cv2.destroyAllWindows()
