import cv2
import numpy as np


# here we store in matrix where we click on image and count those values/clicks
circles = np.zeros((4,2), np.int)    # matrix : we need 4 points and 2 means need of x,y
counter = 0

# gives x,y points where we click on image
def mousePoints(event, x, y, flags, params):
    global counter
    if event == cv2.EVENT_LBUTTONDOWN:
        # print(x,y)
        circles[counter] = x,y   # store x,y in matrix
        counter += 1
        print(circles)


img = cv2.imread('abc.png')    # take any card image

while True:

    if counter == 4:
        # width, height = 250,350      # 2.5 x 3.5 inches - image size after warp / cutout
        width, height = int(circles[1][0] - circles[0][0]), int(circles[2][1] - circles[0][1])
        print(width, height)

        # warp perspective - capture particular part of image
        # if want to find points,  open image in "Paint" and move cursor it will show below (height,width)points
        pts1 = np.float32([circles[0],circles[1], circles[2], circles[3]]) 
        pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]]) 
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        output = cv2.warpPerspective(img, matrix, (width,height))
        cv2.imshow("image", output)    # this name should not be same from - original image, otherwise show error

    for x in range(0,4):
        cv2.circle(img, (circles[x][0], circles[x][1]), 10, (0,255,0), -1)


    cv2.imshow("Original image", img)                   # Remeber to give same name in both like : "Original image"
    cv2.setMouseCallback("Original image", mousePoints)

    if cv2.waitKey(1) == 27:
        break

