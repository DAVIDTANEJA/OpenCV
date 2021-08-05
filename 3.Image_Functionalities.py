import cv2
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------------------
# Image Smoothening : for blurring image , noise remove.
# Types : 1.LOW Pass Filter (LPS) : remove Noise from images , 2.High Pass Filter : edge detection.
# filters : homogeneous, blur(averaging), homogeneous, median, bilateral.

img = cv2.imread("noisy.jpg")    # noisy image distorted pixels
img = cv2.resize(img,(400,400))

cv2.imshow("original==",img)
kernel = np.ones((5,5),np.float32)/25 # define a kernel for homogeneous function

# 1st filter : homogenous / 2D convolution matrix
# Its work like: each output pixel is the mean of its kernel neigbours , in this all pixel contribute with equal weight.
# kernel is a small shape or matrix which we apply on image. in "kernel is : [(1/kernal(h,w))*kernal] ".
h_filter = cv2.filter2D(img, -1, kernel) # -1 / -2 is desired depth
cv2.imshow("homogeneous==",h_filter)

# 2. blur method (or averaging) filter :
# takes the avg of all the pixels under kernel area and replaces the central element with this average..
blur = cv2.blur(img,(8,8))  # here we have image and kernel as parameter
cv2.imshow("blur==",blur)

# 3.box Filter : increase instensity of image
box = cv2.boxFilter(img, -1, (2,2), normalize=False)  # (img, depth, kernel, normalize) , normalize - for image instensity increase, use 'kernel' acc.
cv2.imshow('box', box)

# 4.Gaussian Filter : it uses different weight kernel, in row as well as in column.
# It means side values are small then centre. It manage distance b/w value of pixels.
gau= cv2.GaussianBlur(img,(5,5),0) # 0 -is sigma x value
cv2.imshow("gau blur=",gau)

# 5.Median Filter : reduction of noise
# computes the median of all the pixels under the kernel window and the central pixel is replaced with this median value.
# This is highly effective in removing 'salt-and-pepper noise' (white dots in black image). Here kernal size must be odd except one
med = cv2.medianBlur(img, 5)            # in this we use square kernel so using - 5 only
cv2.imshow("median filter",med)

# 6.Bilateral filter : is highly effective at noise removal while preserving edges.
# It work like gaussian filter but more focus on edges , it is slow as compare with other filters
# sigma color - more this value makes 1 color around this all colors by mixing them 
bi_f = cv2.bilateralFilter(img, 9, 75, 75)    # (img, neigbour_pixel_diameter, sigma_color, sigma_space)
cv2.imshow("bi_f",bi_f)

# Now plot all the images on graph
titles = ["original_image","homo","blur","gauss","med","bi_f"]
images = [img,h_filter,blur,gau,med,bi_f]
for i in range(6):
    plt.subplot(2, 3, i+1), 
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()

# -------------------------------------------------------------------------------
# 1.High pass and Low pass Filter : here we multiply with 'kernel / matrix' with each pixel and put value in center.
# 2.Motion blur : its directional low-pass filter in which we pass 'kernel / matrix' horizontally 1 = [[0,0,0], [1,1,1], [0,0,0]]

# ----------------------------------
# 1.High pass and Low pass Filter
img = cv2.imread("photo.jpg")
# filters
kernel_identity = np.array([[0,0,0],[0,1,0],[0,0,0]])
kernel_3  = np.ones((3,3), dtype=np.uint8) / 9.0                # divide by 9 bcoz matrix is 3x3
kernel_11 = np.ones((11,11), dtype=np.uint8) / 121.0
# Note : if we don't divide by 9 / 121 it will make matirx value 1 and after applying increase value of pixels
# print(kernel_identity)
# print(kernel_3)

# Apply the filters
output1 = cv2.filter2D(img, -1, kernel_identity)   # (image, depth, kernel)
output2 = cv2.filter2D(img, -1, kernel_3)
output3 = cv2.filter2D(img, -1, kernel_11)

cv2.imshow('same', output1)
cv2.imshow('3 blur', output2)
cv2.imshow('11 blur', output3)
cv2.waitKey(0)


# -----------------
# 2.Motion blur
img = cv2.imread('photo.jpg')
size = 15

kernel = np.zeros((size,size))
kernel[int((size-1)/2),:] = np.ones(size)
kernel = kernel/size

output  = cv2.filter2D(img, -1, kernel)
cv2.imshow('winname', output)
cv2.imshow('winnam1e', img)
cv2.waitKey(0)

# --------------------------------------------------------------------------------------------
# Image sharpness / sharpening
# In addWeighted() : # we have to make 1 by adding - alpha, beta  -like :  1-na + na = 1  ,  0.7 + 0.3

img = cv2.imread('nature.jpg')

#Gauusian kernel for sharpening
gaussian_blur = cv2.GaussianBlur(img, (7,7), 2)

# Sharpening using addweighted()
sharpened1 = cv2.addWeighted(img,1.5, gaussian_blur, -0.5, 0)
sharpened2 = cv2.addWeighted(img,3.5, gaussian_blur, -2.5, 0)
sharpened3 = cv2.addWeighted(img,7.5, gaussian_blur, -6.5, 0)

# Showing the sharpened Images
cv2.imshow('Sharpened 3', sharpened3)
cv2.imshow('Sharpened 2', sharpened2)
cv2.imshow('Sharpened 1', sharpened1)
cv2.imshow('original', img)
cv2.waitKey(0)

# --------------------------------------------------------------------------------------------
# Stauration , Brightness , contrast , sharpness
# saturation - in HSV we can change value of 's'

# Brightness , contrast
alpha = 1    # contrast (1.0 to 3.0)
beta = 10    # brightness (1 to 100)
adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)     # another method : cv2.addWeighted()

# sharpness
kernel = np.ones((3,3), np.uint8)  # 5x5 matrix, type of the object, all values are 1, Remember it should be odd 3x3, 5x5
sharpened = cv2.filter2D(img, -1, kernel)


# using trackbar for : Brightness , Contrast
def blend(x):
    pass

img1 = cv2.imread('abc.png')
img2 = img1.copy()
# img = np.zeros((400,400,3),np.uint8)
cv2.namedWindow('win') #create track bar windows
cv2.createTrackbar('Brightness','win',10,100,blend)  # brightness value from 1.0 to 3.0 , in while loop convert into float
cv2.createTrackbar('Contrast','win', 1, 100,blend)  # contrast value from 1 to 100

while(1):
    alpha = cv2.getTrackbarPos('Brightness','win')
    beta = cv2.getTrackbarPos('Contrast', 'win')

    alpha = float(alpha/10)    # max value for trackbar which becomes '1'
    
    brightened = cv2.addWeighted(img1, alpha, img2, 0, beta)
    cv2.putText(brightened, f"Brightness : {alpha}", (10, 40), cv2.FONT_ITALIC, 1, (0, 125, 255), 2)
    cv2.putText(brightened, f"Contrast : {beta}", (10, 80), cv2.FONT_ITALIC, 1, (0, 125, 255), 2)
    cv2.imshow('Brightness',brightened)
    cv2.imshow("original", img1)

    k=cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    
cv2.waitKey(0)    
cv2.destroyAllWindows()
