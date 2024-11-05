import pytesseract
import glob
from pytesseract import Output
import cv2
import os
import imutils
import numpy as np
import re

cwd = os.getcwd()

#pytesseract.pytesseract.tesseract_cmd = cwd+"/tesseract.exe"
#TESSDATA_PREFIX = cwd+"/tessdata"
#os.environ['TESSERACT_CMD']='C:\\Projects\\ArrowDetection\\Tesseract-OCR\\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = cwd+"\\Tesseract-OCR\\tesseract.exe"
TESSDATA_PREFIX = cwd+"\\Tesseract-OCR\\tessdata"
os.environ['TESSDATA_PREFIX']=cwd+"\\Tesseract-OCR\\tessdata"


# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,3)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((3,3),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((1,1),np.uint8)
    image = cv2.erode(image, kernel, iterations = 1)
    image = cv2.bitwise_not(image)
    return image

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 50, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 

#Noise Removal
def noise_removal(image):
    kernel = np.ones((1,1),np.uint8)
    image = cv2.dilate(image,kernel,iterations=1)
    kernel = np.ones((1,1),np.uint8)
    image = cv2.erode(image,kernel,iterations=1)
    #image = cv2.morphologyEx(image,cv2.MORPH_CLOSE,kernel)
    image = cv2.medianBlur(image, 3)
    return image


for i in [1,3,4,5,6,7,8,9,11,12,13]:
    #custom_config = r'--psm {} -c tessedit_char_whitelist="0123456789"'.format(i)
    #custom_config = r'--psm {} outputbase digits'.format(i)

    custom_config = r'--oem {} --psm {} outputbase digits'.format(3,i) # for setting the PSM type

    folder_path = "C:\\Projects\\ArrowDetection\\runs\\detect\\predict57\\crops\\arrow"
    #folder_path = "C:\\Users\\z031415\\Downloads\\test"

    image_paths = glob.glob(folder_path+"/*.jpg")
    print(" PSM Value: {}".format(i))
    for image_path in image_paths:

        image = cv2.imread(image_path)
        # image = imutils.resize(image,width = 100,inter=cv2.INTER_LINEAR)
        # cv2.imshow("image",image)
        #image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        #image = add_padding(image,20)

        # top = 10
        # bottom = 10
        # left = 10
        # right = 10
        # image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value = (255,255,255))

        # image = cv2.resize(image, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
        
        # image_RGB = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        # _, binarized_image = cv2.threshold(image_RGB, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # enhanced_image = clahe.apply(binarized_image)

        # laplacian = cv2.Laplacian(enhanced_image, cv2.CV_64F)
    
        # # Convert the Laplacian result back to the original image's data type
        # sharpened_image = np.uint8(np.clip(enhanced_image - 0.5*laplacian, 0, 255))

        # kernel = np.ones((3, 3), np.uint8)    
        # image_RGB = cv2.dilate(image_RGB, kernel, iterations=2)
        # image_RGB = cv2.erode(image_RGB, kernel, iterations=1)
        
        #cv2.threshold(cv2.GaussianBlur(enhanced_image, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        image_gray = get_grayscale(image)
        thres,im_bw = cv2.threshold(image_gray,180,255,cv2.THRESH_BINARY)

        #im_bw = erode(im_bw)
        #image_remove_noise = remove_noise(image)
        #image_hist = cv2.equalizeHist(image_remove_noise)
        # image_dilate = dilate(image_remove_noise)
        # image_canny = canny(image)
        #image = noise_removal(im_bw)
        cv2.imshow("Threshold",im_bw)
        

        text = pytesseract.image_to_string(image, config = custom_config)
        print("Text : {}".format(text))

        # image_RGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        # osd = pytesseract.image_to_osd(image_RGB, output_type= Output.DICT)
        # print(osd)

        cv2.waitKey(0)

