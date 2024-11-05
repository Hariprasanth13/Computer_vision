from pdf2image import convert_from_path
import cv2
import numpy as np
import imutils
import os
from ultralytics import YOLO
import pytesseract
import glob
from pytesseract import Output

#To convert the pdf to images
def pdf_to_images():
    images = convert_from_path('sample_pdf2.pdf',poppler_path = r"C:\\poppler-24.02.0\\Library\\bin")

    for i in range(len(images)):
        images[i].save('C:\\Projects\\ArrowDetection\\images\\page'+str(i)+'.jpg','JPEG')

# To detect the drawing within an image
def haar_cascade():
    output_folder = "C:\\Projects\\ArrowDetection\\object_crop"
    cascade_model = cv2.CascadeClassifier('drawing_cascade.xml')

    image = cv2.imread("C:\\Projects\\ArrowDetection\\images\\page1.jpg")

    #cv2.imshow("image",image)
    image = imutils.resize(image,width = 1000)

    original_image = image.copy()
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    result = cascade_model.detectMultiScale(image_gray,minNeighbors =4, minSize= (100,100))

    print(result)

    for (x,y,w,h) in result:

    # To draw a rectangle around the detected face  
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),2)

    cv2.imshow("bounding Image",image)
    cv2.waitKey(0)

    for i, (x, y, w, h) in enumerate(result):
        # Crop the image using bounding box coordinates
        cropped_image = original_image[y:y+h, x:x+w]
        
        # Generate output file path
        output_path = os.path.join(output_folder, f"cropped_{i}.jpg")
        cropped_image = imutils.resize(cropped_image, width = 500)
        # Save the cropped image
        cv2.imwrite(output_path, cropped_image)


def predict_dia():
    # Load a pretrained YOLOv8n model
    model = YOLO('C:\\Projects\\ArrowDetection\\runs\\detect\\train2\\weights\\best.pt')



    # Run inference on 'bus.jpg' with arguments
    results = model.predict(r'C:\\Projects\\ArrowDetection\\images\\page1.jpg', save=True, save_txt = True,save_crop = True)


#Predict the arrows and radius in an image using YOLO model
def predict_dim():
    # Load a pretrained YOLOv8n model
    model = YOLO('C:\\Projects\\ArrowDetection\\runs\\detect\\train\\weights\\best.pt')



    # Run inference on 'bus.jpg' with arguments
    results = model.predict(r'C:\\Projects\\ArrowDetection\\runs\\detect\\predict52\\crops\\drawing\\page12.jpg', save=True, save_txt = True,save_crop = True)


# noise removal
def remove_noise(image):
    #image = imutils.resize(image, width = 200)
    # width = image.shape[2]
    # ratio = 10/width
    # new_height = int(image.shape[1] * ratio)
    # new_width = int(width*ratio)
    # dim = (new_width,new_height)

    # image = cv2.resize(image,dim, interpolation=cv2.INTER_AREA)
    return cv2.medianBlur(image,3)

#erosion
def erode(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((1,1),np.uint8)
    image = cv2.erode(image, kernel, iterations = 1)
    image = cv2.bitwise_not(image)
    return image

# read the measurements
def read_digits():
    cwd = os.getcwd()

    #pytesseract.pytesseract.tesseract_cmd = cwd+"/tesseract.exe"
    #TESSDATA_PREFIX = cwd+"/tessdata"
    #os.environ['TESSERACT_CMD']='C:\\Projects\\ArrowDetection\\Tesseract-OCR\\tesseract.exe'
    pytesseract.pytesseract.tesseract_cmd = cwd+"\\Tesseract-OCR\\tesseract.exe"
    TESSDATA_PREFIX = cwd+"\\Tesseract-OCR\\tessdata"
    os.environ['TESSDATA_PREFIX']=cwd+"\\Tesseract-OCR\\tessdata"
    for i in [1,3,4,5,6,7,8,9,11,12,13]:
    #custom_config = r'--psm {} -c tessedit_char_whitelist="0123456789"'.format(i)
    #custom_config = r'--psm {} outputbase digits'.format(i)
        custom_config = r'--oem {} --psm {} outputbase digits'.format(3,i) # for setting the PSM type

        #folder_path = "C:\\Projects\\ArrowDetection\\runs\\detect\\predict19\\crops\\arrow"
        folder_path = "C:\\Projects\\ArrowDetection\\runs\\detect\\predict55\\crops\\arrow"

        image_paths = glob.glob(folder_path+"/*.jpg")
        print(" PSM Value: {}".format(i))
        for image_path in image_paths:

            image = cv2.imread(image_path)
            cv2.imshow("image",image)

            #image = add_padding(image,20)

            # top = 10
            # bottom = 10
            # left = 10
            # right = 10
            # image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value = (255,255,255))

            # image = cv2.resize(image, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
            # image = imutils.resize(image,width = 300)
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

            image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            thres,im_bw = cv2.threshold(image_gray,180,255,cv2.THRESH_BINARY)
            im_bw = erode(im_bw)

            #image_remove_noise = remove_noise(image_gray)
            #image_hist = cv2.equalizeHist(image_remove_noise)
            # image_dilate = dilate(image_remove_noise)
            # image_canny = canny(image)    
            
            cv2.imshow("Threshold",im_bw)

            text = pytesseract.image_to_string(im_bw, config = custom_config)
            print("Text : {}".format(text))

            # image_RGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            # osd = pytesseract.image_to_osd(image_RGB, output_type= Output.DICT)
            # print(osd)

            cv2.waitKey(0)
if __name__ == "__main__":
    pdf_to_images()
    #haar_cascade()
    predict_dia()
    predict_dim()
    read_digits()


