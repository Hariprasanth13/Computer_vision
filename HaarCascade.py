import cv2
import numpy as np
import imutils
import os

output_folder = "C:\\Projects\\ArrowDetection\\object_crop"
cascade_model = cv2.CascadeClassifier('drawing_cascade.xml')

image = cv2.imread("C:\\Projects\\ArrowDetection\\images\\page1.jpg")



#image = imutils.resize(image,width = 1500 , inter=cv2.INTER_LINEAR)
image = image.astype(np.uint8)
cv2.imshow("image",image)
original_image = image.copy()
print(image.shape)

image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
result = cascade_model.detectMultiScale(image_gray,minNeighbors =4, minSize= (100,100))

print(result)

for (x,y,w,h) in result:

   # To draw a rectangle around the detected images
   cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),2)

cv2.imshow("bounding Image",image)
cv2.waitKey(0)

for i, (x, y, w, h) in enumerate(result):
    # Crop the image using bounding box coordinates
    cropped_image = original_image[y:y+h, x:x+w]
    
    # Generate output file path
    output_path = os.path.join(output_folder, f"cropped_{i}.jpg")
    #cropped_image = imutils.resize(cropped_image, width = 500)
    # Save the cropped image
    cv2.imwrite(output_path, cropped_image)