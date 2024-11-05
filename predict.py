from ultralytics import YOLO
import cv2

# Load a pretrained YOLOv8n model
model = YOLO('C:\\Projects\\ArrowDetection\\runs\\detect\\train\\weights\\best.pt')



# Run inference on 'bus.jpg' with arguments
results = model.predict(r'C:\\Projects\\ArrowDetection\\runs\\detect\\predict52\\crops\\drawing\\page12.jpg', save=True, save_txt = True,save_crop = True)
# image = cv2.imread("d6.jpg")

# image_RBG = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

# for r in results:
#     #print("Coordinates are: {}".format(r.boxes.xywh))
#     boxes = (r.boxes.xywh).numpy()
# for box in boxes:
#     (x,y,w,h) = box.astype(int).flatten()
#     print(box.shape)
    # image_cropped = image_RBG[y,x,y+h,x+w]
    # cv2.imshow(image_cropped)