from ultralytics import YOLO
import cv2

# Load a pretrained YOLOv8n model
model = YOLO('C:\\Projects\\ArrowDetection\\runs\\detect\\train2\\weights\\best.pt')



# Run inference on 'bus.jpg' with arguments
results = model.predict(r'C:\\Projects\\ArrowDetection\\images\\page1.jpg', save=True, save_txt = True,save_crop = True)