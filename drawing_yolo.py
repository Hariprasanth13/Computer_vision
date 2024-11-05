from ultralytics import YOLO


model = YOLO("yolov8n.yaml")  # build a new model from YAML

results = model.train(data="conf\\config.yaml", epochs=100)