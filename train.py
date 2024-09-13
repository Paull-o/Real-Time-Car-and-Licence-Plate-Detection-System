from ultralytics import YOLO

model1 = YOLO("yolov8n.yaml")

results1 = model1.train(data="config.yaml", epochs=1)