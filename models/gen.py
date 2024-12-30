
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.export(format="tflite", int8=True, imgsz=320, data="coco128.yml")
