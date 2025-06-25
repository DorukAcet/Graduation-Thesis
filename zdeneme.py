from ultralytics import YOLO

yolo_model = YOLO('yolov11_best.pt')
print("YOLO model class mapping:", yolo_model.names)
