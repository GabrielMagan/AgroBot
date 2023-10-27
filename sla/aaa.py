from ultralytics import YOLO

model = YOLO("yolov8s.pt")
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
results = model.predict(source="0", show=True)