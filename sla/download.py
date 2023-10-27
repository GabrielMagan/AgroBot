# !yolo task=detect mode=train model=yolov8s.pt data="lettuce_detection-1/data.yaml" epochs=25 imgsz=800 plots=True


from roboflow import Roboflow
rf = Roboflow(api_key="zdTM6OsedstH8M2Ec4SW")
project = rf.workspace("okitha-gunasekara").project("arduino_uno")
dataset = project.version(2).download("yolov8")

