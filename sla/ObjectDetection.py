import cv2
from ultralytics import YOLO
import supervision as sv

def main():
    model = YOLO("last.pt")

    box_annotator = sv.BoxAnnotator(thickness=2,text_thickness=2,text_scale=1, text_color=sv.Color.white(), color=sv.Color.green())
    frame = cv2.imread("arduino.jpg")
    frame = cv2.flip(frame, 1)
    result = model(frame, agnostic_nms=True)[0]
    detections = sv.Detections.from_ultralytics(result)
    labels = [
        f"{model.model.names[class_id]} {round((confidence * 100), 2)} %"
        for xy, mask, confidence, class_id, _
        in detections
    ]
    frame = box_annotator.annotate(scene=frame,detections=detections,labels=labels)
    cv2.imshow("AgroBot Vision", frame)

    cv2.waitKey(0)


if __name__ == "__main__":
    main()
