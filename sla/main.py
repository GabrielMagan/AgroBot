import cv2
import argparse

from ultralytics import YOLO
import supervision as sv

import time


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution",
        default=[1280, 720],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("best.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        start = time.time()
        result = model(frame, agnostic_nms=True)[0]

        detections = sv.Detections.from_ultralytics(result)
        labels = [
            f"{model.model.names[class_id]} {round((confidence*100),2 ) } %"
            for xy, mask, confidence, class_id, _
            in detections
        ]
        frame = box_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )

        end = time.time()
        fps_label = f'FPS: {round((1.0 / (end - start)), 2)}'
        cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),5)
        cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.imshow("AgroBot Vision", frame)

        if (cv2.waitKey(1) == 27):
            break


if __name__ == "__main__":
    main()