import cv2
import numpy as np
from ultralytics import YOLO, SAM
import os

# Load models globally
yolo_model = YOLO("best.pt")
sam_model = SAM("sam2_b.pt")

def detect_and_segment(image_path, conf=0.25, overlay_alpha=0.35):
    yolo_results = yolo_model(image_path, conf=conf)
    image = cv2.imread(image_path)
    detections = []

    for result in yolo_results:
        boxes = result.boxes

        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]
            conf_score = float(box.conf[0])

            detections.append({
                "class": cls_name,
                "confidence": round(conf_score, 2),
                "bbox": xyxy
            })

            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{cls_name} {conf_score:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

            if cls_name.lower() != "no_tumor":
                mask_result = sam_model.predict(result.orig_img, bboxes=[xyxy])
                mask = mask_result[0].masks[0].data.cpu().numpy().squeeze().astype(np.float32)
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
                mask = (mask * 255).astype("uint8")

                overlay = image.copy()
                overlay[mask > 0] = [255, 0, 0]
                image = cv2.addWeighted(overlay, overlay_alpha, image, 1 - overlay_alpha, 0)

    return image, detections
