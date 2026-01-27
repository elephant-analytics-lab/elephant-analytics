from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parents[3]
MODEL_PATH = BASE_DIR / "models" / "elephant_yolov8n_all_v4.pt"
FRAME_STRIDE = 30  # process every 30th frame to keep runtime reasonable
SHORT_VIDEO_MAX_FRAMES = 180  # ~6s at 30fps; scan every frame for short clips
IOU_THRESHOLD = 0.3


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def detect_image(image_path: Path, output_dir: Path) -> list[dict]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model weights: {MODEL_PATH}")

    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError("Failed to read image")

    results = model.predict(source=image, verbose=False)
    detections: list[dict] = []
    if results:
        res = results[0]
        names = res.names or {}
        next_id = 1
        for box in res.boxes:
            cls_id = int(box.cls[0]) if box.cls is not None else -1
            conf = float(box.conf[0]) if box.conf is not None else 0.0
            xyxy = np.array(box.xyxy[0].tolist(), dtype=np.float32)
            track_id = next_id
            next_id += 1
            detections.append(
                {
                    "frame_index": 0,
                    "track_id": track_id,
                    "cls_id": cls_id,
                    "cls_label": names.get(cls_id),
                    "conf": conf,
                    "x1": float(xyxy[0]),
                    "y1": float(xyxy[1]),
                    "x2": float(xyxy[2]),
                    "y2": float(xyxy[3]),
                }
            )
            x1, y1, x2, y2 = xyxy.astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
            label = names.get(cls_id, "elephant")
            cv2.putText(
                image,
                f"ID {track_id} {label} {conf:.2f}",
                (x1, max(y1 - 6, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    frame_path = output_dir / "frame_000000.jpg"
    cv2.imwrite(str(frame_path), image)
    return detections


def detect_and_annotate(video_path: Path, output_dir: Path) -> list[dict]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model weights: {MODEL_PATH}")

    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Failed to open video")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = 1 if total_frames and total_frames <= SHORT_VIDEO_MAX_FRAMES else FRAME_STRIDE

    detections: list[dict] = []
    prev_boxes: list[np.ndarray] = []
    prev_ids: list[int] = []
    next_id = 1

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % stride != 0:
            frame_index += 1
            continue

        results = model.predict(source=frame, verbose=False)
        if not results:
            frame_index += 1
            continue

        res = results[0]
        names = res.names or {}
        frame_boxes: list[np.ndarray] = []
        frame_ids: list[int] = []

        for box in res.boxes:
            cls_id = int(box.cls[0]) if box.cls is not None else -1
            conf = float(box.conf[0]) if box.conf is not None else 0.0
            xyxy = np.array(box.xyxy[0].tolist(), dtype=np.float32)

            match_id = None
            best_iou = 0.0
            for prev_box, prev_id in zip(prev_boxes, prev_ids):
                score = _iou(xyxy, prev_box)
                if score > best_iou and score >= IOU_THRESHOLD:
                    best_iou = score
                    match_id = prev_id

            if match_id is None:
                match_id = next_id
                next_id += 1

            frame_boxes.append(xyxy)
            frame_ids.append(match_id)

            detections.append(
                {
                    "frame_index": frame_index,
                    "track_id": match_id,
                    "cls_id": cls_id,
                    "cls_label": names.get(cls_id),
                    "conf": conf,
                    "x1": float(xyxy[0]),
                    "y1": float(xyxy[1]),
                    "x2": float(xyxy[2]),
                    "y2": float(xyxy[3]),
                }
            )

        for xyxy, track_id, cls_id, conf in zip(
            frame_boxes,
            frame_ids,
            [int(b.cls[0]) if b.cls is not None else -1 for b in res.boxes],
            [float(b.conf[0]) if b.conf is not None else 0.0 for b in res.boxes],
        ):
            x1, y1, x2, y2 = xyxy.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            label = names.get(cls_id, "elephant")
            cv2.putText(
                frame,
                f"ID {track_id} {label} {conf:.2f}",
                (x1, max(y1 - 6, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        frame_path = output_dir / f"frame_{frame_index:06d}.jpg"
        cv2.imwrite(str(frame_path), frame)

        prev_boxes = frame_boxes
        prev_ids = frame_ids
        frame_index += 1

    cap.release()
    return detections
