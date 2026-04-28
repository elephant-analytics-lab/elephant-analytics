import os
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parents[3]
MODEL_PATH = BASE_DIR / "models" / "best.pt"
# Default to stride 4 for faster turnaround on long videos.
# Keep it configurable so we can tune per machine/client without code edits.
FRAME_STRIDE = max(1, int(os.getenv("EA_FRAME_STRIDE", "4")))
SHORT_VIDEO_MAX_FRAMES = 180  # ~6s at 30fps; scan every frame for short clips
IOU_THRESHOLD = 0.3
INFERENCE_DEVICE = os.getenv("EA_INFERENCE_DEVICE", "auto").strip().lower() or "auto"
USE_TRACKER = True
TRACKER_ALGO = "botsort"
TRACKER_CONFIG = str(Path(__file__).resolve().with_name("botsort_elephant.yaml"))
VIDEO_CONF = 0.20
VIDEO_IOU = 0.50
VIDEO_IMGSZ = max(320, int(os.getenv("EA_VIDEO_IMGSZ", "960")))
WRITE_ANNOTATED_FRAMES = os.getenv("EA_WRITE_ANNOTATED_FRAMES", "1").strip().lower() not in {"0", "false", "no", "off"}
SAVE_EVERY_NTH_FRAME = max(1, int(os.getenv("EA_SAVE_EVERY_NTH_FRAME", "1")))
FRAME_JPEG_QUALITY = max(40, min(100, int(os.getenv("EA_FRAME_JPEG_QUALITY", "85"))))
RECONNECT_MAX_GAP_FRAMES = 360
RECONNECT_IOU_THRESHOLD = 0.15
RECONNECT_CENTER_DISTANCE = 0.95
PREV_FRAME_IOU_RECOVER = 0.20
PREV_FRAME_CENTER_RECOVER = 0.55
TAIL_GUARD_FRAMES = 450
TAIL_GUARD_IOU_RECOVER = 0.04
TAIL_GUARD_CENTER_RECOVER = 0.95
TAIL_GUARD_MIN_NEW_TRACK_CONF = 0.60
CLASS_MISMATCH_PENALTY = 0.03
DISPLAY_ID_REUSE_GAP_FRAMES = 240
BOX_THICKNESS = 2
LABEL_FONT_SCALE = 0.52
LABEL_FONT_THICKNESS = 1
TAG_ALPHA = 0.90
CORNER_LENGTH = 14
LABEL_MARGIN = 4
ID_ACCENT_COLOR = (0, 255, 255)    # classic yellow (BGR)
ID_TAG_BG = (0, 255, 255)          # yellow tag background
ID_TEXT_COLOR = (20, 20, 20)       # dark text
ID_STROKE_COLOR = (0, 0, 0)        # black outline for contrast

LAST_TRACKING_DIAGNOSTICS: dict[str, object] = {}
_YOLO_MODEL_CACHE: dict[str, YOLO] = {}

PIPELINE_MODE_PRESETS: dict[str, dict[str, object]] = {
    "fast": {
        "frame_stride": 4,
        "video_imgsz": 832,
        "video_conf": 0.22,
        "video_iou": 0.50,
        "write_annotated_frames": True,
        "save_every_nth_frame": 2,
        "frame_jpeg_quality": 80,
        "use_tracker": False,
        "target_sample_fps": 2.0,
    },
    "fast_trend": {
        "frame_stride": 4,
        "video_imgsz": 832,
        "video_conf": 0.22,
        "video_iou": 0.50,
        "write_annotated_frames": True,
        "save_every_nth_frame": 2,
        "frame_jpeg_quality": 80,
        "use_tracker": False,
        "target_sample_fps": 2.0,
    },
    "quality": {
        "frame_stride": 2,
        "video_imgsz": 960,
        "video_conf": 0.20,
        "video_iou": 0.50,
        "write_annotated_frames": True,
        "save_every_nth_frame": 1,
        "frame_jpeg_quality": 88,
        "use_tracker": True,
        "target_sample_fps": 0.0,
    },
}


def _resolve_inference_device() -> str:
    if INFERENCE_DEVICE != "auto":
        return INFERENCE_DEVICE
    try:
        import torch  # lazy import to keep startup light

        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _get_yolo_model(model_path: Path) -> YOLO:
    key = str(model_path.resolve())
    model = _YOLO_MODEL_CACHE.get(key)
    if model is not None:
        return model
    model = YOLO(key)
    _YOLO_MODEL_CACHE[key] = model
    return model


def _save_frame(frame_path: Path, frame: np.ndarray, jpeg_quality: int | None = None) -> None:
    quality = int(jpeg_quality if jpeg_quality is not None else FRAME_JPEG_QUALITY)
    quality = max(40, min(100, quality))
    cv2.imwrite(str(frame_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])


def _draw_detection_lightweight(
    frame: np.ndarray,
    box: np.ndarray,
    label_text: str,
    color: tuple[int, int, int] = (96, 165, 250),
) -> None:
    x1, y1, x2, y2 = [int(v) for v in box.tolist()]
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = str(label_text)[:32]
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    tx1 = x1
    ty2 = max(th + baseline + 2, y1)
    ty1 = max(0, ty2 - (th + baseline + 4))
    tx2 = min(w - 1, tx1 + tw + 8)
    cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), color, -1)
    cv2.putText(
        frame,
        text,
        (tx1 + 4, ty2 - baseline - 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )


def _resolve_runtime_video_config(pipeline_mode: str | None) -> dict[str, object]:
    cfg: dict[str, object] = {
        "frame_stride": FRAME_STRIDE,
        "video_imgsz": VIDEO_IMGSZ,
        "video_conf": VIDEO_CONF,
        "video_iou": VIDEO_IOU,
        "write_annotated_frames": WRITE_ANNOTATED_FRAMES,
        "save_every_nth_frame": SAVE_EVERY_NTH_FRAME,
        "frame_jpeg_quality": FRAME_JPEG_QUALITY,
        "use_tracker": USE_TRACKER,
        "target_sample_fps": 0.0,
    }
    mode = str(pipeline_mode or "").strip().lower()
    if mode and mode in PIPELINE_MODE_PRESETS:
        cfg.update(PIPELINE_MODE_PRESETS[mode])

    cfg["frame_stride"] = max(1, int(cfg["frame_stride"]))
    cfg["video_imgsz"] = max(320, int(cfg["video_imgsz"]))
    cfg["video_conf"] = float(cfg["video_conf"])
    cfg["video_iou"] = float(cfg["video_iou"])
    cfg["write_annotated_frames"] = bool(cfg["write_annotated_frames"])
    cfg["save_every_nth_frame"] = max(1, int(cfg["save_every_nth_frame"]))
    cfg["frame_jpeg_quality"] = max(40, min(100, int(cfg["frame_jpeg_quality"])))
    cfg["use_tracker"] = bool(cfg["use_tracker"])
    cfg["target_sample_fps"] = max(0.0, float(cfg["target_sample_fps"]))
    cfg["mode"] = mode or "default"
    return cfg


def estimate_sampled_frame_target(
    video_path: Path,
    pipeline_mode: str | None = None,
) -> dict[str, float | int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Failed to open video")
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()

    runtime_cfg = _resolve_runtime_video_config(pipeline_mode)
    stride = int(runtime_cfg["frame_stride"])
    run_target_sample_fps = float(runtime_cfg["target_sample_fps"])
    run_save_every_nth = int(runtime_cfg["save_every_nth_frame"])

    if run_target_sample_fps > 0 and video_fps > 0:
        stride = max(1, int(round(video_fps / run_target_sample_fps)))
    if total_frames and total_frames <= SHORT_VIDEO_MAX_FRAMES:
        stride = 1

    sampled_frames = 0
    if total_frames > 0:
        sampled_frames = ((total_frames - 1) // stride) + 1
    saved_frames = 0
    if sampled_frames > 0:
        saved_frames = ((sampled_frames - 1) // max(1, run_save_every_nth)) + 1

    return {
        "total_frames": total_frames,
        "source_fps": video_fps,
        "frame_stride": stride,
        "save_every_nth_frame": run_save_every_nth,
        "sampled_frame_target": sampled_frames,
        "saved_frame_target": saved_frames,
    }


def _format_age_label(cls_label: object, cls_id: int) -> str:
    txt = str(cls_label or "").strip().lower()
    if "baby" in txt or "calf" in txt:
        return "Baby"
    if "juvenile" in txt or "subadult" in txt:
        return "Juvenile"
    if "adult" in txt:
        return "Adult"
    return f"Class {cls_id}" if cls_id >= 0 else "Unknown"

PHOTO_SETTINGS = {
    "default": {
        "conf": 0.35,
        "non_baby_min_conf": 0.35,
        "baby_min_conf": 0.35,
        "iou": 0.4,
        "imgsz": 640,
        "duplicate_iou": 0.72,
        "cross_class_duplicate_iou": 1.01,
        "cross_class_max_conf": -1.0,
        "cross_class_confidence_gap": 1.0,
        "center_distance": 0.18,
        "min_area_ratio": 0.55,
        "confidence_gap": 0.08,
        "strict_duplicate_iou": 0.9,
        "strict_center_distance": 0.08,
        "strict_area_ratio": 0.7,
        "tile_enabled": False,
        "tile_overlap": 0.2,
    },
    "labeling_data_v2_s": {
        "conf": 0.16,
        "non_baby_min_conf": 0.24,
        "baby_min_conf": 0.15,
        "iou": 0.45,
        "imgsz": 960,
        "duplicate_iou": 0.74,
        "cross_class_duplicate_iou": 0.68,
        "cross_class_max_conf": 0.55,
        "cross_class_confidence_gap": 0.03,
        "center_distance": 0.14,
        "min_area_ratio": 0.62,
        "confidence_gap": 0.03,
        "strict_duplicate_iou": 0.88,
        "strict_center_distance": 0.08,
        "strict_area_ratio": 0.72,
        "small_object_area_ratio": 0.01,
        "small_object_conf_relax": 0.03,
        "large_object_area_ratio": 0.08,
        "large_object_conf_boost": 0.22,
        "large_duplicate_iou": 0.55,
        "large_confidence_gap": 0.04,
        "containment_ratio": 0.9,
        "containment_confidence_gap": 0.02,
        "tile_enabled": False,
        "tile_overlap": 0.25,
    },
    "labeling_data_v2_s_small": {
        "conf": 0.12,
        "non_baby_min_conf": 0.2,
        "baby_min_conf": 0.12,
        "iou": 0.45,
        "imgsz": 960,
        "duplicate_iou": 0.8,
        "cross_class_duplicate_iou": 0.74,
        "cross_class_max_conf": 0.5,
        "cross_class_confidence_gap": 0.05,
        "center_distance": 0.14,
        "min_area_ratio": 0.62,
        "confidence_gap": 0.06,
        "strict_duplicate_iou": 0.88,
        "strict_center_distance": 0.08,
        "strict_area_ratio": 0.72,
        "small_object_area_ratio": 0.02,
        "small_object_conf_relax": 0.07,
        "large_object_area_ratio": 0.12,
        "large_object_conf_boost": 0.1,
        "large_duplicate_iou": 0.65,
        "large_confidence_gap": 0.05,
        "containment_ratio": 0.92,
        "containment_confidence_gap": 0.03,
        "tile_enabled": False,
        "tile_overlap": 0.25,
    },
    "labeling_data_v2_s_large": {
        "conf": 0.18,
        "non_baby_min_conf": 0.3,
        "baby_min_conf": 0.18,
        "iou": 0.45,
        "imgsz": 960,
        "duplicate_iou": 0.72,
        "cross_class_duplicate_iou": 0.62,
        "cross_class_max_conf": 0.55,
        "cross_class_confidence_gap": 0.02,
        "center_distance": 0.14,
        "min_area_ratio": 0.62,
        "confidence_gap": 0.02,
        "strict_duplicate_iou": 0.86,
        "strict_center_distance": 0.1,
        "strict_area_ratio": 0.7,
        "small_object_area_ratio": 0.01,
        "small_object_conf_relax": 0.02,
        "large_object_area_ratio": 0.08,
        "large_object_conf_boost": 0.3,
        "large_duplicate_iou": 0.52,
        "large_confidence_gap": 0.02,
        "containment_ratio": 0.84,
        "containment_confidence_gap": 0.0,
        "max_box_aspect_ratio": 2.2,
        "border_large_area_ratio": 0.06,
        "border_touch_edges": 2,
        "border_low_conf_max": 0.5,
        "tile_enabled": False,
        "tile_overlap": 0.25,
    },
    "roboflow_v1_fresh2": {
        "conf": 0.3,
        "non_baby_min_conf": 0.3,
        "baby_min_conf": 0.3,
        "iou": 0.45,
        "imgsz": 640,
        "duplicate_iou": 0.78,
        "cross_class_duplicate_iou": 1.01,
        "cross_class_max_conf": -1.0,
        "cross_class_confidence_gap": 1.0,
        "center_distance": 0.16,
        "min_area_ratio": 0.58,
        "confidence_gap": 0.1,
        "strict_duplicate_iou": 0.88,
        "strict_center_distance": 0.08,
        "strict_area_ratio": 0.7,
        "tile_enabled": False,
        "tile_overlap": 0.2,
    },
}

PHOTO_SETTINGS_DEFAULTS = {
    "small_object_area_ratio": 0.0,
    "small_object_conf_relax": 0.0,
    "large_object_area_ratio": 1.01,
    "large_object_conf_boost": 0.0,
    "large_duplicate_iou": 0.82,
    "large_confidence_gap": 0.1,
    "containment_ratio": 1.01,
    "containment_confidence_gap": 1.0,
    "max_box_aspect_ratio": 999.0,
    "border_large_area_ratio": 1.01,
    "border_touch_edges": 99,
    "border_low_conf_max": -1.0,
}

# Keep the registry aligned with the two customer-facing model choices.
# Legacy aliases stay mapped so older saved jobs can still resolve the same file.
MODEL_REGISTRY = {
    "best": BASE_DIR / "models" / "best.pt",
    "labeling_data_v2_s": BASE_DIR / "models" / "elephant_v3_labeling_data_v2_s.pt",
    "labeling_data_v2_s_small": BASE_DIR / "models" / "elephant_v3_labeling_data_v2_s.pt",
    "labeling_data_v2_s_large": BASE_DIR / "models" / "elephant_v3_labeling_data_v2_s.pt",
    "roboflow_v1_fresh2": BASE_DIR / "models" / "elephant_online_roboflow_v1.pt",
}


def available_models() -> dict[str, str]:
    return {k: str(v) for k, v in MODEL_REGISTRY.items() if v.exists()}


def resolve_model_path(model_key: str | None) -> Path:
    if model_key and model_key in MODEL_REGISTRY and MODEL_REGISTRY[model_key].exists():
        return MODEL_REGISTRY[model_key]
    if MODEL_PATH.exists():
        return MODEL_PATH
    best = MODEL_REGISTRY["best"]
    if best.exists():
        return best
    raise FileNotFoundError(
        "Missing model weights. Put a model under models/ (e.g. elephant_yolov8n_all_v4.pt or best.pt)."
    )


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


def _box_area(box: np.ndarray) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _center_distance_ratio(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax = (box_a[0] + box_a[2]) / 2.0
    ay = (box_a[1] + box_a[3]) / 2.0
    bx = (box_b[0] + box_b[2]) / 2.0
    by = (box_b[1] + box_b[3]) / 2.0
    w = max(box_a[2] - box_a[0], box_b[2] - box_b[0], 1.0)
    h = max(box_a[3] - box_a[1], box_b[3] - box_b[1], 1.0)
    diag = max((w**2 + h**2) ** 0.5, 1.0)
    return (((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5) / diag


def _edge_anchor(
    box: np.ndarray,
    frame_w: int,
    frame_h: int,
) -> str:
    # Border exits are one of the few reliable hints we get when an animal
    # leaves the frame and reappears shortly after on the same side.
    margin_x = max(72.0, frame_w * 0.08)
    margin_y = max(54.0, frame_h * 0.08)
    if float(box[0]) <= margin_x:
        return "left"
    if float(box[2]) >= max(frame_w - margin_x, 0):
        return "right"
    if float(box[1]) <= margin_y:
        return "top"
    if float(box[3]) >= max(frame_h - margin_y, 0):
        return "bottom"
    return ""


def _resolve_canonical_track_id(
    raw_track_id: int,
    box: np.ndarray,
    cls_id: int,
    frame_index: int,
    frame_w: int,
    frame_h: int,
    raw_to_canonical: dict[int, int],
    canonical_last_seen: dict[int, dict[str, object]],
    used_canonical_in_frame: set[int],
    next_id: int,
    diag: dict[str, int] | None = None,
) -> tuple[int, int]:
    canonical = raw_to_canonical.get(raw_track_id)
    if canonical is not None:
        if canonical in used_canonical_in_frame:
            return -1, next_id
        return canonical, next_id

    best_id = None
    best_score = 0.0
    current_edge = _edge_anchor(box, frame_w, frame_h)
    for candidate_id, meta in canonical_last_seen.items():
        if candidate_id in used_canonical_in_frame:
            continue
        last_frame = int(meta["frame_index"])
        gap = frame_index - last_frame
        if gap <= 0 or gap > RECONNECT_MAX_GAP_FRAMES:
            continue
        prev_box = np.array(meta["box"], dtype=np.float32)
        overlap = _iou(box, prev_box)
        center_dist = _center_distance_ratio(box, prev_box)
        previous_edge = str(meta.get("edge_anchor") or "")
        edge_reentry = bool(current_edge and current_edge == previous_edge)
        # Adjacent sampled frames can have larger camera displacement; allow looser
        # geometric gating only for very short gaps to reduce sudden late-video ID resets.
        short_gap = gap <= (FRAME_STRIDE * 3)
        gate_iou = 0.0 if edge_reentry else (0.02 if short_gap else RECONNECT_IOU_THRESHOLD)
        gate_center = 1.45 if edge_reentry else (1.15 if short_gap else RECONNECT_CENTER_DISTANCE)
        if overlap < gate_iou and center_dist > gate_center:
            continue
        class_penalty = CLASS_MISMATCH_PENALTY if int(meta["cls_id"]) != cls_id else 0.0
        # Prefer larger overlap, then smaller center distance; mild penalty for class mismatch.
        edge_bonus = 0.08 if edge_reentry else 0.0
        score = overlap + max(0.0, (gate_center - center_dist) * 0.25) + edge_bonus - class_penalty
        if score > best_score:
            best_score = score
            best_id = candidate_id

    if best_id is not None:
        canonical = int(best_id)
        if diag is not None:
            diag["history_reconnects"] = int(diag.get("history_reconnects", 0)) + 1
    else:
        canonical = next_id
        next_id += 1
        if diag is not None:
            diag["new_ids_created"] = int(diag.get("new_ids_created", 0)) + 1

    raw_to_canonical[raw_track_id] = canonical
    return canonical, next_id


def _recover_prev_frame_track_id(
    box: np.ndarray,
    cls_id: int,
    prev_boxes: list[np.ndarray],
    prev_ids: list[int],
    prev_cls_ids: list[int],
    used_canonical_in_frame: set[int],
) -> int | None:
    best_id = None
    best_score = 0.0
    for pbox, pid, pcls in zip(prev_boxes, prev_ids, prev_cls_ids):
        if pid in used_canonical_in_frame:
            continue
        overlap = _iou(box, pbox)
        center_dist = _center_distance_ratio(box, pbox)
        if overlap < PREV_FRAME_IOU_RECOVER and center_dist > PREV_FRAME_CENTER_RECOVER:
            continue
        class_penalty = CLASS_MISMATCH_PENALTY if pcls != cls_id else 0.0
        score = overlap + max(0.0, (PREV_FRAME_CENTER_RECOVER - center_dist) * 0.35) - class_penalty
        if score > best_score:
            best_score = score
            best_id = pid
    return best_id


def _recover_tail_track_id(
    box: np.ndarray,
    cls_id: int,
    frame_w: int,
    frame_h: int,
    canonical_last_seen: dict[int, dict[str, object]],
    frame_index: int,
    used_canonical_in_frame: set[int],
) -> int | None:
    best_id = None
    best_score = 0.0
    current_edge = _edge_anchor(box, frame_w, frame_h)
    for cid, meta in canonical_last_seen.items():
        if cid in used_canonical_in_frame:
            continue
        prev_box = np.array(meta["box"], dtype=np.float32)
        overlap = _iou(box, prev_box)
        center_dist = _center_distance_ratio(box, prev_box)
        previous_edge = str(meta.get("edge_anchor") or "")
        edge_reentry = bool(current_edge and current_edge == previous_edge)
        gate_iou = 0.0 if edge_reentry else TAIL_GUARD_IOU_RECOVER
        gate_center = 1.55 if edge_reentry else TAIL_GUARD_CENTER_RECOVER
        if overlap < gate_iou and center_dist > gate_center:
            continue
        gap = frame_index - int(meta["frame_index"])
        if gap <= 0 or gap > RECONNECT_MAX_GAP_FRAMES * 2:
            continue
        class_penalty = CLASS_MISMATCH_PENALTY if int(meta["cls_id"]) != cls_id else 0.0
        edge_bonus = 0.1 if edge_reentry else 0.0
        score = overlap + max(0.0, (gate_center - center_dist) * 0.2) + edge_bonus - class_penalty
        if score > best_score:
            best_score = score
            best_id = cid
    return best_id


def get_last_tracking_diagnostics() -> dict[str, object]:
    return dict(LAST_TRACKING_DIAGNOSTICS)


def _compact_display_id(
    canonical_id: int,
    frame_index: int,
    canonical_last_seen: dict[int, dict[str, object]],
    canonical_to_display: dict[int, int],
    display_last_seen: dict[int, int],
    used_display_in_frame: set[int] | None = None,
) -> int:
    # Release stale canonical->display bindings so small ID numbers can be reused.
    for cid in list(canonical_to_display.keys()):
        meta = canonical_last_seen.get(cid)
        if not meta:
            canonical_to_display.pop(cid, None)
            continue
        if frame_index - int(meta["frame_index"]) > DISPLAY_ID_REUSE_GAP_FRAMES:
            canonical_to_display.pop(cid, None)

    if canonical_id in canonical_to_display:
        display_id = canonical_to_display[canonical_id]
        if used_display_in_frame is not None and display_id in used_display_in_frame:
            canonical_to_display.pop(canonical_id, None)
        else:
            display_last_seen[display_id] = frame_index
            if used_display_in_frame is not None:
                used_display_in_frame.add(display_id)
            return display_id

    used = set(canonical_to_display.values())
    if used_display_in_frame is not None:
        used |= set(used_display_in_frame)
    reusable = [
        did
        for did, last_frame in display_last_seen.items()
        if did not in used and (frame_index - int(last_frame)) > DISPLAY_ID_REUSE_GAP_FRAMES
    ]
    if reusable:
        display_id = min(reusable)
    else:
        display_id = (max(used) + 1) if used else 1

    canonical_to_display[canonical_id] = display_id
    display_last_seen[display_id] = frame_index
    if used_display_in_frame is not None:
        used_display_in_frame.add(display_id)
    return display_id


def _intersection_over_smaller(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = _box_area(box_a)
    area_b = _box_area(box_b)
    smaller = min(area_a, area_b)
    return inter / smaller if smaller > 0 else 0.0


def _color_for_id(track_id: int) -> tuple[int, int, int]:
    return ID_ACCENT_COLOR


def _rect_intersection_area(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def _draw_detection_card(
    frame: np.ndarray,
    box: np.ndarray,
    label_text: str,
    color: tuple[int, int, int],
    occupied_tag_rects: list[tuple[int, int, int, int]] | None = None,
    avoid_rects: list[tuple[int, int, int, int]] | None = None,
) -> None:
    x1, y1, x2, y2 = box.astype(int)
    frame_h, frame_w = frame.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_w - 1, x2)
    y2 = min(frame_h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return

    # Scale annotation thickness by resolution so labels remain readable on wide aerial frames.
    scale_ref = min(frame_h, frame_w) / 1080.0
    box_thickness = max(2, int(round(BOX_THICKNESS * scale_ref)))
    box_outline_thickness = box_thickness + 1
    font_scale = float(np.clip(LABEL_FONT_SCALE + (0.22 * scale_ref), 0.60, 0.95))
    font_thickness = max(1, int(round(LABEL_FONT_THICKNESS * scale_ref)))
    text_outline_thickness = font_thickness + 2

    # Double-stroke box improves contrast on bright grass without changing tracking logic.
    cv2.rectangle(frame, (x1, y1), (x2, y2), ID_STROKE_COLOR, box_outline_thickness, cv2.LINE_AA)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness, cv2.LINE_AA)

    label = label_text
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    pad_x = max(8, int(round(10 * scale_ref)))
    pad_y = max(6, int(round(8 * scale_ref)))
    tag_h = th + (pad_y * 2) + text_outline_thickness
    tag_w = tw + (pad_x * 2) + (text_outline_thickness * 2) + 4

    # Avoid label overlap in crowded scenes by shifting down in small steps.
    if occupied_tag_rects is None:
        occupied_tag_rects = []

    if avoid_rects is None:
        avoid_rects = []

    # Candidate anchors around each box; choose placement that avoids overlap with tags and body boxes.
    candidate_anchors = [
        (x1, y1 - tag_h - 6),          # top-left
        (x2 - tag_w, y1 - tag_h - 6),  # top-right
        (x1, y2 + 6),                  # bottom-left
        (x2 - tag_w, y2 + 6),          # bottom-right
        (x1 - tag_w - 6, y1),          # left
        (x2 + 6, y1),                  # right
    ]

    max_tag_x1 = max(0, frame_w - tag_w - 1)
    max_tag_y1 = max(0, frame_h - tag_h - 1)
    own_rect = (x1, y1, x2, y2)
    tag_area = max(1, tag_w * tag_h)

    best_rect: tuple[int, int, int, int] | None = None
    best_score: tuple[int, float, float] | None = None
    for ax, ay in candidate_anchors:
        tx1 = int(np.clip(ax, 0, max_tag_x1))
        ty1 = int(np.clip(ay, 0, max_tag_y1))
        tx2 = min(frame_w - 1, tx1 + tag_w)
        ty2 = min(frame_h - 1, ty1 + tag_h)
        candidate_rect = (tx1, ty1, tx2, ty2)

        tag_overlap = 0
        for occ in occupied_tag_rects:
            if _rect_intersection_area(candidate_rect, occ) > 0:
                tag_overlap = 1
                break

        body_overlap_area = 0
        for rect in avoid_rects:
            # Ignore own box to allow normal edge placement.
            if all(abs(a - b) <= 2 for a, b in zip(rect, own_rect)):
                continue
            body_overlap_area += _rect_intersection_area(candidate_rect, rect)
        body_overlap_ratio = body_overlap_area / float(tag_area)

        # Slight preference for top placements to keep labels visually consistent.
        vertical_bias = abs((y1 - tag_h - 6) - ty1) / max(1.0, float(frame_h))
        score = (tag_overlap, body_overlap_ratio, vertical_bias)
        if best_score is None or score < best_score:
            best_score = score
            best_rect = candidate_rect

    if best_rect is None:
        tag_x1 = int(np.clip(x1, 0, max_tag_x1))
        tag_y1 = int(np.clip(y1 - tag_h - 6, 0, max_tag_y1))
        tag_x2 = min(frame_w - 1, tag_x1 + tag_w)
        tag_y2 = min(frame_h - 1, tag_y1 + tag_h)
    else:
        tag_x1, tag_y1, tag_x2, tag_y2 = best_rect
    occupied_tag_rects.append((tag_x1, tag_y1, tag_x2, tag_y2))

    overlay = frame.copy()
    cv2.rectangle(overlay, (tag_x1, tag_y1), (tag_x2, tag_y2), ID_TAG_BG, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, TAG_ALPHA, frame, 1.0 - TAG_ALPHA, 0, frame)
    cv2.rectangle(frame, (tag_x1, tag_y1), (tag_x2, tag_y2), ID_STROKE_COLOR, 1, cv2.LINE_AA)
    text_x = min(tag_x1 + pad_x + text_outline_thickness, max(0, frame_w - tw - 2))
    text_y = tag_y2 - baseline - pad_y
    cv2.putText(
        frame,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        ID_STROKE_COLOR,
        text_outline_thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        ID_TEXT_COLOR,
        font_thickness,
        cv2.LINE_AA,
    )


def _resolve_photo_settings(model_key: str | None) -> dict[str, float]:
    resolved = PHOTO_SETTINGS.get(model_key or "", PHOTO_SETTINGS["default"])
    return {**PHOTO_SETTINGS_DEFAULTS, **resolved}


def _passes_size_aware_confidence(
    det: dict,
    settings: dict[str, float],
    image_area: float,
) -> bool:
    score = float(det["conf"])
    label = (det.get("cls_label") or "").lower()
    threshold = float(settings["baby_min_conf"]) if "baby" in label else float(settings["non_baby_min_conf"])

    box = np.array([det["x1"], det["y1"], det["x2"], det["y2"]], dtype=np.float32)
    area_ratio = (_box_area(box) / image_area) if image_area > 0 else 0.0

    # Allow weaker confidence for tiny elephants.
    if area_ratio <= float(settings["small_object_area_ratio"]):
        threshold -= float(settings["small_object_conf_relax"])
    # Demand stronger confidence for very large boxes to reduce noisy large-object hits.
    if area_ratio >= float(settings["large_object_area_ratio"]):
        threshold += float(settings["large_object_conf_boost"])

    threshold = max(threshold, float(settings["conf"]))
    return score >= threshold


def _passes_shape_and_border_filter(
    det: dict,
    settings: dict[str, float],
    image_w: int,
    image_h: int,
    image_area: float,
) -> bool:
    box = np.array([det["x1"], det["y1"], det["x2"], det["y2"]], dtype=np.float32)
    bw = max(0.0, float(box[2] - box[0]))
    bh = max(0.0, float(box[3] - box[1]))
    if bw <= 1.0 or bh <= 1.0:
        return False

    aspect = max(bw / max(bh, 1e-6), bh / max(bw, 1e-6))
    if aspect > float(settings["max_box_aspect_ratio"]):
        return False

    area_ratio = (_box_area(box) / image_area) if image_area > 0 else 0.0
    touch_eps = 2.0
    touch_edges = 0
    if box[0] <= touch_eps:
        touch_edges += 1
    if box[1] <= touch_eps:
        touch_edges += 1
    if box[2] >= (float(image_w) - touch_eps):
        touch_edges += 1
    if box[3] >= (float(image_h) - touch_eps):
        touch_edges += 1

    if (
        area_ratio >= float(settings["border_large_area_ratio"])
        and touch_edges >= int(settings["border_touch_edges"])
        and float(det["conf"]) <= float(settings["border_low_conf_max"])
    ):
        return False
    return True


def _predict_boxes(
    model: YOLO,
    image: np.ndarray,
    names: dict[int, str] | None,
    conf: float,
    iou: float,
    imgsz: int,
    offset_x: int = 0,
    offset_y: int = 0,
) -> list[dict]:
    names = names or {}
    resolved_device = _resolve_inference_device()
    use_fp16 = resolved_device.startswith("cuda")
    results = model.predict(
        source=image,
        verbose=False,
        device=resolved_device,
        half=use_fp16,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
    )
    raw_detections: list[dict] = []
    if not results:
        return raw_detections
    res = results[0]
    names = res.names or names
    for box in res.boxes:
        cls_id = int(box.cls[0]) if box.cls is not None else -1
        score = float(box.conf[0]) if box.conf is not None else 0.0
        xyxy = np.array(box.xyxy[0].tolist(), dtype=np.float32)
        raw_detections.append(
            {
                "frame_index": 0,
                "track_id": 0,
                "cls_id": cls_id,
                "cls_label": names.get(cls_id),
                "conf": score,
                "x1": float(xyxy[0] + offset_x),
                "y1": float(xyxy[1] + offset_y),
                "x2": float(xyxy[2] + offset_x),
                "y2": float(xyxy[3] + offset_y),
            }
        )
    return raw_detections


def _predict_tiled_boxes(
    model: YOLO,
    image: np.ndarray,
    settings: dict[str, float],
) -> list[dict]:
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return []
    overlap = float(settings["tile_overlap"])
    tile_w = max(w // 2, 1)
    tile_h = max(h // 2, 1)
    step_x = max(int(tile_w * (1.0 - overlap)), 1)
    step_y = max(int(tile_h * (1.0 - overlap)), 1)
    x_starts = sorted({0, max(w - tile_w, 0), step_x})
    y_starts = sorted({0, max(h - tile_h, 0), step_y})

    names: dict[int, str] = {}
    raw_detections: list[dict] = []
    for y0 in y_starts:
        for x0 in x_starts:
            x1 = min(x0 + tile_w, w)
            y1 = min(y0 + tile_h, h)
            tile = image[y0:y1, x0:x1]
            if tile.size == 0:
                continue
            raw_detections.extend(
                _predict_boxes(
                    model,
                    tile,
                    names=names,
                    conf=settings["conf"],
                    iou=settings["iou"],
                    imgsz=int(settings["imgsz"]),
                    offset_x=x0,
                    offset_y=y0,
                )
            )
    return raw_detections


def _dedupe_photo_boxes(
    raw_detections: list[dict],
    settings: dict[str, float],
    image_area: float,
) -> list[dict]:
    kept: list[dict] = []
    for det in sorted(raw_detections, key=lambda d: float(d["conf"]), reverse=True):
        box = np.array([det["x1"], det["y1"], det["x2"], det["y2"]], dtype=np.float32)
        is_duplicate = False
        for existing in kept:
            existing_box = np.array(
                [existing["x1"], existing["y1"], existing["x2"], existing["y2"]],
                dtype=np.float32,
            )
            overlap = _iou(box, existing_box)
            area = _box_area(box)
            existing_area = _box_area(existing_box)
            area_ratio = (
                min(area, existing_area) / max(area, existing_area)
                if area > 0 and existing_area > 0
                else 0.0
            )
            center_ratio = _center_distance_ratio(box, existing_box)
            strict_duplicate = (
                overlap >= settings["strict_duplicate_iou"]
                and center_ratio <= settings["strict_center_distance"]
                and area_ratio >= settings["strict_area_ratio"]
            )
            if strict_duplicate:
                is_duplicate = True
                break
            containment = _intersection_over_smaller(box, existing_box)
            if (
                containment >= float(settings["containment_ratio"])
                and (float(existing["conf"]) - float(det["conf"])) >= float(settings["containment_confidence_gap"])
            ):
                is_duplicate = True
                break
            if (
                det["cls_id"] != existing["cls_id"]
                and overlap >= settings["cross_class_duplicate_iou"]
                and float(det["conf"]) <= settings["cross_class_max_conf"]
                and (float(existing["conf"]) - float(det["conf"])) >= settings["cross_class_confidence_gap"]
            ):
                is_duplicate = True
                break
            if det["cls_id"] != existing["cls_id"]:
                continue
            adaptive_duplicate_iou = float(settings["duplicate_iou"])
            adaptive_conf_gap = float(settings["confidence_gap"])
            larger_area_ratio = max(area, existing_area) / image_area if image_area > 0 else 0.0
            if larger_area_ratio >= float(settings["large_object_area_ratio"]):
                adaptive_duplicate_iou = min(
                    adaptive_duplicate_iou,
                    float(settings["large_duplicate_iou"]),
                )
                adaptive_conf_gap = min(
                    adaptive_conf_gap,
                    float(settings["large_confidence_gap"]),
                )
            if (
                overlap >= adaptive_duplicate_iou
                and center_ratio <= settings["center_distance"]
                and area_ratio >= settings["min_area_ratio"]
                and (float(existing["conf"]) - float(det["conf"])) >= adaptive_conf_gap
            ):
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(det)
    return kept


def detect_image(image_path: Path, output_dir: Path, model_key: str | None = None) -> list[dict]:
    model_path = resolve_model_path(model_key)
    photo_settings = _resolve_photo_settings(model_key)

    output_dir.mkdir(parents=True, exist_ok=True)

    model = _get_yolo_model(model_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError("Failed to read image")
    image_h, image_w = image.shape[:2]
    image_area = float(image.shape[0] * image.shape[1])

    raw_detections = _predict_boxes(
        model,
        image,
        names={},
        conf=photo_settings["conf"],
        iou=photo_settings["iou"],
        imgsz=int(photo_settings["imgsz"]),
    )
    if photo_settings.get("tile_enabled"):
        raw_detections.extend(_predict_tiled_boxes(model, image, photo_settings))
    raw_detections = [
        det
        for det in raw_detections
        if _passes_size_aware_confidence(det, photo_settings, image_area)
        and _passes_shape_and_border_filter(det, photo_settings, image_w, image_h, image_area)
    ]

    detections = _dedupe_photo_boxes(raw_detections, photo_settings, image_area)
    occupied_tag_rects: list[tuple[int, int, int, int]] = []
    detection_rects = [
        (
            int(det["x1"]),
            int(det["y1"]),
            int(det["x2"]),
            int(det["y2"]),
        )
        for det in detections
    ]
    for idx, det in enumerate(detections, start=1):
        det["track_id"] = idx
        cls_id = int(det.get("cls_id", -1))
        age_tag = _format_age_label(det.get("cls_label"), cls_id)
        box = np.array([det["x1"], det["y1"], det["x2"], det["y2"]], dtype=np.float32)
        image_label = f"ID {idx:02d}" if age_tag.startswith("Class ") or age_tag == "Unknown" else f"ID {idx:02d} | {age_tag}"
        _draw_detection_card(
            image,
            box=box,
            label_text=image_label,
            color=_color_for_id(idx),
            occupied_tag_rects=occupied_tag_rects,
            avoid_rects=detection_rects,
        )

    frame_path = output_dir / "frame_000000.jpg"
    _save_frame(frame_path, image)
    return detections


def detect_and_annotate(
    video_path: Path,
    output_dir: Path,
    model_key: str | None = None,
    pipeline_mode: str | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> list[dict]:
    model_path = resolve_model_path(model_key)

    output_dir.mkdir(parents=True, exist_ok=True)

    model = _get_yolo_model(model_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Failed to open video")
    resolved_device = _resolve_inference_device()
    use_fp16 = resolved_device.startswith("cuda")
    runtime_cfg = _resolve_runtime_video_config(pipeline_mode)
    run_video_conf = float(runtime_cfg["video_conf"])
    run_video_iou = float(runtime_cfg["video_iou"])
    run_video_imgsz = int(runtime_cfg["video_imgsz"])
    run_write_annotated = bool(runtime_cfg["write_annotated_frames"])
    run_save_every_nth = int(runtime_cfg["save_every_nth_frame"])
    run_jpeg_quality = int(runtime_cfg["frame_jpeg_quality"])
    run_use_tracker = bool(runtime_cfg["use_tracker"])
    run_target_sample_fps = float(runtime_cfg["target_sample_fps"])

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    stride = int(runtime_cfg["frame_stride"])
    if run_target_sample_fps > 0 and video_fps > 0:
        stride = max(1, int(round(video_fps / run_target_sample_fps)))
    if total_frames and total_frames <= SHORT_VIDEO_MAX_FRAMES:
        stride = 1

    detections: list[dict] = []
    prev_boxes: list[np.ndarray] = []
    prev_ids: list[int] = []
    prev_cls_ids: list[int] = []
    next_id = 1
    tracker_mode = TRACKER_ALGO if run_use_tracker else "none"
    tracker_warned = False
    raw_to_canonical: dict[int, int] = {}
    canonical_to_display: dict[int, int] = {}
    display_last_seen: dict[int, int] = {}
    canonical_last_seen: dict[int, dict[str, object]] = {}
    # Smooth age labels per identity so rendered tags do not flip on every
    # slightly noisy single-frame classification.
    canonical_age_votes: dict[int, dict[str, int]] = {}
    raw_bytetrack_ids_seen: set[int] = set()
    diag: dict[str, int] = {
        "frames_read": 0,
        "frames_sampled": 0,
        "frames_with_detections": 0,
        "total_detections": 0,
        "history_reconnects": 0,
        "prev_frame_recoveries": 0,
        "new_ids_created": 0,
        "fallback_iou_frames": 0,
        "tail_recoveries": 0,
        "tail_new_id_suppressed": 0,
        "id_switch_overrides": 0,
    }
    detection_total_s = 0.0
    tracking_total_s = 0.0
    frame_write_total_s = 0.0
    frame_annotation_total_s = 0.0
    read_total_s = 0.0
    grab_total_s = 0.0
    yolo_preprocess_ms_total = 0.0
    yolo_inference_ms_total = 0.0
    yolo_postprocess_ms_total = 0.0
    frames_saved = 0
    fallback_used = False

    frame_index = 0
    end_of_stream = False
    while not end_of_stream:
        if should_cancel and should_cancel():
            raise RuntimeError("Processing cancelled")
        t_read = time.perf_counter()
        ret, frame = cap.read()
        read_total_s += time.perf_counter() - t_read
        if not ret:
            break
        diag["frames_read"] = int(diag["frames_read"]) + 1
        diag["frames_sampled"] = int(diag["frames_sampled"]) + 1

        if run_use_tracker and tracker_mode in {"bytetrack", "botsort"}:
            try:
                t_infer = time.perf_counter()
                results = model.track(
                    source=frame,
                    verbose=False,
                    device=resolved_device,
                    half=use_fp16,
                    persist=True,
                    tracker=TRACKER_CONFIG,
                    conf=run_video_conf,
                    iou=run_video_iou,
                    imgsz=run_video_imgsz,
                )
                infer_elapsed = time.perf_counter() - t_infer
                detection_total_s += infer_elapsed
                tracking_total_s += infer_elapsed
            except Exception:
                # Fall back to existing IoU tracking if ByteTrack is unavailable in runtime.
                tracker_mode = "iou"
                fallback_used = True
                if not tracker_warned:
                    print("Tracker unavailable; falling back to IoU tracking.")
                    tracker_warned = True
                t_infer = time.perf_counter()
                results = model.predict(
                    source=frame,
                    verbose=False,
                    device=resolved_device,
                    half=use_fp16,
                    conf=run_video_conf,
                    iou=run_video_iou,
                    imgsz=run_video_imgsz,
                )
                detection_total_s += time.perf_counter() - t_infer
        else:
            t_infer = time.perf_counter()
            results = model.predict(
                source=frame,
                verbose=False,
                device=resolved_device,
                half=use_fp16,
                conf=run_video_conf,
                iou=run_video_iou,
                imgsz=run_video_imgsz,
            )
            detection_total_s += time.perf_counter() - t_infer
            if run_use_tracker:
                diag["fallback_iou_frames"] = int(diag["fallback_iou_frames"]) + 1
        if not results:
            # Old behavior decoded every frame then skipped by modulo.
            # We now fast-skip intermediate frames with grab() to reduce decode overhead
            # while keeping the same sampled-frame cadence for tracking quality.
            skipped = 0
            for _ in range(max(stride - 1, 0)):
                if should_cancel and should_cancel():
                    end_of_stream = True
                    raise RuntimeError("Processing cancelled")
                if not cap.grab():
                    end_of_stream = True
                    break
                skipped += 1
            if skipped:
                diag["frames_read"] = int(diag["frames_read"]) + skipped
            frame_index += 1 + skipped
            continue

        res = results[0]
        speed = getattr(res, "speed", None) or {}
        yolo_preprocess_ms_total += float(speed.get("preprocess", 0.0) or 0.0)
        yolo_inference_ms_total += float(speed.get("inference", 0.0) or 0.0)
        yolo_postprocess_ms_total += float(speed.get("postprocess", 0.0) or 0.0)
        names = res.names or {}
        if len(res.boxes) > 0:
            diag["frames_with_detections"] = int(diag["frames_with_detections"]) + 1
            diag["total_detections"] = int(diag["total_detections"]) + int(len(res.boxes))
        frame_boxes: list[np.ndarray] = []
        frame_ids: list[int] = []
        frame_cls_ids: list[int] = []
        used_canonical_in_frame: set[int] = set()
        used_display_in_frame: set[int] = set()

        occupied_tag_rects: list[tuple[int, int, int, int]] = []
        frame_detection_rects = [
            (
                int(b.xyxy[0][0]),
                int(b.xyxy[0][1]),
                int(b.xyxy[0][2]),
                int(b.xyxy[0][3]),
            )
            for b in res.boxes
        ]
        if not run_use_tracker:
            fast_tag_rects: list[tuple[int, int, int, int]] = []
            for box in res.boxes:
                cls_id = int(box.cls[0]) if box.cls is not None else -1
                conf = float(box.conf[0]) if box.conf is not None else 0.0
                xyxy = np.array(box.xyxy[0].tolist(), dtype=np.float32)
                detections.append(
                    {
                        "frame_index": frame_index,
                        "track_id": None,
                        "cls_id": cls_id,
                        "cls_label": names.get(cls_id),
                        "conf": conf,
                        "x1": float(xyxy[0]),
                        "y1": float(xyxy[1]),
                        "x2": float(xyxy[2]),
                        "y2": float(xyxy[3]),
                    }
                )
                if run_write_annotated:
                    t_annot = time.perf_counter()
                    _draw_detection_card(
                        frame,
                        box=xyxy,
                        label_text=_format_age_label(names.get(cls_id), cls_id),
                        color=ID_ACCENT_COLOR,
                        occupied_tag_rects=fast_tag_rects,
                        avoid_rects=frame_detection_rects,
                    )
                    frame_annotation_total_s += time.perf_counter() - t_annot
            if (frame_index % run_save_every_nth) == 0:
                frame_path = output_dir / f"frame_{frame_index:06d}.jpg"
                t_write = time.perf_counter()
                _save_frame(frame_path, frame, jpeg_quality=run_jpeg_quality)
                frame_write_total_s += time.perf_counter() - t_write
                frames_saved += 1
            prev_boxes = []
            prev_ids = []
            prev_cls_ids = []
            skipped = 0
            for _ in range(max(stride - 1, 0)):
                if should_cancel and should_cancel():
                    end_of_stream = True
                    raise RuntimeError("Processing cancelled")
                t_grab = time.perf_counter()
                if not cap.grab():
                    grab_total_s += time.perf_counter() - t_grab
                    end_of_stream = True
                    break
                grab_total_s += time.perf_counter() - t_grab
                skipped += 1
            if skipped:
                diag["frames_read"] = int(diag["frames_read"]) + skipped
            frame_index += 1 + skipped
            continue

        for box in res.boxes:
            t_track_logic = time.perf_counter()
            cls_id = int(box.cls[0]) if box.cls is not None else -1
            conf = float(box.conf[0]) if box.conf is not None else 0.0
            xyxy = np.array(box.xyxy[0].tolist(), dtype=np.float32)
            match_id = None
            raw_track_id: int | None = None
            if tracker_mode in {"bytetrack", "botsort"} and getattr(box, "id", None) is not None:
                try:
                    raw_track_id = int(box.id[0])
                    raw_bytetrack_ids_seen.add(raw_track_id)
                    match_id, next_id = _resolve_canonical_track_id(
                        raw_track_id=raw_track_id,
                        box=xyxy,
                        cls_id=cls_id,
                        frame_index=frame_index,
                        frame_w=int(frame.shape[1]),
                        frame_h=int(frame.shape[0]),
                        raw_to_canonical=raw_to_canonical,
                        canonical_last_seen=canonical_last_seen,
                        used_canonical_in_frame=used_canonical_in_frame,
                        next_id=next_id,
                        diag=diag,
                    )
                    if match_id == -1:
                        match_id = None
                except Exception:
                    match_id = None

            # Override potentially switched raw IDs with strong previous-frame continuity.
            recovered_override = _recover_prev_frame_track_id(
                box=xyxy,
                cls_id=cls_id,
                prev_boxes=prev_boxes,
                prev_ids=prev_ids,
                prev_cls_ids=prev_cls_ids,
                used_canonical_in_frame=used_canonical_in_frame,
            )
            if recovered_override is not None and match_id is not None and recovered_override != match_id:
                match_id = recovered_override
                if raw_track_id is not None:
                    raw_to_canonical[raw_track_id] = match_id
                diag["id_switch_overrides"] = int(diag["id_switch_overrides"]) + 1

            if match_id is None:
                best_iou = 0.0
                for prev_box, prev_id in zip(prev_boxes, prev_ids):
                    if prev_id in used_canonical_in_frame:
                        continue
                    score = _iou(xyxy, prev_box)
                    if score > best_iou and score >= IOU_THRESHOLD:
                        best_iou = score
                        match_id = prev_id

            if match_id is None:
                recovered_id = _recover_prev_frame_track_id(
                    box=xyxy,
                    cls_id=cls_id,
                    prev_boxes=prev_boxes,
                    prev_ids=prev_ids,
                    prev_cls_ids=prev_cls_ids,
                    used_canonical_in_frame=used_canonical_in_frame,
                )
                if recovered_id is not None:
                    match_id = recovered_id
                    diag["prev_frame_recoveries"] = int(diag["prev_frame_recoveries"]) + 1

            if match_id is None:
                in_tail = total_frames > 0 and frame_index >= max(total_frames - TAIL_GUARD_FRAMES, 0)
                if in_tail:
                    tail_id = _recover_tail_track_id(
                        box=xyxy,
                        cls_id=cls_id,
                        frame_w=int(frame.shape[1]),
                        frame_h=int(frame.shape[0]),
                        canonical_last_seen=canonical_last_seen,
                        frame_index=frame_index,
                        used_canonical_in_frame=used_canonical_in_frame,
                    )
                    if tail_id is not None:
                        match_id = tail_id
                        diag["tail_recoveries"] = int(diag["tail_recoveries"]) + 1

            if match_id is None:
                in_tail = total_frames > 0 and frame_index >= max(total_frames - TAIL_GUARD_FRAMES, 0)
                if in_tail and conf < TAIL_GUARD_MIN_NEW_TRACK_CONF:
                    diag["tail_new_id_suppressed"] = int(diag["tail_new_id_suppressed"]) + 1
                    continue

            if match_id is None:
                match_id = next_id
                next_id += 1
                diag["new_ids_created"] = int(diag["new_ids_created"]) + 1

            display_id = _compact_display_id(
                canonical_id=match_id,
                frame_index=frame_index,
                canonical_last_seen=canonical_last_seen,
                canonical_to_display=canonical_to_display,
                display_last_seen=display_last_seen,
                used_display_in_frame=used_display_in_frame,
            )
            frame_boxes.append(xyxy)
            frame_ids.append(match_id)
            frame_cls_ids.append(cls_id)
            used_canonical_in_frame.add(match_id)
            age_votes = canonical_age_votes.setdefault(match_id, {})
            age_label = _format_age_label(names.get(cls_id), cls_id)
            if age_label != "Unknown":
                age_votes[age_label] = int(age_votes.get(age_label, 0)) + 1
            stable_age_label = age_label
            if age_votes:
                stable_age_label = max(
                    age_votes.items(),
                    key=lambda item: (item[1], item[0] == age_label, item[0] != "Unknown"),
                )[0]
            canonical_last_seen[match_id] = {
                "box": xyxy.tolist(),
                "frame_index": frame_index,
                "cls_id": cls_id,
                "age_label": stable_age_label,
                "edge_anchor": _edge_anchor(xyxy, int(frame.shape[1]), int(frame.shape[0])),
            }

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
            tracking_total_s += time.perf_counter() - t_track_logic

            if run_write_annotated:
                t_annot = time.perf_counter()
                _draw_detection_card(
                    frame,
                    box=xyxy,
                    label_text=f"ID {display_id:02d} | {stable_age_label}",
                    color=_color_for_id(display_id),
                    occupied_tag_rects=occupied_tag_rects,
                    avoid_rects=frame_detection_rects,
                )
                frame_annotation_total_s += time.perf_counter() - t_annot

        if (frame_index % run_save_every_nth) == 0:
            frame_path = output_dir / f"frame_{frame_index:06d}.jpg"
            t_write = time.perf_counter()
            _save_frame(frame_path, frame, jpeg_quality=run_jpeg_quality)
            frame_write_total_s += time.perf_counter() - t_write
            frames_saved += 1

        prev_boxes = frame_boxes
        prev_ids = frame_ids
        prev_cls_ids = frame_cls_ids
        skipped = 0
        for _ in range(max(stride - 1, 0)):
            if should_cancel and should_cancel():
                end_of_stream = True
                raise RuntimeError("Processing cancelled")
            t_grab = time.perf_counter()
            if not cap.grab():
                grab_total_s += time.perf_counter() - t_grab
                end_of_stream = True
                break
            grab_total_s += time.perf_counter() - t_grab
            skipped += 1
        if skipped:
            diag["frames_read"] = int(diag["frames_read"]) + skipped
        frame_index += 1 + skipped

    cap.release()
    t_checkpoint = time.perf_counter()
    checkpoint_counts: dict[int, int] = {}
    for det in detections:
        idx = int(det.get("frame_index", 0))
        checkpoint_counts[idx] = int(checkpoint_counts.get(idx, 0)) + 1
    sampled_checkpoint_extraction_s = time.perf_counter() - t_checkpoint
    LAST_TRACKING_DIAGNOSTICS.clear()
    LAST_TRACKING_DIAGNOSTICS.update(
        {
            "tracker_mode_start": TRACKER_ALGO if run_use_tracker else "iou",
            "tracker_mode_end": tracker_mode,
            "fallback_used": fallback_used,
            "pipeline_mode": str(runtime_cfg["mode"]),
            "inference_device": resolved_device,
            "fp16": use_fp16,
            "video_imgsz": run_video_imgsz,
            "video_conf": run_video_conf,
            "video_iou": run_video_iou,
            "use_tracker": run_use_tracker,
            "target_sample_fps": run_target_sample_fps,
            "write_annotated_frames": run_write_annotated,
            "save_every_nth_frame": run_save_every_nth,
            "frame_jpeg_quality": run_jpeg_quality,
            "frame_stride": stride,
            "raw_bytetrack_ids_seen": len(raw_bytetrack_ids_seen),
            "canonical_ids_seen": len(canonical_last_seen),
            "max_display_id_seen": max(display_last_seen.keys()) if display_last_seen else 0,
            "detection_total_s": round(detection_total_s, 4),
            "tracking_total_s": round(tracking_total_s, 4),
            "frame_write_total_s": round(frame_write_total_s, 4),
            "frame_annotation_total_s": round(frame_annotation_total_s, 4),
            "read_total_s": round(read_total_s, 4),
            "grab_total_s": round(grab_total_s, 4),
            "yolo_preprocess_ms_total": round(yolo_preprocess_ms_total, 4),
            "yolo_inference_ms_total": round(yolo_inference_ms_total, 4),
            "yolo_postprocess_ms_total": round(yolo_postprocess_ms_total, 4),
            "frames_saved": int(frames_saved),
            "sampled_checkpoint_extraction_s": round(sampled_checkpoint_extraction_s, 4),
            **{k: int(v) for k, v in diag.items()},
        }
    )
    return detections
