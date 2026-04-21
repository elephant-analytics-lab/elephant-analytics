from __future__ import annotations

import os
from pathlib import Path

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parents[3]
HABITAT_DEVICE = os.getenv("EA_HABITAT_DEVICE", "auto").strip().lower() or "auto"
HABITAT_FRAME_STRIDE = max(1, int(os.getenv("EA_HABITAT_FRAME_STRIDE", "1")))
_HABITAT_MODEL_CACHE: dict[str, YOLO] = {}
HABITAT_MODE_PRESETS: dict[str, dict[str, object]] = {
    "fast": {
        "frame_stride": 3,
        "device": "auto",
    },
    "quality": {
        "frame_stride": 1,
        "device": "auto",
    },
}


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _get_model(model_path: Path) -> YOLO:
    key = str(model_path.resolve())
    cached = _HABITAT_MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    model = YOLO(key)
    _HABITAT_MODEL_CACHE[key] = model
    return model


def _resolve_runtime_config(pipeline_mode: str | None) -> dict[str, object]:
    cfg: dict[str, object] = {
        "frame_stride": HABITAT_FRAME_STRIDE,
        "device": HABITAT_DEVICE,
    }
    mode = str(pipeline_mode or "").strip().lower()
    if mode and mode in HABITAT_MODE_PRESETS:
        cfg.update(HABITAT_MODE_PRESETS[mode])
    cfg["frame_stride"] = max(1, int(cfg["frame_stride"]))
    cfg["device"] = str(cfg["device"] or "auto").strip().lower() or "auto"
    cfg["mode"] = mode or "default"
    return cfg


def _default_habitat_weight_candidates() -> list[Path]:
    return [
        BASE_DIR / "best.pt",
        BASE_DIR / "runs" / "classify" / "habitat_cls_v1" / "weights" / "best.pt",
        BASE_DIR / "models" / "habitat_cls_v1_best.pt",
        BASE_DIR / "models" / "habitat_best.pt",
    ]


def resolve_habitat_model_path() -> Path | None:
    env_path = os.getenv("HABITAT_MODEL_PATH", "").strip()
    if env_path:
        p = Path(env_path)
        if p.exists() and p.is_file():
            return p
    for candidate in _default_habitat_weight_candidates():
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def available_habitat_model() -> str | None:
    p = resolve_habitat_model_path()
    return str(p) if p else None


def _frame_index_from_name(path: Path) -> int:
    stem = path.stem
    if "_" not in stem:
        return 0
    idx = stem.rsplit("_", 1)[-1]
    return int(idx) if idx.isdigit() else 0


def classify_habitat_frames(frames_dir: Path, pipeline_mode: str | None = None) -> list[dict]:
    model_path = resolve_habitat_model_path()
    if model_path is None:
        return []
    if not frames_dir.exists() or not frames_dir.is_dir():
        return []

    runtime_cfg = _resolve_runtime_config(pipeline_mode)
    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    if int(runtime_cfg["frame_stride"]) > 1:
        frame_paths = frame_paths[:: int(runtime_cfg["frame_stride"])]
    if not frame_paths:
        return []

    resolved_device = _resolve_device(str(runtime_cfg["device"]))
    model = _get_model(model_path)
    results = model.predict(
        source=[str(p) for p in frame_paths],
        verbose=False,
        device=resolved_device,
        half=resolved_device.startswith("cuda"),
    )

    rows: list[dict] = []
    for frame_path, res in zip(frame_paths, results):
        probs = getattr(res, "probs", None)
        names = getattr(res, "names", None) or {}
        if probs is None:
            continue
        cls_id = int(probs.top1)
        conf = float(probs.top1conf)
        label = str(names.get(cls_id, f"class_{cls_id}"))
        rows.append(
            {
                "frame_index": _frame_index_from_name(frame_path),
                "habitat_label": label,
                "conf": conf,
            }
        )
    return rows
