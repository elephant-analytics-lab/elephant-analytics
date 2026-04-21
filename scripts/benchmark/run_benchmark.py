from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.pipeline.detector import detect_and_annotate, get_last_tracking_diagnostics  # noqa: E402
from backend.app.pipeline.habitat import classify_habitat_frames  # noqa: E402


def video_duration_seconds(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0
    frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    if frames <= 0 or fps <= 0:
        return 0.0
    return frames / fps


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Elephant Analytics video processing runtime.")
    parser.add_argument("--video", required=True, help="Path to input video (.mp4/.mov/.avi).")
    parser.add_argument(
        "--mode",
        default="fast_trend",
        choices=["fast_trend", "value_1x", "balanced_2fps", "fast", "quality", "default"],
        help="Pipeline mode.",
    )
    parser.add_argument("--model-key", default=None, help="Optional detector model key.")
    parser.add_argument("--runs", type=int, default=1, help="Number of repeated runs.")
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep generated frames directories (default deletes them).",
    )
    args = parser.parse_args()

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Missing video: {video_path}")

    outputs_root = REPO_ROOT / "data" / "outputs" / "benchmarks"
    outputs_root.mkdir(parents=True, exist_ok=True)

    duration = video_duration_seconds(video_path)
    runs_payload: list[dict] = []
    mode = None if args.mode == "default" else args.mode

    for i in range(args.runs):
        run_id = f"{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{i+1}"
        frames_dir = outputs_root / f"{video_path.stem}_{args.mode}_{run_id}"
        frames_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        detections = detect_and_annotate(
            video_path=video_path,
            output_dir=frames_dir,
            model_key=args.model_key,
            pipeline_mode=mode,
        )
        habitat_rows = classify_habitat_frames(frames_dir, pipeline_mode=mode)
        elapsed = time.perf_counter() - t0
        ratio = (elapsed / duration) if duration > 0 else None
        tracking_diag = get_last_tracking_diagnostics()

        run_payload = {
            "run_index": i + 1,
            "video_path": str(video_path),
            "video_duration_seconds": duration,
            "mode": args.mode,
            "model_key": args.model_key,
            "processing_seconds": elapsed,
            "processing_ratio": ratio,
            "detections": len(detections),
            "habitat_frames": len(habitat_rows),
            "tracking_diagnostics": tracking_diag,
        }
        runs_payload.append(run_payload)

        if not args.keep_frames:
            shutil.rmtree(frames_dir, ignore_errors=True)

    out = {
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "runs": runs_payload,
        "mean_processing_seconds": sum(r["processing_seconds"] for r in runs_payload) / max(len(runs_payload), 1),
        "mean_processing_ratio": (
            sum((r["processing_ratio"] or 0.0) for r in runs_payload) / max(len(runs_payload), 1)
            if duration > 0
            else None
        ),
    }
    report_path = outputs_root / f"benchmark_{video_path.stem}_{args.mode}_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.json"
    report_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"report_path": str(report_path), "mean_processing_ratio": out["mean_processing_ratio"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
