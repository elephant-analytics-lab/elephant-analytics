from pathlib import Path
import subprocess
import json
import cv2
import pandas as pd
from ultralytics import YOLO

# Correct COCO class id for 'elephant'
ELEPHANT = 20


def _ffprobe(path: str):
    """Return basic video stream info (width/height/fps/duration)."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration",
        "-of", "json", path
    ]
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)["streams"][0]


def _normalize(in_path: str, out_path: str, fps: int = 25, width: int = 1280):
    """Transcode to H.264 MP4 with fixed fps/width; -2 keeps aspect ratio."""
    cmd = [
        "ffmpeg", "-y", "-i", in_path,
        "-vf", f"scale={width}:-2,fps={fps}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-movflags", "+faststart",
        out_path
    ]
    subprocess.run(cmd, check=True)


def _encode_h264(in_path: str, out_path: str, fps: float):
    """Re-encode an MP4 to H.264 for browser playback."""
    cmd = [
        "ffmpeg", "-y", "-i", in_path,
        "-r", f"{fps}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-movflags", "+faststart",
        out_path
    ]
    subprocess.run(cmd, check=True)


def run_job(
    raw_video: str,
    job_dir: str,
    fps: int = 25,
    width: int = 1280,
    gps=None,
    progress_cb=lambda p, m: None,
):
    """
    Pipeline:
      ffprobe -> normalize -> YOLO detect -> simple IoU tracking -> draw ->
      OpenCV writes annotated_raw.mp4 -> FFmpeg re-encodes to annotated.mp4 (H.264)
      -> write CSVs + summary
    """
    job = Path(job_dir)
    job.mkdir(parents=True, exist_ok=True)

    norm = job / "normalized.mp4"
    annotated_raw = job / "annotated_raw.mp4"
    annotated_final = job / "annotated.mp4"

    # Probe input
    progress_cb(10, "ffprobe")
    info = _ffprobe(raw_video)

    # Normalize input
    progress_cb(20, "normalize")
    _normalize(raw_video, str(norm), fps=fps, width=width)

    # Load YOLO
    progress_cb(35, "load model")
    model = YOLO("yolov8n.pt")  # auto-download on first run

    # Open normalized video
    cap = cv2.VideoCapture(str(norm))
    if not cap.isOpened():
        raise RuntimeError("Cannot open normalized video with OpenCV.")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = float(cap.get(cv2.CAP_PROP_FPS)) or float(fps)

    # OpenCV writer (raw; we will re-encode with FFmpeg later)
    # mp4v is widely available; browser-compat will be handled by FFmpeg
    writer = cv2.VideoWriter(
        str(annotated_raw),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (W, H),
    )
    if not writer.isOpened():
        raise RuntimeError("Failed to open VideoWriter for annotated_raw.mp4")

    # Accumulators
    det_rows, track_rows = [], []
    tracks, lost, next_id, frame_idx = {}, {}, 1, 0

    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        iw = max(0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0, min(ay2, by2) - max(ay1, by1))
        inter = iw * ih
        aa = (ax2 - ax1) * (ay2 - ay1)
        bb = (bx2 - bx1) * (by2 - by1)
        denom = max(1e-9, aa + bb - inter)
        return inter / denom

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # YOLO inference (slightly lower conf so we see boxes)
        res = model.predict(frame, conf=0.25, classes=[ELEPHANT], verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []
        confs = res.boxes.conf.cpu().numpy() if res.boxes is not None else []

        for b, s in zip(boxes, confs):
            det_rows.append([frame_idx, *b.tolist(), float(s)])

        # simple IoU tracking
        assigned = set()
        new_tracks = {}
        for tid, tb in tracks.items():
            best, idx = 0.0, -1
            for j, b in enumerate(boxes):
                if j in assigned:
                    continue
                v = iou(tb, b)
                if v > best:
                    best, idx = v, j
            if best >= 0.3:
                new_tracks[tid] = boxes[idx]
                assigned.add(idx)
                lost[tid] = 0
            else:
                lost[tid] = lost.get(tid, 0) + 1
                if lost[tid] <= 5:
                    new_tracks[tid] = tb

        for j, b in enumerate(boxes):
            if j in assigned:
                continue
            new_tracks[next_id] = b
            lost[next_id] = 0
            next_id += 1

        tracks = new_tracks

        # draw & record
        for tid, b in tracks.items():
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, f"ID {tid}", (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
            track_rows.append([frame_idx, tid, x1, y1, x2, y2, 1.0])

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 30 == 0:
            progress_cb(min(95, 35 + frame_idx // 3), "inference")

    cap.release()
    writer.release()

    # Re-encode to H.264 for browser playback (final file served to UI)
    progress_cb(97, "encode h264")
    _encode_h264(str(annotated_raw), str(annotated_final), FPS)

    # Write CSVs (even if empty)
    pd.DataFrame(det_rows, columns=["frame", "x1", "y1", "x2", "y2", "score"]).to_csv(
        job / "dets.csv", index=False
    )
    pd.DataFrame(
        track_rows, columns=["frame", "track_id", "x1", "y1", "x2", "y2", "score"]
    ).to_csv(job / "tracks.csv", index=False)

    # Summary
    df_tracks = pd.DataFrame(
        track_rows, columns=["frame", "track_id", "x1", "y1", "x2", "y2", "score"]
    )
    summary = {
        "frames": int(frame_idx),
        "unique_elephants": int(df_tracks.track_id.nunique()) if not df_tracks.empty else 0,
        "avg_group_size": float(
            df_tracks.groupby("frame").track_id.nunique().mean()
        ) if not df_tracks.empty else 0.0,
        "fps": float(FPS),
        "width": W,
        "height": H,
    }

    return {
        "video_name": Path(raw_video).name,
        "duration_s": float(info.get("duration", 0)),
        "summary": summary,
    }
