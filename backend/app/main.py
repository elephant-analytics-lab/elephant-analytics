from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .pipeline.detector import detect_and_annotate, detect_image
from .pipeline.processor import extract_gps_points, extract_image_gps_points, write_gpx
from .storage.db import (
    create_job,
    get_detection_count,
    get_detection_series,
    get_detections,
    get_gps_points,
    get_job,
    get_jobs,
    get_class_distribution,
    get_timeline,
    get_unique_track_count,
    init_db,
    insert_detections,
    insert_gps_points,
    update_job,
    clear_history,
)

BASE_DIR = Path(__file__).resolve().parents[2]
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"

app = FastAPI(title="Elephant Analytics API", version="0.1.0")
logger = logging.getLogger("elephant-analytics")


def _get_video_fps(path: Path) -> float | None:
    try:
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
    except Exception:
        return None
    if not fps or fps <= 0:
        return None
    return float(fps)


@app.on_event("startup")
def _startup() -> None:
    init_db()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    job_id = uuid.uuid4().hex
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".mp4", ".mov", ".avi", ".jpg", ".jpeg", ".png"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    video_path = UPLOAD_DIR / f"{job_id}{suffix}"
    contents = await file.read()
    video_path.write_bytes(contents)
    create_job(job_id, file.filename, str(video_path))
    return {"job_id": job_id, "filename": file.filename}


@app.post("/process/{job_id}")
def process_video(job_id: str) -> dict:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    update_job(job_id, "processing")
    try:
        video_path = Path(job["video_path"])
        if not video_path.exists():
            raise RuntimeError("Video file not found")

        suffix = video_path.suffix.lower()
        points = []
        gpx_path = None
        if suffix in {".mp4", ".mov", ".avi"}:
            points = extract_gps_points(video_path)
            insert_gps_points(job_id, points)

            if points:
                gpx_path = OUTPUT_DIR / f"{job_id}.gpx"
                write_gpx(points, gpx_path, name=job["filename"])

            frames_dir = OUTPUT_DIR / "frames" / job_id
            detections = detect_and_annotate(video_path, frames_dir)
        else:
            frames_dir = OUTPUT_DIR / "frames" / job_id
            detections = detect_image(video_path, frames_dir)
            points = extract_image_gps_points(video_path)
            insert_gps_points(job_id, points)
            if points:
                gpx_path = OUTPUT_DIR / f"{job_id}.gpx"
                write_gpx(points, gpx_path, name=job["filename"])

        insert_detections(job_id, detections)
        detections_path = OUTPUT_DIR / f"{job_id}_detections.json"
        detections_path.write_text(
            json.dumps({"job_id": job_id, "detections": detections}, indent=2),
            encoding="utf-8",
        )

        update_job(job_id, "complete", gpx_path=str(gpx_path) if gpx_path else None)
    except Exception as exc:
        logger.exception("Processing failed for job_id=%s", job_id)
        update_job(job_id, "failed", message=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "job_id": job_id,
        "status": "complete",
        "gps_points": len(points),
        "detections": len(detections),
    }


@app.get("/jobs")
def list_jobs() -> list[dict]:
    return get_jobs()


@app.get("/results/{job_id}")
def job_results(job_id: str) -> JSONResponse:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    points = get_gps_points(job_id)
    detections_count = get_detection_count(job_id)
    return JSONResponse({"job": job, "gps": points, "detections_count": detections_count})


@app.get("/timeline/{job_id}")
def job_timeline(job_id: str) -> JSONResponse:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    rows = get_timeline(job_id)
    return JSONResponse({"job_id": job_id, "timeline": rows})


@app.get("/metrics/{job_id}")
def job_metrics(job_id: str) -> JSONResponse:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    detections_count = get_detection_count(job_id)
    unique_tracks = get_unique_track_count(job_id)
    if unique_tracks == 0:
        unique_tracks = detections_count

    class_rows = get_class_distribution(job_id)
    total = sum(row["cnt"] for row in class_rows) or 0
    age_distribution = {}
    for row in class_rows:
        label = row["cls_label"]
        age_distribution[label] = round((row["cnt"] / total) * 100, 2) if total else 0

    herd_series = get_detection_series(job_id)
    avg_herd_size = 0.0
    if herd_series:
        avg_herd_size = sum(row["herd_size"] for row in herd_series) / len(herd_series)

    fps = None
    video_path = Path(job["video_path"])
    if video_path.suffix.lower() in {".mp4", ".mov", ".avi"}:
        fps = _get_video_fps(video_path)
        if fps:
            herd_series = [
                {**row, "time_seconds": row["frame_index"] / fps} for row in herd_series
            ]

    return JSONResponse(
        {
            "job_id": job_id,
            "detections_count": detections_count,
            "unique_tracks": unique_tracks,
            "age_distribution": age_distribution,
            "avg_herd_size": avg_herd_size,
            "herd_series": herd_series,
            "fps": fps,
        }
    )


@app.get("/detections/{job_id}")
def job_detections(job_id: str) -> JSONResponse:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    rows = get_detections(job_id)
    return JSONResponse({"job_id": job_id, "detections": rows})


@app.post("/admin/clear-history")
def admin_clear_history() -> dict:
    clear_history()
    return {"ok": True}
