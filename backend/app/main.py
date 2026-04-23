from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
from pathlib import Path
from statistics import median

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .pipeline.detector import (
    available_models,
    detect_and_annotate,
    detect_image,
    get_last_tracking_diagnostics,
)
from .pipeline.habitat import (
    available_habitat_model,
    classify_habitat_frames,
    get_last_habitat_diagnostics,
)
from .pipeline.processor import extract_gps_points, extract_image_gps_points, write_gpx
from .storage.db import (
    create_job,
    get_detection_count,
    get_detection_series,
    get_detections,
    get_gps_points,
    get_job,
    get_jobs,
    get_jobs_minimal,
    get_class_distribution,
    get_habitat_distribution,
    get_habitat_timeline,
    get_timeline,
    get_unique_track_count,
    get_stable_track_count,
    init_db,
    insert_detections,
    insert_habitat_rows,
    insert_gps_points,
    update_job,
    clear_history,
    delete_jobs,
)

BASE_DIR = Path(__file__).resolve().parents[2]
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"
UPLOAD_CHUNK_SIZE = 1024 * 1024 * 8  # 8 MB
MAX_UPLOAD_BYTES = 1024 * 1024 * 1024 * 4  # 4 GB

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


def _clear_directory_contents(path: Path) -> dict[str, int]:
    removed_files = 0
    removed_dirs = 0
    if not path.exists():
        return {"files_removed": 0, "dirs_removed": 0}
    for child in path.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
                removed_dirs += 1
            else:
                child.unlink(missing_ok=True)
                removed_files += 1
        except Exception:
            logger.exception("Failed to remove path during clear_history: %s", child)
    return {"files_removed": removed_files, "dirs_removed": removed_dirs}


def _build_sampled_visible_count_series(herd_series: list[dict]) -> list[dict]:
    out: list[dict] = []
    for row in herd_series:
        frame_index = int(row.get("frame_index", 0))
        sampled_visible_count = int(row.get("herd_size", 0) or 0)
        item = {
            "frame_index": frame_index,
            "sampled_visible_count": sampled_visible_count,
        }
        if row.get("time_seconds") is not None:
            item["time_seconds"] = float(row["time_seconds"])
        out.append(item)
    return out


def _build_confidence_profile(detections: list[dict]) -> dict:
    confs = []
    for det in detections:
        try:
            confs.append(float(det.get("conf", 0.0)))
        except Exception:
            continue
    if not confs:
        return {
            "sample_size": 0,
            "avg_confidence": 0.0,
            "median_confidence": 0.0,
            "high_confidence_share": 0.0,
        }
    high_conf = sum(1 for value in confs if value >= 0.75)
    return {
        "sample_size": len(confs),
        "avg_confidence": round(sum(confs) / len(confs), 4),
        "median_confidence": round(float(median(confs)), 4),
        "high_confidence_share": round(high_conf / len(confs), 4),
    }


def _build_observed_age_composition(class_rows: list[dict]) -> list[dict]:
    total = sum(int(row["cnt"]) for row in class_rows) or 0
    composition = []
    for row in class_rows:
        count = int(row["cnt"])
        composition.append(
            {
                "age_class": str(row["cls_label"]),
                "count": count,
                "share": round((count / total), 4) if total else 0.0,
            }
        )
    return composition


def _remove_job_artifacts(job_ids: list[str], jobs: list[dict]) -> dict[str, int]:
    removed_files = 0
    removed_dirs = 0
    ids_set = {j for j in job_ids if j}
    for row in jobs:
        video_path = row.get("video_path")
        gpx_path = row.get("gpx_path")
        for p in (video_path, gpx_path):
            if not p:
                continue
            path = Path(p)
            if path.exists() and path.is_file():
                try:
                    path.unlink(missing_ok=True)
                    removed_files += 1
                except Exception:
                    logger.exception("Failed deleting artifact file: %s", path)
    frames_root = OUTPUT_DIR / "frames"
    for job_id in ids_set:
        frames_dir = frames_root / job_id
        if frames_dir.exists() and frames_dir.is_dir():
            try:
                shutil.rmtree(frames_dir, ignore_errors=True)
                removed_dirs += 1
            except Exception:
                logger.exception("Failed deleting frames dir: %s", frames_dir)
        det_json = OUTPUT_DIR / f"{job_id}_detections.json"
        if det_json.exists() and det_json.is_file():
            try:
                det_json.unlink(missing_ok=True)
                removed_files += 1
            except Exception:
                logger.exception("Failed deleting detections json: %s", det_json)
    return {"files_removed": removed_files, "dirs_removed": removed_dirs}


class DeleteJobsRequest(BaseModel):
    job_ids: list[str]


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
    total_bytes = 0
    try:
        with video_path.open("wb") as out:
            while True:
                chunk = await file.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="File too large. Maximum upload size is 4 GB.",
                    )
                out.write(chunk)
    except HTTPException:
        if video_path.exists():
            video_path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()
    create_job(job_id, file.filename, str(video_path))
    return {"job_id": job_id, "filename": file.filename}


@app.post("/process/{job_id}")
def process_video(
    job_id: str,
    model_key: str | None = None,
    pipeline_mode: str | None = None,
) -> dict:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    update_job(job_id, "processing")
    tracking_diagnostics: dict | None = None
    habitat_rows: list[dict] = []
    process_timing: dict[str, float] = {}
    habitat_diagnostics: dict | None = None
    try:
        process_start = time.perf_counter()
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
            t_detect = time.perf_counter()
            detections = detect_and_annotate(
                video_path,
                frames_dir,
                model_key=model_key,
                pipeline_mode=pipeline_mode,
            )
            process_timing["detection_stage_s"] = round(time.perf_counter() - t_detect, 4)
            tracking_diagnostics = get_last_tracking_diagnostics()
        else:
            frames_dir = OUTPUT_DIR / "frames" / job_id
            t_detect = time.perf_counter()
            detections = detect_image(video_path, frames_dir, model_key=model_key)
            process_timing["detection_stage_s"] = round(time.perf_counter() - t_detect, 4)
            points = extract_image_gps_points(video_path)
            insert_gps_points(job_id, points)
            if points:
                gpx_path = OUTPUT_DIR / f"{job_id}.gpx"
                write_gpx(points, gpx_path, name=job["filename"])

        insert_detections(job_id, detections)
        t_habitat = time.perf_counter()
        habitat_rows = classify_habitat_frames(frames_dir, pipeline_mode=pipeline_mode)
        habitat_diagnostics = get_last_habitat_diagnostics()
        process_timing["habitat_classification_s"] = round(time.perf_counter() - t_habitat, 4)
        insert_habitat_rows(job_id, habitat_rows)
        detections_path = OUTPUT_DIR / f"{job_id}_detections.json"
        detections_path.write_text(
            json.dumps({"job_id": job_id, "detections": detections}, indent=2),
            encoding="utf-8",
        )
        process_timing["process_total_s"] = round(time.perf_counter() - process_start, 4)
        if tracking_diagnostics:
            process_timing["tracking_total_s"] = round(
                float(tracking_diagnostics.get("tracking_total_s", 0.0)), 4
            )
            process_timing["detection_total_s"] = round(
                float(tracking_diagnostics.get("detection_total_s", 0.0)), 4
            )
            process_timing["preview_asset_generation_s"] = round(
                float(tracking_diagnostics.get("frame_write_total_s", 0.0)), 4
            )
            process_timing["sampled_checkpoint_extraction_s"] = round(
                float(tracking_diagnostics.get("sampled_checkpoint_extraction_s", 0.0)), 4
            )
            process_timing["read_s"] = round(
                float(tracking_diagnostics.get("read_total_s", 0.0)), 4
            )
            process_timing["grab_s"] = round(
                float(tracking_diagnostics.get("grab_total_s", 0.0)), 4
            )
            process_timing["annotation_s"] = round(
                float(tracking_diagnostics.get("frame_annotation_total_s", 0.0)), 4
            )
            process_timing["yolo_preprocess_ms_total"] = round(
                float(tracking_diagnostics.get("yolo_preprocess_ms_total", 0.0)), 4
            )
            process_timing["yolo_inference_ms_total"] = round(
                float(tracking_diagnostics.get("yolo_inference_ms_total", 0.0)), 4
            )
            process_timing["yolo_postprocess_ms_total"] = round(
                float(tracking_diagnostics.get("yolo_postprocess_ms_total", 0.0)), 4
            )
            process_timing["frames_saved"] = int(tracking_diagnostics.get("frames_saved", 0))
        if habitat_diagnostics:
            process_timing["habitat_predict_s"] = round(
                float(habitat_diagnostics.get("predict_s", 0.0)), 4
            )
            process_timing["habitat_frame_files_used"] = int(
                habitat_diagnostics.get("frame_files_used", 0)
            )
            process_timing["habitat_frame_files_total"] = int(
                habitat_diagnostics.get("frame_files_total", 0)
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
        "habitat_frames": len(habitat_rows),
        "model_key": model_key,
        "pipeline_mode": pipeline_mode or "default",
        "tracking_diagnostics": tracking_diagnostics,
        "habitat_diagnostics": habitat_diagnostics,
        "timings": process_timing,
    }


@app.get("/models")
def list_models() -> dict:
    return {"models": available_models(), "habitat_model": available_habitat_model()}


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
    raw_unique_tracks = get_unique_track_count(job_id)
    stable_unique_tracks = get_stable_track_count(job_id, min_samples=3, min_avg_conf=0.30)
    has_track_ids = (raw_unique_tracks > 0) or (stable_unique_tracks > 0)
    unique_tracks = stable_unique_tracks if stable_unique_tracks > 0 else raw_unique_tracks
    if not has_track_ids:
        unique_tracks = 0

    class_rows = get_class_distribution(job_id)
    total = sum(row["cnt"] for row in class_rows) or 0
    age_distribution = {}
    for row in class_rows:
        label = row["cls_label"]
        age_distribution[label] = round((row["cnt"] / total) * 100, 2) if total else 0

    herd_series = get_detection_series(job_id)
    avg_herd_size = 0.0
    peak_herd_size = 0
    if herd_series:
        avg_herd_size = sum(row["herd_size"] for row in herd_series) / len(herd_series)
        peak_herd_size = max(row["herd_size"] for row in herd_series)

    # Tracking IDs can fragment when elephants occlude/exit/re-enter.
    # Cap reported population by the observed concurrent peak to avoid
    # over-counting identity fragments as new elephants.
    if has_track_ids and peak_herd_size > 0 and unique_tracks > peak_herd_size:
        unique_tracks = peak_herd_size

    fps = None
    video_path = Path(job["video_path"])
    if video_path.suffix.lower() in {".mp4", ".mov", ".avi"}:
        fps = _get_video_fps(video_path)
    if fps:
        herd_series = [
            {**row, "time_seconds": row["frame_index"] / fps} for row in herd_series
        ]

    habitat_dist_rows = get_habitat_distribution(job_id)
    habitat_timeline = get_habitat_timeline(job_id)
    habitat_total = sum(int(row["cnt"]) for row in habitat_dist_rows) or 0
    habitat_distribution = [
        {
            "habitat": row["habitat_label"],
            "percentage": round((int(row["cnt"]) / habitat_total) * 100.0, 2) if habitat_total else 0.0,
            "count": int(row["cnt"]),
            "avg_confidence": round(float(row["avg_conf"]), 4) if row["avg_conf"] is not None else 0.0,
        }
        for row in habitat_dist_rows
    ]
    dominant_habitat = habitat_distribution[0]["habitat"] if habitat_distribution else "-"
    dominant_habitat_share = habitat_distribution[0]["percentage"] if habitat_distribution else 0.0
    habitat_switches = max(len(habitat_timeline) - 1, 0)
    habitat_coverage = 100.0 if habitat_total > 0 else 0.0
    t_bundle = time.perf_counter()
    detections = get_detections(job_id)
    confidence_profile = _build_confidence_profile(detections)
    observed_age_composition = _build_observed_age_composition(class_rows)
    sampled_visible_count_series = _build_sampled_visible_count_series(herd_series)
    sampled_counts = [row["sampled_visible_count"] for row in sampled_visible_count_series]
    current_sampled_visible_count = sampled_counts[-1] if sampled_counts else 0
    min_sampled_visible_count = min(sampled_counts) if sampled_counts else 0
    peak_sampled_visible_count = max(sampled_counts) if sampled_counts else 0
    avg_sampled_visible_count = (
        round(sum(sampled_counts) / len(sampled_counts), 2) if sampled_counts else 0.0
    )
    caveats = [
        "Sampled checkpoint-based value. Not frame-perfect live inference.",
        "Observed age composition is derived from detections and should be interpreted as sampled observations.",
        "Without strict identity continuity guarantees, avoid over-interpreting individual-level persistence across the entire video.",
    ]
    fast_review_bundle = {
        "schema_version": "2026-04-21",
        "naming": {"primary_count_metric": "sampled_visible_count"},
        "review": {
            "sampled_visible_count": current_sampled_visible_count,
            "run_summary": {
                "detections_count": detections_count,
                "avg_herd_size": round(avg_herd_size, 2),
                "peak_herd_size": peak_herd_size,
            },
            "run_details": {
                "fps": fps,
                "dominant_habitat": dominant_habitat,
                "dominant_habitat_share": dominant_habitat_share,
            },
        },
        "trend": {
            "sampled_visible_count_over_time": sampled_visible_count_series,
            "min_sampled_visible_count": min_sampled_visible_count,
            "avg_sampled_visible_count": avg_sampled_visible_count,
            "peak_sampled_visible_count": peak_sampled_visible_count,
        },
        "frame_inspection": {
            "source": "sampled_frames",
            "detections_available": bool(detections),
        },
        "run_diagnostics": {
            "confidence_profile": confidence_profile,
            "observed_age_composition": observed_age_composition,
            "caveats": caveats,
        },
        "context": {
            "habitat_summary": {
                "dominant_habitat": dominant_habitat,
                "dominant_habitat_share": dominant_habitat_share,
                "habitat_switches": habitat_switches,
                "habitat_coverage": habitat_coverage,
            },
            "habitat_timeline": habitat_timeline,
        },
    }
    tracking_bundle = {
        "schema_version": "2026-04-21",
        "availability": {
            "has_tracking_rows": bool(has_track_ids),
            "has_track_ids": bool(has_track_ids),
        },
        "identity_metrics": {
            "unique_tracks": unique_tracks,
            "raw_unique_tracks": raw_unique_tracks,
            "stable_unique_tracks": stable_unique_tracks,
        },
        "timeline_hint": {
            "endpoint": f"/timeline/{job_id}",
            "note": "Tracking-aware timeline is optional deep analysis and should not block fast review.",
        },
    }
    results_bundle = {
        "schema_version": "2026-04-21",
        "naming": {"primary_count_metric": "sampled_visible_count"},
        "fast_review_bundle": fast_review_bundle,
        "tracking_bundle": tracking_bundle,
    }
    bundle_generation_s = round(time.perf_counter() - t_bundle, 4)

    return JSONResponse(
        {
            "job_id": job_id,
            "detections_count": detections_count,
            "unique_tracks": unique_tracks,
            "raw_unique_tracks": raw_unique_tracks,
            "stable_unique_tracks": stable_unique_tracks,
            "age_distribution": age_distribution,
            "avg_herd_size": avg_herd_size,
            "peak_herd_size": peak_herd_size,
            "herd_series": herd_series,
            "fps": fps,
            "habitat_distribution": habitat_distribution,
            "habitat_timeline": habitat_timeline,
            "dominant_habitat": dominant_habitat,
            "dominant_habitat_share": dominant_habitat_share,
            "habitat_switches": habitat_switches,
            "habitat_coverage": habitat_coverage,
            "fast_review_bundle": fast_review_bundle,
            "tracking_bundle": tracking_bundle,
            "results_bundle": results_bundle,
            "timings": {"bundle_generation_s": bundle_generation_s},
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
    upload_stats = _clear_directory_contents(UPLOAD_DIR)
    output_stats = _clear_directory_contents(OUTPUT_DIR)
    return {
        "ok": True,
        "uploads_files_removed": upload_stats["files_removed"],
        "uploads_dirs_removed": upload_stats["dirs_removed"],
        "outputs_files_removed": output_stats["files_removed"],
        "outputs_dirs_removed": output_stats["dirs_removed"],
    }


@app.post("/admin/delete-jobs")
def admin_delete_jobs(payload: DeleteJobsRequest) -> dict:
    job_ids = [j.strip() for j in payload.job_ids if j and j.strip()]
    if not job_ids:
        raise HTTPException(status_code=400, detail="No job_ids provided")
    jobs = get_jobs_minimal(job_ids)
    deleted_jobs = delete_jobs(job_ids)
    artifact_stats = _remove_job_artifacts(job_ids, jobs)
    return {
        "ok": True,
        "requested_jobs": len(job_ids),
        "deleted_jobs": deleted_jobs,
        "files_removed": artifact_stats["files_removed"],
        "dirs_removed": artifact_stats["dirs_removed"],
    }
