from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import uuid
import shutil
import json

from .pipeline.run import run_job

app = FastAPI(title="Elephant Backend", version="0.1")

ROOT = Path(__file__).resolve().parents[1]
STORAGE = ROOT / "app" / "storage"
STORAGE.mkdir(parents=True, exist_ok=True)

ALLOWED = {".mp4", ".mov", ".avi", ".m4v"}
JOB_STATE: dict[str, dict] = {}  # in-memory status (MVP)


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: int = 0
    message: str = ""


@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    lat: float | None = Form(None),
    lon: float | None = Form(None),
    fps: int = Form(25),
    width: int = Form(1280),
):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    job_id = uuid.uuid4().hex[:12]
    job_dir = STORAGE / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    raw_path = job_dir / f"upload{ext}"
    with raw_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    JOB_STATE[job_id] = {"status": "processing", "progress": 5, "message": "uploaded"}
    try:
        out = run_job(
            raw_video=str(raw_path),
            job_dir=str(job_dir),
            fps=fps,
            width=width,
            gps=(lat, lon) if lat is not None and lon is not None else None,
            progress_cb=lambda p, m: JOB_STATE[job_id].update(progress=p, message=m),
        )
        (job_dir / "result.json").write_text(json.dumps(out, indent=2), "utf-8")
        JOB_STATE[job_id].update(status="done", progress=100, message="ok")
    except Exception as e:
        JOB_STATE[job_id].update(status="error", message=str(e))
        raise

    return {"job_id": job_id, "status": JOB_STATE[job_id]["status"]}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    st = JOB_STATE.get(job_id)
    if not st:
        raise HTTPException(404, "job not found")
    return JobStatus(job_id=job_id, **st)


@app.get("/api/jobs/{job_id}/results")
def job_results(job_id: str):
    job_dir = STORAGE / job_id
    meta_file = job_dir / "result.json"
    if not meta_file.exists():
        raise HTTPException(404, "results not ready")
    meta = json.loads(meta_file.read_text("utf-8"))
    base = f"/api/jobs/{job_id}/artifact"
    return {
        "job_id": job_id,
        "video_name": meta["video_name"],
        "duration_s": meta["duration_s"],
        "summary": meta["summary"],
        "detections_url": f"{base}/dets.csv",
        "tracks_url": f"{base}/tracks.csv",
        "annotated_video_url": f"{base}/annotated.mp4",
        # Optional fallback for debugging; expose normalized video too
        "normalized_video_url": f"{base}/normalized.mp4",
    }


@app.get("/api/jobs/{job_id}/artifact/{name}")
def artifact(job_id: str, name: str):
    path = STORAGE / job_id / name
    if not path.exists():
        raise HTTPException(404, "not found")

    # Set media type so browsers play/preview correctly
    media_type = None
    lname = name.lower()
    if lname.endswith(".mp4"):
        media_type = "video/mp4"
    elif lname.endswith(".csv"):
        media_type = "text/csv"
    elif lname.endswith(".json"):
        media_type = "application/json"

    return FileResponse(path, media_type=media_type)
