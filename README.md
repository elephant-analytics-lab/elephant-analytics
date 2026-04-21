# Elephant Analytics

Local MVP for elephant detection, tracking, and habitat analytics from drone media.

## Overview

Elephant Analytics provides:

- Media upload (video or image)
- Elephant detection and frame annotation (YOLO-based)
- Tracking and timeline summaries
- Demographic breakdown (Adult/Juvenile/Baby)
- Habitat classification over frames
- GPS extraction and map visualization
- Streamlit dashboard for analysis workflow

## Tech Stack

- Frontend: Streamlit
- Backend API: FastAPI + Uvicorn
- CV/ML: Ultralytics YOLO, OpenCV
- Data/analytics: NumPy, Pandas, Plotly, Altair
- Mapping: Folium + optional local TileServer
- Storage: SQLite + local filesystem

## Project Structure

```text
elephant-analytics/
|- backend/
|  |- app/
|  |  |- main.py                # FastAPI entrypoint
|  |  |- pipeline/              # detector, processor, habitat modules
|  |  `- storage/               # SQLite logic
|- streamlit/
|  `- app.py                    # Streamlit UI entrypoint
|- data/
|  |- uploads/                  # uploaded media
|  `- outputs/                  # frames, detections, generated artifacts
|- models/
|  `- base_models/              # local YOLO base weights
|- runs/                        # training outputs and experiments
|- docker-compose.yml           # optional tile server
`- requirements.txt             # Python dependencies
```

## Quick Start (Local)

### 1. Create virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Start backend API

```powershell
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

### 4. Start Streamlit app (new terminal)

```powershell
streamlit run streamlit/app.py
```

Streamlit expects backend at:

- `http://127.0.0.1:8000`

## Optional: Local Tile Server

If you have MBTiles in `tiles/`, start TileServer:

```powershell
docker compose up tileserver
```

Default port:

- `http://127.0.0.1:8080`

## Run On Other Computers (Docker)

Use the bundled runner compose file to start backend + frontend + tileserver together:

```powershell
docker compose -f docker-compose.runner.yml up --build
```

Open:

- Streamlit UI: `http://127.0.0.1:8501`
- Backend API: `http://127.0.0.1:8000`
- Tile server: `http://127.0.0.1:8080`

Stop:

```powershell
docker compose -f docker-compose.runner.yml down
```

Notes:

- Share the full project folder (including `data/`, `models/`, and `tiles/` if needed).
- If `tiles/malaysia_sg_z12.mbtiles` is missing, map basemap may be unavailable.

## API Endpoints (Current)

- `GET /health`
- `GET /models`
- `POST /upload`
- `POST /process/{job_id}`
- `GET /jobs`
- `GET /results/{job_id}`
- `GET /timeline/{job_id}`
- `GET /metrics/{job_id}`
- `GET /detections/{job_id}`
- `POST /admin/delete-jobs`
- `POST /admin/clear-history`

## Notes

- Supported upload types: `.mp4`, `.mov`, `.avi`, `.jpg`, `.jpeg`, `.png`
- Max upload size: `4 GB`
- Data is stored locally in SQLite and filesystem directories under `data/`

## Performance Tuning (Local)

You can tune runtime with environment variables before starting backend:

```powershell
$env:EA_FRAME_STRIDE="2"               # sample every 2nd frame
$env:EA_INFERENCE_DEVICE="auto"        # auto / cpu / cuda:0 / mps
$env:EA_VIDEO_IMGSZ="960"              # lower for speed, higher for small-object recall
$env:EA_WRITE_ANNOTATED_FRAMES="1"     # set 0 to skip expensive box/label drawing
$env:EA_SAVE_EVERY_NTH_FRAME="2"       # store fewer preview frames to reduce I/O
$env:EA_FRAME_JPEG_QUALITY="80"        # lower quality writes faster/smaller files
$env:EA_HABITAT_FRAME_STRIDE="2"       # classify habitat on sampled saved frames
```

Recommended for faster local turnaround with stable tracking:

- Keep `EA_FRAME_STRIDE=2` (good ID continuity vs speed).
- Keep detection tracking on sampled frames, but reduce disk writes with `EA_SAVE_EVERY_NTH_FRAME=2`.
- If GPU exists, keep `EA_INFERENCE_DEVICE=auto` so backend uses GPU automatically.

### Pipeline Mode

`POST /process/{job_id}` now supports `pipeline_mode` query:

- `pipeline_mode=fast_trend`: trend-focused mode using full tracking logic at ~2 FPS
- `pipeline_mode=value_1x`: high value mode using tracking at ~5 FPS (near 1:1 target)
- `pipeline_mode=balanced_2fps`: light ByteTrack mode with ~2 FPS sampling
- `pipeline_mode=quality`: identity-accurate mode (ByteTrack + denser evidence)
- omit `pipeline_mode`: use current environment/default settings

Note:
- `fast_trend` keeps detection boxes enabled, but saves fewer frames to stay fast.

Example:

```powershell
curl -X POST "http://127.0.0.1:8000/process/<job_id>?model_key=labeling_data_v2_s&pipeline_mode=fast_trend"
```

### Benchmark Scripts

Run single-video benchmark and output processing ratio report:

```powershell
python scripts/benchmark/run_benchmark.py --video "data/uploads/<your_video>.mp4" --mode fast --model-key labeling_data_v2_s --runs 1
python scripts/benchmark/run_benchmark.py --video "data/uploads/<your_video>.mp4" --mode quality --model-key labeling_data_v2_s --runs 1
```

Compare two reports:

```powershell
python scripts/benchmark/compare_runs.py --base "<fast_report>.json" --candidate "<quality_report>.json"
```
