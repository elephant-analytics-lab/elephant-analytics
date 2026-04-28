# Elephant Analytics

React-first delivery for elephant detection, continuity review, habitat analysis, and GPS-backed spatial context from drone media.

## Stack

- Frontend: React + Vite
- Backend: FastAPI + Uvicorn
- CV/ML: Ultralytics YOLO, OpenCV, NumPy
- Storage: SQLite + local filesystem
- Mapping: Leaflet + optional local TileServer GL

## Active Product Surface

Customer-facing UI is now the React frontend.

Current user-facing options:

- Detection models
  - `Research Model v3_2s`
  - `Online Model`
- Processing modes
  - `fast_trend`
  - `quality`

Streamlit is now a legacy reference surface and is not required for the delivery flow below.

## Project Structure

```text
elephant-analytics/
|- backend/
|  `- app/                      # FastAPI API, pipeline, storage
|- frontend/                    # React frontend
|- data/
|  |- uploads/                  # uploaded media
|  `- outputs/                  # frames, detections, generated artifacts
|- models/                      # active local model weights
|- tiles/                       # optional MBTiles / map assets
|- docker-compose.react.yml     # backend + React + tileserver
`- requirements.txt             # backend runtime dependencies
```

## Required Models

Current runtime expects these model files:

- `models/elephant_v3_labeling_data_v2_s.pt`
- `models/elephant_online_roboflow_v1.pt`
- `models/habitat_cls_v1_best.pt`

## Local Run

### 1. Create a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install backend dependencies

```powershell
pip install -r requirements.txt
```

### 3. Start backend

```powershell
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

### 4. Start frontend

```powershell
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 8501
```

Open:

- React UI: `http://127.0.0.1:8501`
- Backend API: `http://127.0.0.1:8000`

## Docker Run

```powershell
docker compose -f docker-compose.react.yml up --build
```

Open:

- React UI: `http://127.0.0.1:8502`
- Backend API: `http://127.0.0.1:8000`
- Tile server: `http://127.0.0.1:8080`

Stop:

```powershell
docker compose -f docker-compose.react.yml down
```

## API Endpoints

- `GET /health`
- `GET /models`
- `POST /upload`
- `POST /process/{job_id}`
- `GET /jobs`
- `POST /jobs/{job_id}/cancel`
- `GET /results/{job_id}`
- `GET /timeline/{job_id}`
- `GET /metrics/{job_id}`
- `GET /detections/{job_id}`
- `GET /media/preview/{job_id}/meta`
- `GET /media/frame/{job_id}/{frame_index}`
- `POST /admin/delete-jobs`
- `POST /admin/clear-history`

## Runtime Notes

- Supported uploads: `.mp4`, `.mov`, `.avi`, `.jpg`, `.jpeg`, `.png`
- Max upload size: `4 GB`
- Data is stored locally under `data/`
- Habitat outputs require `models/habitat_cls_v1_best.pt`

## Performance Tuning

Tune backend runtime with environment variables before startup:

```powershell
$env:EA_FRAME_STRIDE="2"
$env:EA_INFERENCE_DEVICE="auto"
$env:EA_VIDEO_IMGSZ="960"
$env:EA_WRITE_ANNOTATED_FRAMES="1"
$env:EA_SAVE_EVERY_NTH_FRAME="2"
$env:EA_FRAME_JPEG_QUALITY="80"
$env:EA_HABITAT_FRAME_STRIDE="2"
```

Recommended defaults:

- `fast_trend` for quick triage and lighter review runs
- `quality` for stronger continuity and denser saved evidence
- `EA_INFERENCE_DEVICE=auto` on GPU-capable machines

Example:

```powershell
curl -X POST "http://127.0.0.1:8000/process/<job_id>?model_key=labeling_data_v2_s&pipeline_mode=fast_trend"
```
