# Elephant Analytics (Local MVP)

Detect, track, and analyze elephants in drone videos — **fully offline** 

## Status
- ✅ Repo scaffolded (app/ui/docs/data/models)
- ✅ Local-only stack
- ⬜ Pipeline code
- ⬜ ByteTrack + analytics charts

## Stack (local only)
- UI: **Streamlit**
- API (optional now, recommended later): **FastAPI** served by **Uvicorn**
- Background (optional): **Celery**
- Video prep: **FFmpeg** (system install)
- CV/Draw: **OpenCV**
- Detection: **Ultralytics YOLO**
- Tracking: simple IoU (placeholder for ByteTrack)
- Analytics: **NumPy**, **Pandas**, **Plotly**
- Storage: **SQLite** + local disk

## Folder Structure

app/ # backend (FastAPI, pipeline, utils)
ui/ # Streamlit UI
docs/ # notes/diagrams
models/ # YOLO weights (.pt) — ignored by git
data/ # local artifacts — ignored by git
uploads/
outputs/
workspace/