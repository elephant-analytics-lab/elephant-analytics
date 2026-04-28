# Dependency Audit

Last updated: 2026-04-23

This file records which libraries are currently required for local delivery, which ones are tied only to the legacy Streamlit frontend, and which ones look removable or optional.

## Current frontend modes

- `docker-compose.runner.yml`: React frontend is now the preferred frontend on port `8501`
- `docker-compose.react.yml`: React frontend alternate launch on port `8502`
- `docker-compose.streamlit.yml`: legacy Streamlit fallback on port `8501`

## Python dependencies in `requirements.txt`

### Required for backend runtime

- `fastapi`
- `uvicorn[standard]`
- `opencv-python-headless`
- `ultralytics`
- `numpy`
- `python-multipart`

Reason:
- These are used directly by `backend/app/main.py`, `backend/app/pipeline/detector.py`, and `backend/app/pipeline/habitat.py`.

### Required only if keeping the Streamlit frontend

- `streamlit`
- `pandas`
- `altair`
- `folium`
- `streamlit-folium`
- `requests`

Reason:
- These are referenced in `streamlit/app.py`.
- If Streamlit remains part of the shipped product, keep them.
- If React fully replaces Streamlit later, these can be removed from Python runtime requirements.

### Present but currently unused or optional

- `celery`
- `plotly`
- `matplotlib`
- `python-dotenv`

Status:
- `celery`: no current code usage found in `backend/` or `streamlit/`
- `plotly`: no current code usage found in `backend/` or `streamlit/`
- `matplotlib`: imported in `streamlit/app.py` but no active usage found
- `python-dotenv`: not imported in current runtime code; optional only if `.env` loading is added later

## React frontend dependencies in `frontend/package.json`

### Required for the new React frontend

- `react`
- `react-dom`
- `react-router-dom`
- `recharts`
- `leaflet`

### Build tooling only

- `vite`
- `@vitejs/plugin-react`

Note:
- These are only needed to build or develop the React frontend.
- They do not belong in Python `requirements.txt`.

## Client delivery implications

### If shipping the current Streamlit fallback

Keep in Python requirements:

- `streamlit`
- `fastapi`
- `uvicorn[standard]`
- `opencv-python-headless`
- `ultralytics`
- `numpy`
- `pandas`
- `altair`
- `folium`
- `streamlit-folium`
- `requests`
- `python-multipart`

Can be removed candidate:

- `celery`
- `plotly`
- `matplotlib`
- `python-dotenv`

### If shipping the React version

Python runtime can be reduced to:

- `fastapi`
- `uvicorn[standard]`
- `opencv-python-headless`
- `ultralytics`
- `numpy`
- `python-multipart`

Potentially still needed depending on backend evolution:

- `requests` is not needed by the backend today
- `pandas`, `altair`, `folium`, `streamlit-folium`, `streamlit` are not needed once Streamlit is removed

## Non-library runtime dependencies

These are not Python packages but are still required by the app:

- `ffmpeg`
- OpenCV system libs from `Dockerfile`
- `exiftool` for GPS extraction
- model weight files under `models/`
- habitat model weight if habitat output is expected
- local tile data under `tiles/` if map tiles are needed

## Verification performed

- React dependencies installed with `npm.cmd install`
- React frontend production build succeeded with `npm.cmd run build`
- Python dependency usage cross-checked with `rg` import search across `backend/` and `streamlit/`
- `requirements.react.txt` added as the React-first backend runtime dependency set
