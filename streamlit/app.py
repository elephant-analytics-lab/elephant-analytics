import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import cv2
from datetime import datetime
import time
import requests
from pathlib import Path
from datetime import timezone, timedelta

import altair as alt
import matplotlib.pyplot as plt
from contextlib import contextmanager

# NEW: Folium
import folium
from streamlit_folium import st_folium

# ===================== CONFIG =====================

BACKEND_BASE_URL = os.getenv("EA_BACKEND_BASE_URL", "http://127.0.0.1:8000")
MAX_VIDEO_MB = 4096
TILE_SERVER_URL = os.getenv("EA_TILE_URL", "")
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

MODEL_OPTIONS = {
    "V1 - YOLOv8n v4 (trained)": "default_v4",
    "V2 - Labeling Data v1": "labeling_data_v1",
    "V3 - Labeling Data v2_s": "labeling_data_v2_s",
    "Online Model - Roboflow v1": "roboflow_v1_fresh2",
}

# Keep all internal model mappings, but only expose curated options in UI.
VISIBLE_MODEL_OPTIONS = {
    "Research Model v2_s (Recommended)": MODEL_OPTIONS["V3 - Labeling Data v2_s"],
    "Online Baseline (Roboflow v1)": MODEL_OPTIONS["Online Model - Roboflow v1"],
}

PIPELINE_MODE_OPTIONS = {
    "Original Pipeline (Identity Accurate)": "quality",
    "Fast Trend (2 FPS)": "fast_trend",
    "Value 1x (Tracked ~5 FPS)": "value_1x",
}
SHOW_PIPELINE_MODE_CONTROL = True
DEFAULT_PIPELINE_MODE_LABEL = "Original Pipeline (Identity Accurate)"

st.set_page_config(
    page_title="Elephant Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "use_mock" not in st.session_state:
    st.session_state["use_mock"] = False

# ---------- Simple theme (CSS) ----------
st.markdown(
    """
    <style>
    :root {
        --font-ui: "Segoe UI", "Helvetica Neue", "Noto Sans", Arial, sans-serif;
        --color-forest-green: #2d5016;
        --color-forest-green-light: #4f7f29;
        --color-warm-orange: #e07b39;
        --color-warm-orange-light: #f59550;
        --color-border: #e5e5e0;
        --color-surface: rgba(2, 6, 23, 0.2);
        --color-surface-strong: rgba(2, 6, 23, 0.32);
        --color-text-primary: #222222;
        --color-text-secondary: #555555;
        --color-text-muted: #888888;
        --space-1: 0.35rem;
        --space-2: 0.6rem;
        --space-3: 0.9rem;
        --space-4: 1.2rem;
        --radius-md: 12px;
        --radius-lg: 18px;
        --shadow-card: 0 8px 18px rgba(15,23,42,0.06);
        --shadow-card-hover: 0 16px 30px rgba(15,23,42,0.10);
    }
    html, body, [class^="css"] {
        font-family: var(--font-ui);
    }
    button[title="View fullscreen"],
    button[aria-label="View fullscreen"],
    [data-testid="StyledFullScreenButton"],
    [data-testid="stImage"] button,
    [data-testid="stImageContainer"] button {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
        width: 0 !important;
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        border: 0 !important;
    }

    /* If your Streamlit supports st.container(border=True), style the bordered wrapper nicely */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: var(--radius-lg) !important;
        box-shadow: var(--shadow-card);
        border: 1px solid rgba(229,229,224,0.9) !important;
        padding: 1rem 1.2rem !important;
        background: rgba(255,255,255,0.02);
    }

    .ea-pill {
        display: inline-block;
        padding: 0.15rem 0.65rem;
        border-radius: 999px;
        font-size: 0.7rem;
    }
    .ea-pill-adult    { background: #dcfce7; color: #166534; }
    .ea-pill-juvenile { background: #dbeafe; color: #1d4ed8; }
    .ea-pill-baby     { background: #fef3c7; color: #b45309; }
    .ea-badge-complete  { background:#dcfce7; color:#166534; }
    .ea-badge-processing{ background:#dbeafe; color:#1d4ed8; }
    .ea-badge-queued    { background:#fef3c7; color:#b45309; }
    .ea-badge-failed    { background:#fee2e2; color:#b91c1c; }

    .ea-hero-title {
        font-size: 1.9rem;
        font-weight: 650;
        margin-bottom: 0.5rem;
    }
    .ea-hero-subtitle {
        font-size: 1rem;
        color: var(--color-text-secondary);
    }

    /* Sidebar look-and-feel */
    [data-testid="stSidebar"] {
        background:
            radial-gradient(560px 200px at 0% -10%, rgba(245, 158, 11, 0.14), transparent 62%),
            radial-gradient(420px 180px at 100% 0%, rgba(34, 197, 94, 0.10), transparent 65%),
            linear-gradient(180deg, #141a24 0%, #111722 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.24);
    }
    [data-testid="stSidebar"] h1 {
        font-size: 1.9rem !important;
        font-weight: 820 !important;
        letter-spacing: 0.2px;
        margin-bottom: 0.7rem;
        color: #f8fafc !important;
        text-shadow:
            0 1px 0 rgba(255,255,255,0.16),
            0 8px 18px rgba(0,0,0,0.34);
    }
    [data-testid="stSidebar"] label[data-testid="stWidgetLabel"] p {
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 700;
        color: #f59e0b !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] > label {
        border: 1px solid rgba(148, 163, 184, 0.28);
        border-radius: 12px;
        margin-bottom: 0.38rem;
        padding: 0.38rem 0.5rem;
        background: rgba(15, 23, 42, 0.28);
        transition: all 140ms ease;
    }
    [data-testid="stSidebar"] [role="radiogroup"] > label:hover {
        border-color: rgba(251, 191, 36, 0.55);
        background: rgba(15, 23, 42, 0.48);
        transform: translateX(2px);
    }
    [data-testid="stSidebar"] [role="radiogroup"] > label p {
        color: #e2e8f0 !important;
        font-size: 0.95rem !important;
        font-weight: 620;
        text-shadow: 0 1px 0 rgba(0,0,0,0.28);
    }
    [data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {
        border-color: rgba(245, 158, 11, 0.92);
        background: linear-gradient(90deg, rgba(245, 158, 11, 0.20), rgba(245, 158, 11, 0.08));
        box-shadow: 0 0 0 1px rgba(245, 158, 11, 0.18) inset, 0 10px 18px rgba(0,0,0,0.24);
    }
    [data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) p {
        color: #fff7ed !important;
        font-weight: 760;
    }
    /* Hide Streamlit uploader plus icon for cleaner single-file UX */
    [data-testid="stFileUploader"] section [aria-hidden="true"] {
        display: none !important;
    }
    .ea-surface {
        border: 1px solid rgba(148, 163, 184, 0.24);
        border-radius: var(--radius-md);
        background: var(--color-surface);
        padding: var(--space-3) var(--space-4);
    }
    .ea-surface-strong {
        border: 1px solid rgba(148, 163, 184, 0.28);
        border-radius: var(--radius-md);
        background: linear-gradient(180deg, var(--color-surface-strong), rgba(2, 6, 23, 0.16));
        padding: var(--space-3) var(--space-4);
    }
    .ea-mini-kpi {
        border: 1px solid rgba(148, 163, 184, 0.24);
        border-radius: var(--radius-md);
        background: rgba(2, 6, 23, 0.26);
        padding: 0.72rem 0.75rem;
    }
    .ea-mini-kpi .k {
        margin: 0;
        font-size: 0.75rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #94a3b8;
        font-weight: 700;
    }
    .ea-mini-kpi .v {
        margin: 0.16rem 0 0 0;
        font-size: 1.2rem;
        font-weight: 780;
        color: #f8fafc;
        line-height: 1.1;
    }
    .ea-page-hero {
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 16px;
        padding: 1.08rem 1.1rem 0.95rem 1.1rem;
        margin-bottom: 0.85rem;
        background:
            radial-gradient(900px 180px at 12% -10%, rgba(245,158,11,0.14), transparent 62%),
            radial-gradient(780px 190px at 88% -20%, rgba(34,197,94,0.12), transparent 62%),
            linear-gradient(180deg, rgba(15,23,42,0.24), rgba(15,23,42,0.08));
    }
    .ea-page-hero .kicker {
        margin: 0;
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #f59e0b;
        font-weight: 700;
    }
    .ea-page-hero h2 {
        margin: 0.16rem 0 0.22rem 0;
        font-size: 1.52rem;
        letter-spacing: -0.2px;
    }
    .ea-page-hero p {
        margin: 0;
        color: #cbd5e1;
        font-size: 0.91rem;
    }
    .ea-block-head {
        margin: 0 0 0.55rem 0;
        padding-left: 0.62rem;
        border-left: 3px solid rgba(56, 189, 248, 0.82);
        font-size: 1.06rem;
        font-weight: 760;
        letter-spacing: 0.01em;
        color: #f8fafc;
    }
    .ea-note {
        margin: 0;
        color: #94a3b8;
        font-size: 0.83rem;
    }
    .vega-actions,
    .vega-embed summary {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ===================== SMALL UI HELPERS =====================

@contextmanager
def card():
    """
    Use a bordered container if supported, otherwise fall back to plain container.
    This avoids the "empty white blocks" caused by trying to wrap Streamlit components in raw HTML divs.
    """
    try:
        with st.container(border=True):
            yield
    except TypeError:
        with st.container():
            yield


# ===================== MOCK DATA HELPERS =====================

def mock_detection_summary():
    return {
        "unique_elephants": 12,
        "age_distribution": {"adult": 67, "juvenile": 25, "baby": 8},
        "avg_herd_size": 4.3,
        "main_habitats": "Grassland, Oil Palm",
        "video_duration": "8:42",
    }

def empty_detection_summary():
    return {
        "unique_elephants": 0,
        "age_distribution": {"adult": 0, "juvenile": 0, "baby": 0},
        "avg_herd_size": 0.0,
        "main_habitats": "-",
        "video_duration": "-",
    }

def mock_herd_series():
    times = [
        "0:00","0:30","1:00","1:30","2:00","2:30","3:00","3:30",
        "4:00","4:30","5:00","5:30","6:00","6:30","7:00","7:30","8:00"
    ]
    counts = [3, 5, 7, 6, 8, 9, 7, 6, 5, 4, 6, 8, 7, 5, 4, 5, 6]
    return pd.DataFrame({"time": times, "herd_size": counts})

def empty_herd_series():
    return pd.DataFrame({"time": [], "herd_size": []})

def mock_behaviour_distribution():
    return pd.Series(
        {
            "Grazing": 42,
            "Travelling": 28,
            "Resting": 18,
            "Socialising": 8,
            "Drinking": 4,
        }
    )

def empty_behaviour_distribution():
    return pd.Series({})

def mock_timeline_table():
    rows = [
        ("ELE_001", "Adult", "0:12", "8:23", "8:11", "Herd"),
        ("ELE_002", "Adult", "0:15", "7:45", "7:30", "Herd"),
        ("ELE_003", "Juvenile", "0:18", "8:18", "8:00", "Herd"),
        ("ELE_004", "Adult", "1:22", "6:34", "5:12", "Herd"),
        ("ELE_005", "Baby", "1:30", "5:42", "4:12", "Herd"),
        ("ELE_006", "Adult", "2:05", "8:35", "6:30", "Alone"),
        ("ELE_007", "Juvenile", "2:18", "7:22", "5:04", "Herd"),
        ("ELE_008", "Adult", "3:12", "8:12", "5:00", "Herd"),
    ]
    return pd.DataFrame(
        rows,
        columns=["Elephant ID", "Age class", "First seen", "Last seen", "Duration", "Alone vs herd"],
    )

def empty_timeline_table():
    return pd.DataFrame(
        columns=["Elephant ID", "Age class", "First seen", "Last seen", "Duration", "Alone vs herd"],
    )

def mock_jobs_table():
    rows = [
        ("JOB_2341", "video_001.mp4", "complete", "2025-12-10 14:23", "8:42"),
        ("JOB_2340", "video_002.mp4", "processing", "2025-12-10 13:45", "12:15"),
        ("JOB_2339", "drone_footage_003.mov", "queued", "2025-12-10 13:12", "6:30"),
        ("JOB_2338", "elephant_herd_4k.mp4", "complete", "2025-12-09 16:28", "15:22"),
        ("JOB_2337", "test_video.mp4", "failed", "2025-12-09 15:03", "3:45"),
    ]
    return pd.DataFrame(
        rows, columns=["Job ID", "Video name", "Status", "Submitted at", "Duration"]
    )


def fetch_jobs_from_backend() -> pd.DataFrame | None:
    try:
        resp = requests.get(f"{BACKEND_BASE_URL}/jobs", timeout=10)
        resp.raise_for_status()
        jobs = resp.json()
    except Exception:
        return None

    if not jobs:
        return None

    df = pd.DataFrame(jobs)
    if df.empty:
        return None

    df = df.rename(
        columns={
            "id": "Job ID",
            "filename": "Video name",
            "status": "Status",
            "created_at": "Submitted at",
            "duration_seconds": "Duration (s)",
        }
    )
    if "Duration (s)" not in df.columns:
        df["Duration (s)"] = ""
    return df[["Job ID", "Video name", "Status", "Submitted at", "Duration (s)"]]


def fetch_job_metrics(job_id: str) -> dict | None:
    cache_key = f"metrics_{job_id}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    try:
        resp = requests.get(f"{BACKEND_BASE_URL}/metrics/{job_id}", timeout=20)
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return None
    st.session_state[cache_key] = payload
    return payload


def fetch_job_detections(job_id: str) -> list[dict] | None:
    cache_key = f"detections_{job_id}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    try:
        resp = requests.get(f"{BACKEND_BASE_URL}/detections/{job_id}", timeout=20)
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return None
    detections = payload.get("detections", [])
    st.session_state[cache_key] = detections
    return detections


def normalize_age_distribution(dist: dict) -> dict:
    normalized = {"adult": 0.0, "juvenile": 0.0, "baby": 0.0}
    for label, value in (dist or {}).items():
        key = str(label).strip().lower()
        if key in normalized:
            normalized[key] = value
    return normalized


def normalize_age_class_label(value: object) -> str:
    txt = str(value or "").strip().lower()
    if "baby" in txt or "calf" in txt:
        return "Baby"
    if "juvenile" in txt or "subadult" in txt:
        return "Juvenile"
    if "adult" in txt:
        return "Adult"
    return "Unknown"


def parse_track_id(value: object) -> int | None:
    if value is None:
        return None
    txt = str(value).strip()
    if txt.isdigit():
        return int(txt)
    digits = "".join(ch for ch in txt if ch.isdigit())
    if digits:
        return int(digits)
    return None


def get_real_summary() -> dict | None:
    job_id = st.session_state.get("last_job_id")
    if not job_id:
        return None
    metrics = fetch_job_metrics(job_id)
    if not metrics:
        return None
    age_dist = normalize_age_distribution(metrics.get("age_distribution", {}))
    return {
        "unique_elephants": metrics.get("unique_tracks", 0),
        "age_distribution": age_dist,
        "avg_herd_size": metrics.get("avg_herd_size", 0.0),
        "main_habitats": "-",
        "video_duration": "-",
    }


def format_my_time(value: str) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    try:
        dt = datetime.fromisoformat(value)
        dt = dt.astimezone(timezone(timedelta(hours=8)))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return value


def format_duration(value) -> str:
    if value is None or value == "" or (isinstance(value, float) and pd.isna(value)):
        return ""
    try:
        seconds = int(float(value))
    except Exception:
        return str(value)
    if seconds <= 0:
        seconds = 1
    mins, secs = divmod(seconds, 60)
    if mins:
        return f"{mins}m {secs}s"
    return f"{secs}s"


def format_seconds(value) -> str:
    try:
        total = int(round(float(value)))
    except Exception:
        return str(value)
    mins, secs = divmod(total, 60)
    return f"{mins}:{secs:02d}"


def media_duration_seconds(video_path: str | None) -> float | None:
    if not video_path:
        return None
    suffix = Path(str(video_path)).suffix.lower()
    if suffix not in {".mp4", ".mov", ".avi"}:
        return None
    try:
        cap = cv2.VideoCapture(str(video_path))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        cap.release()
    except Exception:
        return None
    if fps <= 0 or frame_count <= 0:
        return None
    return frame_count / fps


def resolve_job_video_path(job_id: str | None) -> str | None:
    if not job_id:
        return None
    raw_path = None
    payload = st.session_state.get("last_results_payload")
    if payload and (payload.get("job", {}).get("id") == job_id):
        raw_path = payload.get("job", {}).get("video_path")
    else:
        try:
            resp = requests.get(f"{BACKEND_BASE_URL}/results/{job_id}", timeout=10)
            resp.raise_for_status()
            raw_path = resp.json().get("job", {}).get("video_path")
        except Exception:
            raw_path = None

    if not raw_path:
        return None
    p = Path(str(raw_path))
    if p.exists():
        return str(p)
    alt = DATA_DIR / "uploads" / p.name
    if alt.exists():
        return str(alt)
    return None


def format_percent(value) -> str:
    try:
        pct = float(value) * 100.0
    except Exception:
        return "-"
    return f"{pct:.0f}%"


def is_video_filename(name: str | None) -> bool:
    suffix = Path(str(name or "")).suffix.lower()
    return suffix in {".mp4", ".mov", ".avi"}


def latest_gps_df() -> pd.DataFrame | None:
    payload = st.session_state.get("last_results_payload")
    if not payload:
        return None
    gps = payload.get("gps") or []
    if not gps:
        return None
    df = pd.DataFrame(gps)
    if "lat" not in df.columns or "lon" not in df.columns:
        return None
    return df[["lat", "lon"]]


def cached_gps_points(job_id: str) -> pd.DataFrame | None:
    cache_key = f"gps_points_{job_id}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    payload = st.session_state.get("last_results_payload")
    if payload and payload.get("job", {}).get("id") == job_id:
        gps = payload.get("gps") or []
    else:
        try:
            resp = requests.get(f"{BACKEND_BASE_URL}/results/{job_id}", timeout=20)
            resp.raise_for_status()
            gps = resp.json().get("gps") or []
        except Exception:
            gps = []

    if not gps:
        return None
    df = pd.DataFrame(gps)
    if "lat" not in df.columns or "lon" not in df.columns:
        return None
    st.session_state[cache_key] = df[["lat", "lon"]]
    return st.session_state[cache_key]


def resolve_tile_server_url() -> str:
    cache_key = "_resolved_tile_server_url"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    if TILE_SERVER_URL:
        st.session_state[cache_key] = TILE_SERVER_URL
        return TILE_SERVER_URL

    index_candidates = [
        "http://localhost:8080/index.json",
        "http://127.0.0.1:8080/index.json",
    ]
    resolved = ""
    for index_url in index_candidates:
        try:
            resp = requests.get(index_url, timeout=2)
            resp.raise_for_status()
            payload = resp.json()
            if isinstance(payload, list):
                for item in payload:
                    tiles = item.get("tiles") if isinstance(item, dict) else None
                    if isinstance(tiles, list) and tiles:
                        tile_url = str(tiles[0])
                        if "{z}" in tile_url and "{x}" in tile_url and "{y}" in tile_url:
                            resolved = tile_url
                            break
            if resolved:
                break
        except Exception:
            continue

    st.session_state[cache_key] = resolved
    return resolved


def downsample_points(df: pd.DataFrame, max_points: int = 300) -> pd.DataFrame:
    if df is None or df.empty or len(df) <= max_points:
        return df
    step = max(1, len(df) // max_points)
    return df.iloc[::step].reset_index(drop=True)


def latest_available_frame_files() -> tuple[str | None, list[Path]]:
    frames_root = DATA_DIR / "outputs" / "frames"
    if not frames_root.exists():
        return None, []

    newest_job_id: str | None = None
    newest_ts: float | None = None
    newest_files: list[Path] = []

    for job_dir in frames_root.iterdir():
        if not job_dir.is_dir():
            continue
        files = sorted(job_dir.glob("frame_*.jpg"))
        if not files:
            continue
        ts = max(p.stat().st_mtime for p in files)
        if newest_ts is None or ts > newest_ts:
            newest_ts = ts
            newest_job_id = job_dir.name
            newest_files = files

    return newest_job_id, newest_files


def frame_index_from_path(path: Path) -> int | None:
    parts = path.stem.split("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return int(parts[1])
    return None


def ensure_preview_video(job_id: str | None, frame_files: list[Path], fps: int = 12) -> Path | None:
    if not job_id or not frame_files:
        return None
    try:
        latest_mtime = int(max(p.stat().st_mtime for p in frame_files))
    except Exception:
        return None
    signature = f"{len(frame_files)}_{latest_mtime}"
    previews_dir = DATA_DIR / "outputs" / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)
    out_path = previews_dir / f"{job_id}_{signature}.mp4"
    if out_path.exists():
        return out_path

    first = cv2.imread(str(frame_files[0]))
    if first is None:
        return None
    h, w = first.shape[:2]
    if h <= 0 or w <= 0:
        return None
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(max(1, fps)),
        (int(w), int(h)),
    )
    if not writer.isOpened():
        return None

    try:
        for fp in frame_files:
            img = cv2.imread(str(fp))
            if img is None:
                continue
            if img.shape[0] != h or img.shape[1] != w:
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(img)
    finally:
        writer.release()
    return out_path if out_path.exists() else None


def resolve_job_is_video(job_id: str | None, metrics: dict | None = None) -> bool:
    if not job_id:
        return True

    payload = st.session_state.get("last_results_payload")
    if payload and (payload.get("job") or {}).get("id") == job_id:
        return is_video_filename((payload.get("job") or {}).get("filename"))

    jobs_df = fetch_jobs_from_backend()
    if jobs_df is not None and not jobs_df.empty and "Job ID" in jobs_df.columns and "Video name" in jobs_df.columns:
        matched = jobs_df[jobs_df["Job ID"].astype(str) == str(job_id)]
        if not matched.empty:
            return is_video_filename(str(matched.iloc[0].get("Video name", "")))

    series = (metrics or {}).get("herd_series")
    if isinstance(series, list) and len(series) > 0:
        return True

    return False


def fetch_timeline_rows(job_id: str | None) -> list[dict]:
    if not job_id:
        return []
    try:
        resp = requests.get(f"{BACKEND_BASE_URL}/timeline/{job_id}", timeout=12)
        resp.raise_for_status()
        return resp.json().get("timeline", []) or []
    except Exception:
        return []


def estimate_route_distance_km(points_df: pd.DataFrame | None) -> float:
    if points_df is None or points_df.empty or len(points_df) < 2:
        return 0.0
    df = points_df.copy()
    lat = np.radians(pd.to_numeric(df["lat"], errors="coerce").to_numpy(dtype=float))
    lon = np.radians(pd.to_numeric(df["lon"], errors="coerce").to_numpy(dtype=float))
    if len(lat) < 2:
        return 0.0
    dlat = lat[1:] - lat[:-1]
    dlon = lon[1:] - lon[:-1]
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * (np.sin(dlon / 2.0) ** 2)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1e-12, 1.0 - a)))
    return float(np.sum(6371.0 * c))


def estimate_track_distance_px(track_df: pd.DataFrame | None) -> float:
    if track_df is None or track_df.empty or len(track_df) < 2:
        return 0.0
    df = track_df.sort_values("frame_index").copy()
    x_center = (pd.to_numeric(df["x1"], errors="coerce") + pd.to_numeric(df["x2"], errors="coerce")) / 2.0
    y_center = (pd.to_numeric(df["y1"], errors="coerce") + pd.to_numeric(df["y2"], errors="coerce")) / 2.0
    x = x_center.to_numpy(dtype=float)
    y = y_center.to_numpy(dtype=float)
    if len(x) < 2:
        return 0.0
    step_dist = np.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2)
    return float(np.nansum(step_dist))


def preview_section_header(title: str, subtitle: str | None = None) -> None:
    subtitle_html = f'<p style="margin:0.2rem 0 0 0; color:#94a3b8; font-size:0.84rem;">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f"""
        <div style="margin:0.15rem 0 0.55rem 0;">
            <h3 style="margin:0; font-size:1.06rem; letter-spacing:-0.1px;">{title}</h3>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_hero(title: str, subtitle: str, kicker: str = "Analysis") -> None:
    st.markdown(
        f"""
        <section class="ea-page-hero">
            <p class="kicker">{kicker}</p>
            <h2>{title}</h2>
            <p>{subtitle}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


# ===================== PAGE SECTIONS =====================

def hero_section():
    render_page_hero(
        "Client Summary",
        "Clear evidence and key numbers from the latest processed job.",
        kicker="Preview",
    )

    col_main, col_stats = st.columns([1.6, 1])

    job_id = st.session_state.get("last_job_id")
    use_mock = st.session_state.get("use_mock", False)
    summary = (
        mock_detection_summary()
        if use_mock
        else (get_real_summary() or empty_detection_summary())
    )
    metrics = fetch_job_metrics(job_id) if (job_id and not use_mock) else None
    dominant_habitat = "-"
    if metrics:
        dominant_habitat = str(metrics.get("dominant_habitat") or "-")

    unique_count = int(summary.get("unique_elephants", 0) or 0)
    avg_herd = float(summary.get("avg_herd_size", 0.0) or 0.0)
    mode_key_for_summary = str(st.session_state.get("last_pipeline_mode_key", "") or "").strip().lower()
    is_fast_trend_mode_summary = mode_key_for_summary in {"fast_trend", "fast"}
    if is_fast_trend_mode_summary:
        conclusion = (
            f"Fast trend mode: average herd size {avg_herd:.1f}. "
            f"Dominant habitat: {dominant_habitat}. Use Identity Accurate mode for final unique-ID totals."
        )
    elif unique_count > 0:
        conclusion = (
            f"Detected {unique_count} elephants with an average herd size of {avg_herd:.1f}. "
            f"Dominant habitat: {dominant_habitat}."
        )
    else:
        conclusion = "No confirmed elephant tracks in the current output. Validate input quality and rerun if needed."

    with col_main:
        display_job_id = job_id
        frame_files: list[Path] = []

        if job_id:
            frames_dir = DATA_DIR / "outputs" / "frames" / job_id
            frame_files = sorted(frames_dir.glob("frame_*.jpg")) if frames_dir.exists() else []

        if not frame_files:
            fallback_job_id, fallback_files = latest_available_frame_files()
            if fallback_files:
                display_job_id = fallback_job_id
                frame_files = fallback_files
                if not metrics and display_job_id and not use_mock:
                    metrics = fetch_job_metrics(display_job_id)
                    dominant_habitat = str((metrics or {}).get("dominant_habitat") or "-")

        preview_is_video = True if use_mock else resolve_job_is_video(display_job_id, metrics=metrics)
        last_mode_key = str(st.session_state.get("last_pipeline_mode_key", "") or "").strip().lower()
        selected: Path | None = None
        if frame_files:
            if len(frame_files) == 1:
                selected = frame_files[0]
            else:
                idx = st.slider(
                    "Preview frame",
                    min_value=0,
                    max_value=len(frame_files) - 1,
                    value=0,
                    step=1,
                    key=f"hero_preview_frame_{display_job_id or 'latest'}",
                )
                selected = frame_files[idx]
            st.image(str(selected), caption="Field evidence snapshot", width="stretch")

            if display_job_id and display_job_id != job_id:
                st.caption(f"Showing latest available preview from job `{display_job_id}`.")
        else:
            st.info("No processed frame available yet. Run one analysis in Upload & jobs.")

        selected_frame_idx = frame_index_from_path(selected) if selected else None
        st.session_state["preview_display_job_id"] = display_job_id
        st.session_state["preview_frame_index"] = selected_frame_idx
        st.session_state["preview_is_video"] = preview_is_video

        st.markdown(
            f"""
            <div class="ea-surface" style="margin-top:0.55rem;">
                <p style="margin:0; font-size:0.76rem; letter-spacing:0.06em; text-transform:uppercase; color:#94a3b8; font-weight:700;">Summary</p>
                <p style="margin:0.24rem 0 0 0; color:#f8fafc; font-size:0.94rem;">{conclusion}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_stats:
        with card():
            st.markdown("#### Run summary")
            mode_key = str(st.session_state.get("last_pipeline_mode_key", "") or "").strip().lower()
            is_fast_trend_mode = mode_key in {"fast_trend", "fast"}
            k1, k2, k3 = st.columns(3)
            with k1:
                if is_fast_trend_mode:
                    peak_herd = int((metrics or {}).get("peak_herd_size", 0) or 0)
                    st.metric("Maximum seen at once", peak_herd)
                else:
                    st.metric("Unique tracks", unique_count)
            with k2:
                st.metric("Average group size", f"{avg_herd:.2f}")
            with k3:
                st.metric("Dominant habitat", dominant_habitat)

            st.markdown("")
            selected_model = st.session_state.get("last_model_choice", "-")
            selected_pipeline_mode = st.session_state.get("last_pipeline_mode_choice", "-")
            processing_time_txt = "-"
            video_duration_txt = "-"
            proc_seconds_val = None
            video_seconds_val = None
            if display_job_id and not use_mock:
                jobs_meta = fetch_jobs_from_backend()
                if jobs_meta is not None and not jobs_meta.empty:
                    hit = jobs_meta[jobs_meta["Job ID"].astype(str) == str(display_job_id)]
                    if not hit.empty:
                        proc_sec = pd.to_numeric(hit.iloc[0].get("Duration (s)"), errors="coerce")
                        if pd.notna(proc_sec):
                            proc_seconds_val = float(proc_sec)
                            processing_time_txt = format_duration(float(proc_sec)) or "-"

                if st.session_state.get("preview_is_video", True):
                    video_cache_key = f"video_duration_seconds_{display_job_id}"
                    if video_cache_key in st.session_state:
                        video_seconds = st.session_state.get(video_cache_key)
                    else:
                        video_path = None
                        payload = st.session_state.get("last_results_payload")
                        if payload and (payload.get("job", {}).get("id") == display_job_id):
                            video_path = payload.get("job", {}).get("video_path")
                        if not video_path:
                            try:
                                resp = requests.get(f"{BACKEND_BASE_URL}/results/{display_job_id}", timeout=10)
                                resp.raise_for_status()
                                video_path = resp.json().get("job", {}).get("video_path")
                            except Exception:
                                video_path = None
                        video_seconds = media_duration_seconds(video_path)
                        st.session_state[video_cache_key] = video_seconds
                    if video_seconds is not None:
                        video_seconds_val = float(video_seconds)
                        video_duration_txt = format_seconds(video_seconds)

            input_type_txt = "Video" if st.session_state.get("preview_is_video", True) else "Image"
            ratio_txt = "-"
            if (
                input_type_txt == "Video"
                and video_seconds_val is not None
                and proc_seconds_val is not None
                and video_seconds_val > 0
            ):
                ratio_txt = f"{(proc_seconds_val / video_seconds_val):.2f}x"

            st.markdown(
                f"""
                <div class="ea-surface" style="padding:0.85rem 0.9rem; min-height:190px;">
                    <p style="margin:0; color:#f8fafc; font-size:0.92rem; font-weight:760;">Run details</p>
                    <div style="display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:0.7rem; margin-top:0.55rem;">
                        <div style="border:1px solid rgba(148,163,184,0.25); border-radius:10px; padding:0.58rem 0.62rem; background:rgba(2,6,23,0.22);">
                            <p style="margin:0; color:#94a3b8; font-size:0.74rem;">Video duration</p>
                            <p style="margin:0.12rem 0 0 0; color:#93c5fd; font-size:1.32rem; font-weight:780; line-height:1.05;">{video_duration_txt}</p>
                        </div>
                        <div style="border:1px solid rgba(148,163,184,0.25); border-radius:10px; padding:0.58rem 0.62rem; background:rgba(2,6,23,0.22);">
                            <p style="margin:0; color:#94a3b8; font-size:0.74rem;">Processing time</p>
                            <p style="margin:0.12rem 0 0 0; color:#93c5fd; font-size:1.32rem; font-weight:780; line-height:1.05;">{processing_time_txt}</p>
                        </div>
                    </div>
                    <p style="margin:0.45rem 0 0 0; color:#cbd5e1; font-size:0.82rem;">Processing ratio: <span style="color:#f8fafc; font-weight:700;">{ratio_txt}</span></p>
                    <p style="margin:0.3rem 0 0 0; color:#cbd5e1; font-size:0.82rem;">Input type: {input_type_txt}</p>
                    <p style="margin:0.3rem 0 0 0; color:#cbd5e1; font-size:0.82rem;">Model: {selected_model}</p>
                    <p style="margin:0.3rem 0 0 0; color:#cbd5e1; font-size:0.82rem;">Job ID: <code>{display_job_id or '-'}</code></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if is_fast_trend_mode:
                pass

def individual_spotlight():
    preview_section_header("Individuals", "Who appears in the selected evidence frame.")
    display_job_id = st.session_state.get("preview_display_job_id") or st.session_state.get("last_job_id")
    use_mock = st.session_state.get("use_mock", False)
    selected_frame_idx = st.session_state.get("preview_frame_index")

    if use_mock:
        st.caption("Demo mode")
        demo_rows = [
            {"id": "ID 01", "age": "Adult", "confidence": "98%"},
            {"id": "ID 02", "age": "Juvenile", "confidence": "92%"},
            {"id": "ID 03", "age": "Baby", "confidence": "89%"},
        ]
        cols = st.columns(3)
        for i, row in enumerate(demo_rows):
            with cols[i % 3]:
                st.markdown(
                    f"""
                    <div class="ea-surface" style="padding:0.75rem; min-height:132px;">
                        <p style="margin:0; font-size:0.72rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.06em;">{row['age']}</p>
                        <p style="margin:0.34rem 0 0 0; color:#f8fafc; font-size:1.06rem; font-weight:700;">{row['id']}</p>
                        <p style="margin:0.26rem 0 0 0; color:#cbd5e1; font-size:0.84rem;">Confidence: {row['confidence']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        return

    if not display_job_id:
        st.info("No processed result available yet.")
        return

    detections = fetch_job_detections(display_job_id) or []
    if not detections:
        st.info("No detection rows found for this job.")
        return

    df = pd.DataFrame(detections)
    if df.empty:
        st.info("No detection rows found for this job.")
        return

    if selected_frame_idx is not None and "frame_index" in df.columns:
        frame_df = df[df["frame_index"] == int(selected_frame_idx)].copy()
    else:
        frame_df = df.copy()

    if frame_df.empty and "frame_index" in df.columns:
        latest_idx = int(df["frame_index"].max())
        frame_df = df[df["frame_index"] == latest_idx].copy()

    if frame_df.empty:
        st.info("No detections on this frame.")
        return

    frame_df["track_id"] = frame_df["track_id"].astype(int)
    frame_df["age_label"] = frame_df["cls_label"].apply(normalize_age_class_label)
    frame_df["conf"] = pd.to_numeric(frame_df["conf"], errors="coerce").fillna(0.0)
    frame_df = frame_df.sort_values(["track_id", "conf"], ascending=[True, False])
    uniq_df = frame_df.drop_duplicates(subset=["track_id"], keep="first")
    uniq_df = uniq_df.sort_values("track_id")

    st.caption(f"{len(uniq_df)} individuals on this frame")
    view_mode = st.segmented_control(
        "View",
        options=["Cards", "Table"],
        default="Cards",
        key=f"spotlight_view_mode_{display_job_id}",
        label_visibility="collapsed",
    )
    if view_mode == "Table":
        table_df = uniq_df.rename(
            columns={
                "track_id": "ID",
                "age_label": "Age class",
                "conf": "Confidence",
            }
        )
        table_df["ID"] = table_df["ID"].apply(lambda v: f"ID {int(v):02d}")
        table_df["Confidence"] = table_df["Confidence"].apply(format_percent)
        st.dataframe(table_df[["ID", "Age class", "Confidence"]], use_container_width=True, hide_index=True)
        return

    cols = st.columns(4)
    for i, row in enumerate(uniq_df.itertuples(index=False)):
        with cols[i % 4]:
            st.markdown(
                f"""
                <div class="ea-surface" style="padding:0.75rem; min-height:138px;">
                    <p style="margin:0; font-size:0.72rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.06em;">{getattr(row, 'age_label', 'Unknown')}</p>
                    <p style="margin:0.34rem 0 0 0; color:#f8fafc; font-size:1.08rem; font-weight:700;">ID {int(getattr(row, 'track_id', 0)):02d}</p>
                    <p style="margin:0.26rem 0 0 0; color:#cbd5e1; font-size:0.84rem;">Confidence: {format_percent(getattr(row, 'conf', 0.0))}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def result_quality_section():
    preview_section_header("Detection quality", "A clear quality snapshot from the selected run.")
    use_mock = st.session_state.get("use_mock", False)
    display_job_id = st.session_state.get("preview_display_job_id") or st.session_state.get("last_job_id")
    is_video = st.session_state.get("preview_is_video", True)

    if use_mock:
        st.info("Run one backend job to view quality analytics.")
        return
    if not display_job_id:
        st.info("No processed run selected.")
        return

    detections = fetch_job_detections(display_job_id) or []
    if not detections:
        st.info("No detection data available for quality analytics.")
        return

    df = pd.DataFrame(detections)
    if df.empty or "conf" not in df.columns:
        st.info("No confidence data available for quality analytics.")
        return

    df = df.copy()
    df["conf"] = pd.to_numeric(df["conf"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    if "cls_label" in df.columns:
        df["age_label"] = df["cls_label"].apply(normalize_age_class_label)
    else:
        df["age_label"] = "Unknown"
    if "track_id" in df.columns:
        df["track_id"] = pd.to_numeric(df["track_id"], errors="coerce")

    total_detections = int(len(df))
    unique_tracks = int(df["track_id"].nunique()) if "track_id" in df.columns else 0
    avg_conf = float(df["conf"].mean()) if total_detections else 0.0
    high_conf_share = float((df["conf"] >= 0.75).mean()) if total_detections else 0.0

    with card():
        left, right = st.columns([1.6, 1])
        with left:
            quality_level = (
                "Strong"
                if avg_conf >= 0.80
                else "Stable"
                if avg_conf >= 0.65
                else "Review needed"
            )
            st.markdown(
                f"""
                <div class="ea-surface" style="padding:0.9rem 0.95rem;">
                    <p style="margin:0; font-size:0.74rem; letter-spacing:0.08em; text-transform:uppercase; color:#94a3b8; font-weight:700;">Quality summary</p>
                    <p style="margin:0.3rem 0 0 0; font-size:1.06rem; font-weight:700; color:#f8fafc;">
                        {quality_level} detection confidence across this run.
                    </p>
                    <p style="margin:0.36rem 0 0 0; color:#cbd5e1; font-size:0.87rem;">
                        {total_detections:,} detections across {unique_tracks} IDs, with average confidence of {avg_conf * 100:.1f}%.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with right:
            s1, s2 = st.columns(2)
            s1.metric("Avg confidence", f"{avg_conf * 100:.1f}%")
            s2.metric("High confidence", f"{high_conf_share * 100:.1f}%")
        st.caption("Confidence indicates model certainty, not audited field accuracy.")

    with card():
        left, right = st.columns(2)
        with left:
            st.markdown("#### Detection confidence profile")
            conf_tier = np.where(
                df["conf"] >= 0.75,
                "High (>=75%)",
                np.where(df["conf"] >= 0.55, "Medium (55-74%)", "Low (<55%)"),
            )
            tier_df = (
                pd.DataFrame({"tier": conf_tier})
                .value_counts()
                .rename("count")
                .reset_index()
            )
            tier_df["share"] = tier_df["count"] / max(1, total_detections)
            conf_mix_chart = (
                alt.Chart(tier_df)
                .mark_bar(cornerRadiusEnd=6, color="#7f93ab")
                .encode(
                    x=alt.X("share:Q", axis=alt.Axis(title="Share", format="%")),
                    y=alt.Y("tier:N", axis=alt.Axis(title=None, labelAngle=0), sort=["High (>=75%)", "Medium (55-74%)", "Low (<55%)"]),
                    tooltip=[
                        alt.Tooltip("tier:N", title="Tier"),
                        alt.Tooltip("count:Q", title="Detections"),
                        alt.Tooltip("share:Q", title="Share", format=".1%"),
                    ],
                )
                .properties(height=260)
                .configure_axis(grid=False)
                .configure_view(strokeOpacity=0)
            )
            st.altair_chart(conf_mix_chart, use_container_width=True)

        with right:
            st.markdown("#### Observed age composition")
            class_df = (
                df.groupby("age_label", as_index=False)["conf"]
                .agg(["count", "mean"])
                .reset_index()
                .rename(columns={"count": "samples", "mean": "avg_conf"})
            )
            if not class_df.empty:
                class_df["share"] = class_df["samples"] / max(1, total_detections)
                class_df["label"] = class_df["age_label"] + " (" + (class_df["share"] * 100).round(0).astype(int).astype(str) + "%)"
                age_share_chart = (
                    alt.Chart(class_df)
                    .mark_bar(cornerRadiusEnd=6, color="#6f87a8")
                    .encode(
                        x=alt.X("share:Q", axis=alt.Axis(title="Share", format="%")),
                        y=alt.Y("age_label:N", axis=alt.Axis(title=None, labelAngle=0), sort="-x"),
                        tooltip=[
                            alt.Tooltip("age_label:N", title="Age class"),
                            alt.Tooltip("samples:Q", title="Samples"),
                            alt.Tooltip("share:Q", title="Share", format=".1%"),
                            alt.Tooltip("avg_conf:Q", title="Avg confidence", format=".2f"),
                        ],
                    )
                    .properties(height=260)
                    .configure_axis(grid=False)
                    .configure_view(strokeOpacity=0)
                )
                st.altair_chart(age_share_chart, use_container_width=True)
            else:
                st.info("No class-level confidence summary available.")

        summary_txt = (
            "Most detections are high-confidence."
            if high_conf_share >= 0.70
            else "Confidence is moderate; review edge cases."
            if high_conf_share >= 0.45
            else "Confidence is mixed; recommend manual review."
        )
        st.caption(summary_txt)

    if is_video and "frame_index" in df.columns:
        with card():
            st.markdown("#### Confidence trend")
            trend_df = (
                df.groupby("frame_index", as_index=False)
                .agg(
                    detections=("conf", "count"),
                    avg_conf=("conf", "mean"),
                )
                .sort_values("frame_index")
            )
            trend_df["avg_conf_pct"] = trend_df["avg_conf"] * 100.0
            trend_df["avg_conf_smooth"] = (
                trend_df["avg_conf_pct"]
                .rolling(window=max(5, min(35, len(trend_df) // 18 if len(trend_df) > 0 else 5)), center=True, min_periods=1)
                .mean()
            )
            max_points = 240
            if len(trend_df) > max_points:
                step = int(np.ceil(len(trend_df) / max_points))
                trend_df = trend_df.iloc[::step].reset_index(drop=True)

            area = (
                alt.Chart(trend_df)
                .mark_area(color="#1d4ed8", opacity=0.14)
                .encode(
                    x=alt.X("frame_index:Q", axis=alt.Axis(title="Frame", labelAngle=0)),
                    y=alt.Y("avg_conf_smooth:Q", axis=alt.Axis(title="Confidence (%)", labelAngle=0)),
                )
            )
            smooth_line = (
                alt.Chart(trend_df)
                .mark_line(color="#38bdf8", strokeWidth=2.6)
                .encode(
                    x=alt.X("frame_index:Q", axis=alt.Axis(title="Frame", labelAngle=0)),
                    y=alt.Y("avg_conf_smooth:Q", axis=alt.Axis(title="Confidence (%)", labelAngle=0)),
                    tooltip=[
                        alt.Tooltip("frame_index:Q", title="Frame"),
                        alt.Tooltip("avg_conf_smooth:Q", title="Smoothed confidence (%)", format=".1f"),
                        alt.Tooltip("detections:Q", title="Detections"),
                    ],
                )
            )
            raw_line = (
                alt.Chart(trend_df)
                .mark_line(color="#93c5fd", opacity=0.35, strokeWidth=1.2)
                .encode(
                    x=alt.X("frame_index:Q", axis=alt.Axis(title="Frame", labelAngle=0)),
                    y=alt.Y("avg_conf_pct:Q", axis=alt.Axis(title="Confidence (%)", labelAngle=0)),
                )
            )
            st.altair_chart((area + raw_line + smooth_line).properties(height=280), use_container_width=True)


def charts_section():
    if not st.session_state.get("preview_is_video", True):
        return

    use_mock = st.session_state.get("use_mock", False)
    if use_mock:
        herd_df = mock_herd_series()
    else:
        job_id = st.session_state.get("preview_display_job_id") or st.session_state.get("last_job_id")
        metrics = fetch_job_metrics(job_id) if job_id else None
        series = metrics.get("herd_series") if metrics else None
        herd_df = pd.DataFrame(series or [])
    with card():
        header_cols = st.columns([3, 1])
        with header_cols[0]:
            st.markdown("#### Herd trend")
        with header_cols[1]:
            if use_mock:
                herd_mode = "Time"
            else:
                options = ["Frame"]
                if "time_seconds" in herd_df.columns:
                    options = ["Time", "Frame"]
                job_id = st.session_state.get("preview_display_job_id") or st.session_state.get("last_job_id") or "none"
                herd_mode = st.radio(
                    "X axis",
                    options,
                    horizontal=True,
                    label_visibility="collapsed",
                    key=f"herd_x_mode_{job_id}",
                )

        st.caption("Smoothed trend from video frames.")
        if herd_df.empty:
            st.info("No data yet. Enable mock data or process a video.")
            return

        herd_df = herd_df.copy()
        if use_mock:
            herd_df["frame_index"] = np.arange(len(herd_df))
        herd_df = herd_df.sort_values("frame_index").reset_index(drop=True)
        window = max(5, min(25, len(herd_df) // 20 if len(herd_df) > 0 else 5))
        herd_df["herd_smooth"] = herd_df["herd_size"].rolling(window=window, center=True, min_periods=1).mean()
        if "time_seconds" in herd_df.columns:
            herd_df["time_label"] = herd_df["time_seconds"].apply(format_seconds)

        max_points = 160
        if len(herd_df) > max_points:
            step = int(np.ceil(len(herd_df) / max_points))
            chart_df = herd_df.iloc[::step].reset_index(drop=True)
        else:
            chart_df = herd_df

        if use_mock:
            chart_df["time_label"] = chart_df["time"]
            herd_mode = "Time"

        if herd_mode == "Time" and "time_label" in chart_df.columns:
            x_def = alt.X("time_label:N", axis=alt.Axis(title="Time (mm:ss)", labelAngle=0))
            tooltips = [
                alt.Tooltip("time_label:N", title="Time"),
                alt.Tooltip("frame_index:Q", title="Frame"),
                alt.Tooltip("herd_size:Q", title="Raw herd"),
                alt.Tooltip("herd_smooth:Q", title="Smoothed", format=".2f"),
            ]
        else:
            x_def = alt.X("frame_index:Q", axis=alt.Axis(title="Frame", labelAngle=0))
            tooltips = [
                alt.Tooltip("frame_index:Q", title="Frame"),
                alt.Tooltip("time_seconds:Q", title="Time (s)", format=".1f"),
                alt.Tooltip("herd_size:Q", title="Raw herd"),
                alt.Tooltip("herd_smooth:Q", title="Smoothed", format=".2f"),
            ]

        raw_line = (
            alt.Chart(chart_df)
            .mark_line(opacity=0.35, color="#9db7da")
            .encode(x=x_def, y=alt.Y("herd_size:Q", axis=alt.Axis(title=None, labelAngle=0)))
        )
        smooth_line = (
            alt.Chart(chart_df)
            .mark_line(color="#e6f0ff", strokeWidth=2.6)
            .encode(x=x_def, y=alt.Y("herd_smooth:Q", axis=alt.Axis(title=None, labelAngle=0)), tooltip=tooltips)
        )
        st.altair_chart((raw_line + smooth_line).properties(height=320).configure_view(strokeOpacity=0), use_container_width=True)

        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("Min herd", int(herd_df["herd_size"].min()))
        with k2:
            st.metric("Avg herd", f"{float(herd_df['herd_size'].mean()):.2f}")
        with k3:
            st.metric("Peak herd", int(herd_df["herd_size"].max()))

        compact_df = (
            herd_df.groupby("herd_size", as_index=False)
            .size()
            .rename(columns={"size": "Samples", "herd_size": "Herd size"})
            .sort_values("Herd size")
        )
        st.markdown("##### Herd size distribution")
        st.dataframe(compact_df, use_container_width=True, hide_index=True)

def timeline_and_habitat_section():
    with card():
        tab_timeline, tab_habitat = st.tabs(["Individual timeline", "Habitat summary"])
        use_mock = st.session_state.get("use_mock", False)
        df = empty_timeline_table()
        if use_mock:
            df = mock_timeline_table()
        else:
            job_id = st.session_state.get("preview_display_job_id") or st.session_state.get("last_job_id")
            if job_id:
                try:
                    resp = requests.get(f"{BACKEND_BASE_URL}/timeline/{job_id}", timeout=10)
                    resp.raise_for_status()
                    rows = resp.json().get("timeline", [])
                    if rows:
                        df = pd.DataFrame(rows)
                        df = df.rename(
                            columns={
                                "elephant_id": "Elephant ID",
                                "age_class": "Age class",
                                "first_seen": "First seen (frame)",
                                "last_seen": "Last seen (frame)",
                                "duration_frames": "Duration (frames)",
                            }
                        )
                except Exception:
                    df = empty_timeline_table()

        with tab_timeline:
            age_filter = st.selectbox("Filter by age class", ["All", "Adult", "Juvenile", "Baby"], index=0)
            df_show = df.copy()
            if "Age class" in df_show.columns and not df_show.empty:
                df_show["Age class"] = df_show["Age class"].apply(normalize_age_class_label)

            if age_filter != "All" and "Age class" in df_show.columns:
                df_show = df_show[df_show["Age class"] == age_filter]

            if not df_show.empty and "Duration (frames)" in df_show.columns:
                df_show = df_show.sort_values("Duration (frames)", ascending=False).reset_index(drop=True)

            if df_show.empty:
                st.info("No timeline rows for the selected age class.")
            else:
                total_rows = len(df_show)
                st.caption(f"{total_rows} rows")
                c1, c2 = st.columns([2, 1])
                with c1:
                    search_txt = st.text_input("Search Elephant ID", value="", key="timeline_search_elephant_id")
                with c2:
                    page_size = st.selectbox("Rows per page", [10, 20, 50, 100], index=1, key="timeline_page_size")

                filtered = df_show
                if search_txt.strip():
                    needle = search_txt.strip().lower()
                    filtered = filtered[
                        filtered["Elephant ID"].astype(str).str.lower().str.contains(needle)
                    ]
                if filtered.empty:
                    st.info("No matches for this search.")
                else:
                    pages = int(np.ceil(len(filtered) / page_size))
                    page = st.number_input(
                        "Page",
                        min_value=1,
                        max_value=max(1, pages),
                        value=1,
                        step=1,
                        key="timeline_page_no",
                    )
                    start = (int(page) - 1) * page_size
                    end = start + page_size
                    st.dataframe(filtered.iloc[start:end], use_container_width=True, hide_index=True)

        with tab_habitat:
            st.markdown("#### Habitat")
            if not use_mock:
                job_id = st.session_state.get("preview_display_job_id") or st.session_state.get("last_job_id")
                metrics = fetch_job_metrics(job_id) if job_id else None
                habitat_distribution = metrics.get("habitat_distribution", []) if metrics else []
                habitat_timeline = metrics.get("habitat_timeline", []) if metrics else []

                dominant_habitat = metrics.get("dominant_habitat", "-") if metrics else "-"
                dominant_share = float(metrics.get("dominant_habitat_share", 0.0)) if metrics else 0.0
                habitat_switches = int(metrics.get("habitat_switches", 0)) if metrics else 0
                habitat_coverage = float(metrics.get("habitat_coverage", 0.0)) if metrics else 0.0

                k1, k2, k3, k4 = st.columns(4)
                with k1:
                    st.metric("Dominant habitat", dominant_habitat)
                with k2:
                    st.metric("Time in dominant", f"{dominant_share:.0f}%")
                with k3:
                    st.metric("Habitat switches", habitat_switches)
                with k4:
                    st.metric("Coverage", f"{habitat_coverage:.0f}%")

                if not habitat_distribution or not habitat_timeline:
                    st.info(
                        "No habitat output found for this job yet. "
                        "Process a video after placing a habitat classifier weight "
                        "(or set HABITAT_MODEL_PATH)."
                    )
                else:
                    habitat_series = pd.Series(
                        {row["habitat"]: row["percentage"] for row in habitat_distribution},
                        name="percentage",
                    )
                    color_palette = [
                        "#2d6a4f", "#95d5b2", "#bc6c25", "#ddb892",
                        "#4d908e", "#577590", "#f8961e", "#43aa8b",
                    ]
                    habitat_labels = list(habitat_series.index)
                    habitat_colors = alt.Scale(
                        domain=habitat_labels,
                        range=color_palette[: max(1, len(habitat_labels))],
                    )

                    left, right = st.columns([1.35, 1])
                    with left:
                        st.markdown("##### Habitat timeline")
                        timeline_df = pd.DataFrame(habitat_timeline)
                        timeline_chart = (
                            alt.Chart(timeline_df)
                            .mark_bar(cornerRadius=4)
                            .encode(
                                x=alt.X("start:Q", axis=alt.Axis(title="Frame index", labelAngle=0)),
                                x2="end:Q",
                                y=alt.Y("habitat:N", axis=alt.Axis(title=None, labelAngle=0), sort=habitat_labels),
                                color=alt.Color("habitat:N", scale=habitat_colors, legend=None),
                                tooltip=[
                                    alt.Tooltip("habitat:N", title="Habitat"),
                                    alt.Tooltip("start:Q", title="Start frame"),
                                    alt.Tooltip("end:Q", title="End frame"),
                                    alt.Tooltip("confidence:Q", title="Confidence", format=".2f"),
                                    alt.Tooltip("samples:Q", title="Samples"),
                                ],
                            )
                            .properties(height=220)
                        )
                        st.altair_chart(timeline_chart, use_container_width=True)

                    with right:
                        st.markdown("##### Time spent by habitat")
                        distribution_df = (
                            pd.DataFrame(habitat_distribution)
                            .rename(columns={"habitat": "habitat", "percentage": "percentage"})
                            .sort_values("percentage", ascending=True)
                        )
                        distribution_chart = (
                            alt.Chart(distribution_df)
                            .mark_bar(cornerRadiusEnd=6)
                            .encode(
                                x=alt.X("percentage:Q", axis=alt.Axis(title="Percentage", labelAngle=0)),
                                y=alt.Y("habitat:N", axis=alt.Axis(title=None, labelAngle=0)),
                                color=alt.Color("habitat:N", scale=habitat_colors, legend=None),
                                tooltip=[
                                    alt.Tooltip("habitat:N", title="Habitat"),
                                    alt.Tooltip("percentage:Q", title="Time (%)", format=".0f"),
                                    alt.Tooltip("avg_confidence:Q", title="Avg confidence", format=".2f"),
                                ],
                            )
                            .properties(height=220)
                        )
                        st.altair_chart(distribution_chart, use_container_width=True)
            else:
                habitat_series = pd.Series(
                    {"Grassland": 48, "Oil Palm Plantation": 32, "Field Edge": 14, "Forest": 6},
                    name="percentage",
                )
                dominant_habitat = habitat_series.idxmax()
                dominant_share = float(habitat_series.max())

                k1, k2, k3, k4 = st.columns(4)
                with k1:
                    st.metric("Dominant habitat", dominant_habitat)
                with k2:
                    st.metric("Time in dominant", f"{dominant_share:.0f}%")
                with k3:
                    st.metric("Habitat switches", "6")
                with k4:
                    st.metric("Coverage", "100%")

                habitat_colors = alt.Scale(
                    domain=["Forest", "Grassland", "Oil Palm Plantation", "Field Edge"],
                    range=["#2d6a4f", "#95d5b2", "#bc6c25", "#ddb892"],
                )

                left, right = st.columns([1.35, 1])
                with left:
                    st.markdown("##### Habitat timeline")
                    timeline_df = pd.DataFrame(
                        [
                            (0, 30, "Grassland", 0.86),
                            (31, 55, "Oil Palm Plantation", 0.79),
                            (56, 70, "Field Edge", 0.72),
                            (71, 95, "Grassland", 0.88),
                            (96, 110, "Forest", 0.67),
                            (111, 140, "Oil Palm Plantation", 0.83),
                        ],
                        columns=["start", "end", "habitat", "confidence"],
                    )
                    timeline_chart = (
                        alt.Chart(timeline_df)
                        .mark_bar(cornerRadius=4)
                        .encode(
                            x=alt.X("start:Q", axis=alt.Axis(title="Frame index", labelAngle=0)),
                            x2="end:Q",
                            y=alt.Y("habitat:N", axis=alt.Axis(title=None, labelAngle=0), sort=list(habitat_series.index)),
                            color=alt.Color("habitat:N", scale=habitat_colors, legend=None),
                            tooltip=[
                                alt.Tooltip("habitat:N", title="Habitat"),
                                alt.Tooltip("start:Q", title="Start frame"),
                                alt.Tooltip("end:Q", title="End frame"),
                                alt.Tooltip("confidence:Q", title="Confidence", format=".2f"),
                            ],
                        )
                        .properties(height=220)
                    )
                    st.altair_chart(timeline_chart, use_container_width=True)

                with right:
                    st.markdown("##### Time spent by habitat")
                    distribution_df = (
                        habitat_series.sort_values(ascending=True)
                        .rename_axis("habitat")
                        .reset_index(name="percentage")
                    )
                    distribution_chart = (
                        alt.Chart(distribution_df)
                        .mark_bar(cornerRadiusEnd=6)
                        .encode(
                            x=alt.X("percentage:Q", axis=alt.Axis(title="Percentage", labelAngle=0)),
                            y=alt.Y("habitat:N", axis=alt.Axis(title=None, labelAngle=0)),
                            color=alt.Color("habitat:N", scale=habitat_colors, legend=None),
                            tooltip=[
                                alt.Tooltip("habitat:N", title="Habitat"),
                                alt.Tooltip("percentage:Q", title="Time (%)", format=".0f"),
                            ],
                        )
                        .properties(height=220)
                    )
                    st.altair_chart(distribution_chart, use_container_width=True)

                st.markdown("##### Evidence snapshots")
                e1, e2, e3 = st.columns(3)
                with e1:
                    st.caption("Frame 24")
                    st.write("Habitat: **Grassland**")
                    st.write("Confidence: **0.88**")
                with e2:
                    st.caption("Frame 92")
                    st.write("Habitat: **Forest**")
                    st.write("Confidence: **0.67**")
                with e3:
                    st.caption("Frame 123")
                    st.write("Habitat: **Oil Palm Plantation**")
                    st.write("Confidence: **0.83**")
# ===================== FOLIUM MAP HELPERS =====================

def build_folium_map(points_df: pd.DataFrame, track_df: pd.DataFrame | None = None):
    """
    points_df: columns [lat, lon] (detections points)
    track_df (optional): columns [lat, lon] in time order (route/track)
    """
    if points_df.empty:
        center = [0, 0]
        zoom = 2
    else:
        center = [float(points_df["lat"].mean()), float(points_df["lon"].mean())]
        zoom = 13

    tile_url = resolve_tile_server_url()
    if tile_url:
        m = folium.Map(
            location=center,
            zoom_start=zoom,
            tiles=tile_url,
            attr="Local tiles",
            control_scale=True,
        )
    else:
        # Offline map: no online tiles to avoid blinking/refresh issues.
        m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True)

    # Detections as circle markers
    for i, row in points_df.reset_index(drop=True).iterrows():
        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lon"])],
            radius=6,
            weight=1,
            color="#ff6b6b",
            fill=True,
            fill_opacity=0.6,
            popup=f"Detection {i+1}",
        ).add_to(m)

    # Optional route polyline
    if track_df is not None and not track_df.empty:
        coords = track_df[["lat", "lon"]].astype(float).values.tolist()
        folium.PolyLine(coords, color="#4dabf7", weight=4, opacity=0.9, tooltip="Track").add_to(m)

        # start/end markers
        folium.Marker(coords[0], tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
        folium.Marker(coords[-1], tooltip="End", icon=folium.Icon(color="red")).add_to(m)

    return m

def map_section():
    use_mock = st.session_state.get("use_mock", False)
    preview_is_video = st.session_state.get("preview_is_video", True)
    job_id = st.session_state.get("preview_display_job_id") or st.session_state.get("last_job_id")
    points_df = None
    if job_id:
        points_df = cached_gps_points(job_id)
    if points_df is not None:
        points_df = downsample_points(points_df, max_points=200)

    route_km = estimate_route_distance_km(points_df.copy() if points_df is not None and not points_df.empty else None)
    with card():
        st.markdown("#### Detection map")
        st.caption("GPS points from the selected run.")
        st.caption("This map shows drone movement and estimated drone route distance, not elephant travel distance.")

        if points_df is None:
            if use_mock:
                points_df = pd.DataFrame(
                    {
                        "lat": [3.1390, 3.1398, 3.1384, 3.1406, 1.3521],
                        "lon": [101.6869, 101.6882, 101.6855, 101.6848, 103.8198],
                    }
                )
            else:
                points_df = pd.DataFrame(columns=["lat", "lon"])
        else:
            st.caption("Showing GPS track from the latest processed video.")

        track_df = points_df.copy() if not points_df.empty else None

        map_center_lat = None
        map_center_lon = None
        if points_df is not None and not points_df.empty:
            lat_series = pd.to_numeric(points_df["lat"], errors="coerce").dropna()
            lon_series = pd.to_numeric(points_df["lon"], errors="coerce").dropna()
            if not lat_series.empty and not lon_series.empty:
                map_center_lat = float(lat_series.mean())
                map_center_lon = float(lon_series.mean())
        drone_distance_txt = f"{route_km:.2f} km" if (preview_is_video and route_km > 0) else "-"

        st.markdown(
            f"""
            <div class="ea-surface" style="padding:0.75rem 0.85rem; margin:0.2rem 0 0.55rem 0;">
                <p style="margin:0; font-size:0.74rem; letter-spacing:0.06em; text-transform:uppercase; color:#94a3b8; font-weight:700;">Location center</p>
                <div style="display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:0.7rem; margin-top:0.35rem;">
                    <div>
                        <p style="margin:0; color:#94a3b8; font-size:0.78rem;">Latitude</p>
                        <p style="margin:0.08rem 0 0 0; color:#93c5fd; font-size:1.55rem; font-weight:780; line-height:1.05;">{f"{map_center_lat:.5f}" if map_center_lat is not None else "-"}</p>
                    </div>
                    <div>
                        <p style="margin:0; color:#94a3b8; font-size:0.78rem;">Longitude</p>
                        <p style="margin:0.08rem 0 0 0; color:#93c5fd; font-size:1.55rem; font-weight:780; line-height:1.05;">{f"{map_center_lon:.5f}" if map_center_lon is not None else "-"}</p>
                    </div>
                    <div>
                        <p style="margin:0; color:#94a3b8; font-size:0.78rem;">Drone distance</p>
                        <p style="margin:0.08rem 0 0 0; color:#93c5fd; font-size:1.55rem; font-weight:780; line-height:1.05;">{drone_distance_txt}</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        m = build_folium_map(points_df, track_df=track_df)
        map_key = f"map-{job_id}" if job_id else "map-demo"
        st_folium(m, height=420, use_container_width=True, key=map_key)
        tile_url = resolve_tile_server_url()
        if tile_url:
            st.caption(f"Basemap source: {tile_url}")
        else:
            st.caption("Basemap source: offline mode (no tile server connected).")

def upload_jobs_page():
    st.markdown(
        """
        <style>
        .ea-model-card {
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 12px;
            padding: 0.7rem 0.78rem;
            margin-bottom: 0.45rem;
            background: rgba(2, 6, 23, 0.20);
        }
        .ea-model-card strong {
            color: #f8fafc;
            font-size: 0.94rem;
        }
        .ea-model-card p {
            margin: 0.18rem 0 0 0;
            color: #cbd5e1;
            font-size: 0.84rem;
        }
        .ea-model-reco {
            border-color: rgba(56, 189, 248, 0.55);
            border-left: 4px solid rgba(45, 212, 191, 0.9);
            box-shadow: 0 10px 18px rgba(2, 132, 199, 0.14);
            background: linear-gradient(90deg, rgba(45,212,191,0.08), rgba(2,6,23,0.16));
        }
        .ea-reco-pill {
            display: inline-block;
            margin-left: 0.45rem;
            padding: 0.1rem 0.45rem;
            border-radius: 999px;
            font-size: 0.68rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            font-weight: 700;
            color: #ecfeff;
            background: rgba(14, 116, 144, 0.72);
            border: 1px solid rgba(103, 232, 249, 0.5);
            vertical-align: middle;
        }

        /* Calm, professional focus color for form controls */
        [data-testid="stTextInput"] input:focus,
        [data-testid="stTextInput"] input:focus-visible {
            border-color: rgba(56, 189, 248, 0.75) !important;
            box-shadow: 0 0 0 1px rgba(56, 189, 248, 0.25) !important;
        }
        [data-testid="stSelectbox"] [data-baseweb="select"] > div {
            border-color: rgba(148, 163, 184, 0.35) !important;
        }
        [data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within {
            border-color: rgba(56, 189, 248, 0.75) !important;
            box-shadow: 0 0 0 1px rgba(56, 189, 248, 0.22) !important;
        }
        .ea-status-pill {
            display: inline-block;
            padding: 0.14rem 0.55rem;
            border-radius: 999px;
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.03em;
        }
        .ea-status-complete { background:#dcfce7; color:#166534; }
        .ea-status-processing { background:#dbeafe; color:#1d4ed8; }
        .ea-status-queued { background:#fef3c7; color:#92400e; }
        .ea-status-failed { background:#fee2e2; color:#b91c1c; }
        .ea-run-card {
            border: 1px solid rgba(148,163,184,0.24);
            border-radius: 12px;
            padding: 0.65rem 0.75rem;
            margin-bottom: 0.45rem;
            background: linear-gradient(180deg, rgba(2,6,23,0.26), rgba(2,6,23,0.15));
        }
        .ea-run-title {
            margin: 0;
            font-size: 0.95rem;
            font-weight: 700;
            color: #f8fafc;
        }
        .ea-run-meta {
            margin: 0.22rem 0 0 0;
            font-size: 0.8rem;
            color: #94a3b8;
        }
        .ea-detail-card {
            border: 1px solid rgba(148,163,184,0.24);
            border-radius: 12px;
            padding: 0.75rem 0.8rem;
            background: rgba(2,6,23,0.2);
        }
        .ea-detail-line {
            margin: 0 0 0.35rem 0;
            font-size: 0.88rem;
            color: #cbd5e1;
        }
        .ea-detail-list {
            border: 1px solid rgba(148,163,184,0.24);
            border-radius: 12px;
            padding: 0.7rem 0.78rem;
            background: rgba(2,6,23,0.2);
        }
        .ea-detail-item {
            margin: 0 0 0.35rem 0;
            font-size: 0.9rem;
            color: #cbd5e1;
        }
        .ea-detail-item strong {
            color: #f8fafc;
            min-width: 76px;
            display: inline-block;
        }
        .ea-admin-wrap {
            border: 1px solid rgba(248, 113, 113, 0.35);
            border-radius: 12px;
            padding: 0.72rem 0.82rem;
            background: linear-gradient(180deg, rgba(127, 29, 29, 0.2), rgba(2, 6, 23, 0.2));
        }
        .ea-admin-note {
            margin: 0;
            color: #fecaca;
            font-size: 0.82rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    render_page_hero(
        "Start Analysis",
        "Upload field footage, choose a model profile, and generate report-ready outputs in one run.",
        kicker="Upload",
    )

    with card():
        st.subheader("Upload Workspace")

        col_drop, col_meta = st.columns([1.2, 1])
        with col_drop:
            st.markdown("**1. Upload media**")
            if "upload_uploader_rev" not in st.session_state:
                st.session_state["upload_uploader_rev"] = 0
            if "upload_locked_file" not in st.session_state:
                st.session_state["upload_locked_file"] = None
            if "upload_locked" not in st.session_state:
                st.session_state["upload_locked"] = False
            clear = False
            start = False
            uploaded_file = None
            uploader_col, reset_col = st.columns([5.2, 1.1])
            with uploader_col:
                if st.session_state["upload_locked"] and st.session_state["upload_locked_file"]:
                    locked = st.session_state["upload_locked_file"]
                    st.markdown(
                        f"""
                        <div class="ea-model-card" style="margin-top:0.2rem;">
                            <strong>Selected file (locked)</strong>
                            <p>{locked.get('name','-')} | {locked.get('size_mb','-')} MB</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    uploaded_files = [
                        {
                            "name": locked.get("name"),
                            "type": locked.get("type") or "video/mp4",
                            "getvalue": lambda b=locked.get("bytes", b""): b,
                        }
                    ]
                else:
                    uploaded_file = st.file_uploader(
                        "Drag and drop media or click to browse",
                        type=["mp4", "mov", "avi", "jpg", "jpeg", "png"],
                        key=f"upload_single_{st.session_state['upload_uploader_rev']}",
                    )
                    uploaded_files = [uploaded_file] if uploaded_file is not None else []
                    if uploaded_file is not None:
                        st.session_state["upload_locked"] = True
                        st.session_state["upload_locked_file"] = {
                            "name": uploaded_file.name,
                            "type": uploaded_file.type,
                            "bytes": uploaded_file.getvalue(),
                            "size_mb": round(len(uploaded_file.getvalue()) / (1024 * 1024), 2),
                        }
                        try:
                            st.rerun()
                        except Exception:
                            st.experimental_rerun()
            with reset_col:
                st.markdown("<div style='height: 2.0rem;'></div>", unsafe_allow_html=True)
                clear = st.button(
                    "Reset",
                    use_container_width=True,
                    help="Clear selected files and reselect",
                    disabled=not st.session_state["upload_locked"],
                    key="upload_reset_top",
                )

            selected_count_raw = 1 if st.session_state["upload_locked"] else len(uploaded_files)
            if st.session_state["upload_locked"]:
                st.info("1 file selected. Upload is locked. Click **Reset** to choose another file.")

            st.caption(
                f"Supported: MP4, MOV, AVI, JPG, JPEG, PNG | Max {MAX_VIDEO_MB} MB | "
                f"Selected: {selected_count_raw} / 1 file"
            )
            st.markdown("")
            start = st.button("Run Analysis", use_container_width=True, type="primary")
            st.caption(
                "Typical run time depends on footage length and resolution. Recommended input: 1080p, ~30fps."
            )

        with col_meta:
            st.markdown("**2. Configure run**")
            gps_input = st.text_input("Optional GPS coordinates", "")
            model_choice = st.selectbox("Detection profile", list(VISIBLE_MODEL_OPTIONS.keys()), index=0)
            mode_labels = list(PIPELINE_MODE_OPTIONS.keys())
            fallback_label = mode_labels[0] if mode_labels else "Identity Accurate (ByteTrack)"
            default_label = DEFAULT_PIPELINE_MODE_LABEL if DEFAULT_PIPELINE_MODE_LABEL in PIPELINE_MODE_OPTIONS else fallback_label
            if SHOW_PIPELINE_MODE_CONTROL:
                pipeline_mode_choice = st.selectbox(
                    "Pipeline mode",
                    mode_labels,
                    index=mode_labels.index(default_label) if default_label in mode_labels else 0,
                    help="Fast Trend uses sparse sampling without ByteTrack for speed-focused trends. Identity Accurate uses ByteTrack for stronger ID continuity and final reporting.",
                )
            else:
                pipeline_mode_choice = default_label
            selected_mode_key = PIPELINE_MODE_OPTIONS[pipeline_mode_choice]
            if SHOW_PIPELINE_MODE_CONTROL:
                if selected_mode_key == "fast_trend":
                    st.caption("Runs full tracking logic at 2 FPS. Best for quick trend review, not final unique-ID reporting.")
                elif selected_mode_key == "value_1x":
                    st.caption("High value mode: tracked ~5 FPS for near 1:1 processing on most local machines.")
                elif selected_mode_key == "balanced_2fps":
                    st.caption("Uses light ByteTrack with 2 FPS sampling. Good compromise between speed and ID continuity.")
                else:
                    st.caption("Best for final client report where ID continuity and count confidence matter.")

            st.markdown("**Model guide**")
            st.markdown(
                """
                <div class="ea-model-card ea-model-reco">
                    <strong>Research Model v2_s</strong><span class="ea-reco-pill">Recommended</span>
                    <p>Best all-round option for production-style analysis.</p>
                </div>
                <div class="ea-model-card">
                    <strong>Online Baseline (Roboflow v1)</strong>
                    <p>Use as an external benchmark for side-by-side comparison.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            use_mock = st.checkbox("Use mock data for demo", value=st.session_state.get("use_mock", False))
            st.session_state["use_mock"] = use_mock

        if clear:
            st.session_state["upload_locked"] = False
            st.session_state["upload_locked_file"] = None
            st.session_state["upload_uploader_rev"] += 1
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()

        if start:
            if (not uploaded_files) and not use_mock:
                st.error("Please upload at least one file or enable mock data.")
            else:
                with st.spinner("Processing video..."):
                    time.sleep(1.5)
                    if use_mock:
                        mock_items = []
                        if uploaded_files:
                            for f in uploaded_files:
                                mock_items.append(
                                    {
                                        "file_name": f.name,
                                        "summary": mock_detection_summary(),
                                    }
                                )
                        else:
                            mock_items.append(
                                {
                                    "file_name": "demo_video.mp4",
                                    "summary": mock_detection_summary(),
                                }
                            )
                        result = {
                            "mode": "mock",
                            "processed_files": len(mock_items),
                            "items": mock_items,
                        }
                    else:
                        processed_items = []
                        failed_items = []
                        total = len(uploaded_files or [])
                        progress = st.progress(0.0, text="Starting batch processing...")
                        for idx, uploaded_file in enumerate(uploaded_files or [], start=1):
                            try:
                                if isinstance(uploaded_file, dict):
                                    file_name = uploaded_file.get("name", f"upload_{idx}")
                                    file_type = uploaded_file.get("type") or "video/mp4"
                                    file_bytes = uploaded_file.get("getvalue", lambda: b"")()
                                else:
                                    file_name = uploaded_file.name
                                    file_type = uploaded_file.type or "video/mp4"
                                    file_bytes = uploaded_file.getvalue()
                                files = {
                                    "file": (
                                        file_name,
                                        file_bytes,
                                        file_type,
                                    )
                                }
                                upload_resp = requests.post(
                                    f"{BACKEND_BASE_URL}/upload",
                                    files=files,
                                    timeout=120,
                                )
                                upload_resp.raise_for_status()
                                job_id = upload_resp.json()["job_id"]
                                st.session_state["last_job_id"] = job_id

                                process_resp = requests.post(
                                    f"{BACKEND_BASE_URL}/process/{job_id}",
                                    params={
                                        "model_key": VISIBLE_MODEL_OPTIONS[model_choice],
                                        "pipeline_mode": PIPELINE_MODE_OPTIONS[pipeline_mode_choice],
                                    },
                                    timeout=600,
                                )
                                process_resp.raise_for_status()
                                process_data = process_resp.json()
                                processed_items.append(
                                    {
                                        "file_name": file_name,
                                        "job_id": job_id,
                                        "result": process_data,
                                    }
                                )

                                results_resp = requests.get(
                                    f"{BACKEND_BASE_URL}/results/{job_id}",
                                    timeout=60,
                                )
                                if results_resp.ok:
                                    st.session_state["last_results_payload"] = results_resp.json()
                            except Exception as exc:
                                fail_name = uploaded_file.get("name", f"upload_{idx}") if isinstance(uploaded_file, dict) else uploaded_file.name
                                failed_items.append({"file_name": fail_name, "error": str(exc)})

                            progress.progress(
                                idx / max(total, 1),
                                text=f"Processed {idx}/{total} file(s)",
                            )

                        result = {
                            "mode": "batch",
                            "processed_files": len(processed_items),
                            "failed_files": len(failed_items),
                            "items": processed_items,
                            "errors": failed_items,
                        }
                        if not processed_items:
                            st.error("All files failed to process. Please check backend logs and file formats.")
                            return

                st.success(f"Analysis complete. {result.get('processed_files', 0)} file(s) processed.")
                if result.get("failed_files", 0):
                    st.warning(f"{result['failed_files']} file(s) failed. Please retry those files.")
                st.session_state["last_result"] = result
                st.session_state["last_model_choice"] = model_choice
                st.session_state["last_pipeline_mode_choice"] = pipeline_mode_choice
                st.session_state["last_pipeline_mode_key"] = PIPELINE_MODE_OPTIONS[pipeline_mode_choice]
                st.session_state["pending_page_nav"] = "Overview"
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()

    st.markdown("#### Analysis history")
    jobs_df = fetch_jobs_from_backend() if not use_mock else None
    job_id_to_label: dict[str, str] = {}
    label_to_job_id: dict[str, str] = {}

    if jobs_df is not None and not jobs_df.empty and "Job ID" in jobs_df.columns:
        for _, row in jobs_df.iterrows():
            jid = str(row.get("Job ID", ""))
            if not jid:
                continue
            vname = str(row.get("Video name", "Unknown video"))
            submitted = format_my_time(row.get("Submitted at", ""))
            short_id = jid[:8]
            label = f"{vname} | {submitted} | #{short_id}"
            job_id_to_label[jid] = label
            label_to_job_id[label] = jid

    labels_for_admin = list(label_to_job_id.keys()) if not use_mock else []

    if jobs_df is None:
        jobs_df = mock_jobs_table() if use_mock else pd.DataFrame(
            columns=["Job ID", "Video name", "Status", "Submitted at", "Duration"]
        )
    else:
        jobs_df["Submitted at"] = jobs_df["Submitted at"].apply(format_my_time)
        jobs_df["Duration (s)"] = jobs_df["Duration (s)"].apply(format_duration)
        jobs_df = jobs_df.rename(columns={"Duration (s)": "Duration"})

    with card():
        st.markdown('<div class="ea-block-head">Recent runs</div>', unsafe_allow_html=True)
        preview_rows = pd.DataFrame()
        if jobs_df is not None and not jobs_df.empty:
            preview_rows = jobs_df.copy()
            if "Submitted at" in preview_rows.columns:
                preview_rows["_submitted_sort"] = pd.to_datetime(
                    preview_rows["Submitted at"], errors="coerce"
                )
                preview_rows = preview_rows.sort_values(
                    "_submitted_sort", ascending=False, na_position="last"
                ).drop(columns=["_submitted_sort"])
            preview_rows = preview_rows.head(5)
        if preview_rows is None or preview_rows.empty:
            st.caption("No runs yet.")
        else:
            for _, r in preview_rows.iterrows():
                status_txt = str(r.get("Status", "")).lower()
                status_cls = {
                    "complete": "ea-status-complete",
                    "processing": "ea-status-processing",
                    "queued": "ea-status-queued",
                    "failed": "ea-status-failed",
                }.get(status_txt, "ea-status-queued")
                st.markdown(
                    f"""
                    <div class="ea-run-card">
                        <p class="ea-run-title">{r.get("Video name","-")} <span class="ea-status-pill {status_cls}">{status_txt or '-'}</span></p>
                        <p class="ea-run-meta">Submitted: {r.get("Submitted at","-")} | Duration: {r.get("Duration","-")} | Job: {str(r.get("Job ID",""))[:8]}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    if not use_mock and jobs_df is not None and not jobs_df.empty:
        with card():
            st.markdown('<div class="ea-block-head">History view</div>', unsafe_allow_html=True)
            job_ids = jobs_df["Job ID"].dropna().astype(str).tolist() if "Job ID" in jobs_df.columns else []
            if not job_ids:
                st.info("No history available.")
            else:
                default_job = st.session_state.get("last_job_id")
                if default_job not in job_ids:
                    default_job = job_ids[0]
                explorer_labels = [job_id_to_label.get(j, j) for j in job_ids]
                default_label = job_id_to_label.get(default_job, default_job)
                selector_col, action_col = st.columns([3.6, 1.2])
                with selector_col:
                    selected_label = st.selectbox(
                        "Select a history record",
                        options=explorer_labels,
                        index=explorer_labels.index(default_label) if default_label in explorer_labels else 0,
                        key="history_explorer_job_label",
                    )
                    selected_job = label_to_job_id.get(selected_label, default_job)
                with action_col:
                    st.markdown("<div style='height:1.85rem;'></div>", unsafe_allow_html=True)
                    if st.button(
                        "Open in Overview",
                        use_container_width=True,
                        key=f"history_open_preview_top_{selected_job}",
                    ):
                        st.session_state["last_job_id"] = selected_job
                        st.session_state["preview_display_job_id"] = selected_job
                        st.session_state["pending_page_nav"] = "Overview"
                        try:
                            st.rerun()
                        except Exception:
                            st.experimental_rerun()
                results_payload = None
                metrics_payload = None
                try:
                    results_resp = requests.get(f"{BACKEND_BASE_URL}/results/{selected_job}", timeout=15)
                    results_resp.raise_for_status()
                    results_payload = results_resp.json()
                except Exception:
                    results_payload = None
                try:
                    metrics_resp = requests.get(f"{BACKEND_BASE_URL}/metrics/{selected_job}", timeout=15)
                    metrics_resp.raise_for_status()
                    metrics_payload = metrics_resp.json()
                except Exception:
                    metrics_payload = None

                c_info, c_metrics = st.columns([1.25, 1])
                with c_info:
                    job_meta = (results_payload or {}).get("job", {})
                    st.markdown("**Record details**")
                    status_txt = str(job_meta.get("status", "-")).lower()
                    status_cls = {
                        "complete": "ea-status-complete",
                        "processing": "ea-status-processing",
                        "queued": "ea-status-queued",
                        "failed": "ea-status-failed",
                    }.get(status_txt, "ea-status-queued")
                    st.markdown(
                        f"""
                        <div class="ea-detail-list">
                            <p class="ea-detail-item"><strong>File</strong> {job_meta.get('filename', '-')}</p>
                            <p class="ea-detail-item"><strong>Job ID</strong> {selected_job}</p>
                            <p class="ea-detail-item"><strong>Status</strong> <span class="ea-status-pill {status_cls}">{status_txt.title()}</span></p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with c_metrics:
                    if metrics_payload:
                        st.markdown("**Performance summary**")
                        p1, p2, p3 = st.columns(3)
                        p1.metric("Detections", metrics_payload.get("detections_count", 0))
                        p2.metric("Unique tracks", metrics_payload.get("unique_tracks", 0))
                        p3.metric("Avg herd", f"{float(metrics_payload.get('avg_herd_size', 0.0)):.2f}")
                frames_dir = DATA_DIR / "outputs" / "frames" / selected_job
                frame_files = sorted(frames_dir.glob("frame_*.jpg")) if frames_dir.exists() else []
                if frame_files:
                    st.markdown("**Overview**")
                    if len(frame_files) == 1:
                        st.image(str(frame_files[0]), use_container_width=True)
                    else:
                        idx = st.slider(
                            "Frame preview",
                            min_value=0,
                            max_value=len(frame_files) - 1,
                            value=0,
                            step=1,
                            key=f"history_preview_{selected_job}",
                        )
                        st.image(str(frame_files[idx]), use_container_width=True)

    if not use_mock:
        @st.dialog("Confirm deletion")
        def confirm_delete_selected_dialog():
            pending_delete_jobs = st.session_state.get("admin_pending_delete_jobs", [])
            pending_count = len(pending_delete_jobs)
            if pending_count == 0:
                st.info("No selected history items.")
                if st.button("Close", use_container_width=True, key="admin_close_delete_dialog_empty"):
                    st.session_state["admin_open_delete_dialog"] = False
                    try:
                        st.rerun()
                    except Exception:
                        st.experimental_rerun()
                return

            st.warning(
                f"You are about to permanently delete {pending_count} history item(s). This action cannot be undone."
            )
            c_confirm, c_cancel = st.columns(2)
            with c_confirm:
                if st.button(
                    "Yes, permanently delete",
                    type="primary",
                    use_container_width=True,
                    key="admin_confirm_delete_selected_jobs_modal",
                ):
                    try:
                        resp = requests.post(
                            f"{BACKEND_BASE_URL}/admin/delete-jobs",
                            json={"job_ids": pending_delete_jobs},
                            timeout=20,
                        )
                        resp.raise_for_status()
                        payload = resp.json() if resp.content else {}
                        st.session_state["admin_delete_success_msg"] = (
                            f"Deleted {payload.get('deleted_jobs', 0)} job(s). "
                            f"Removed {payload.get('files_removed', 0)} files and {payload.get('dirs_removed', 0)} folders."
                        )
                        st.session_state.pop("admin_pending_delete_jobs", None)
                        st.session_state["admin_open_delete_dialog"] = False
                        try:
                            st.rerun()
                        except Exception:
                            st.experimental_rerun()
                    except Exception as exc:
                        st.error(f"Delete selected failed: {exc}")
            with c_cancel:
                if st.button(
                    "Cancel",
                    use_container_width=True,
                    key="admin_cancel_delete_selected_jobs_modal",
                ):
                    st.session_state.pop("admin_pending_delete_jobs", None)
                    st.session_state["admin_open_delete_dialog"] = False
                    try:
                        st.rerun()
                    except Exception:
                        st.experimental_rerun()

        with card():
            st.markdown('<div class="ea-block-head">Admin controls</div>', unsafe_allow_html=True)
            st.markdown(
                """
                <div class="ea-admin-wrap">
                    <p class="ea-admin-note">Danger zone. Actions below permanently remove records and artifacts.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("")
            select_all = st.checkbox("Select all jobs", value=False, key="admin_select_all_jobs")
            selected_labels = st.multiselect(
                "Select history items to delete",
                options=labels_for_admin,
                default=labels_for_admin if select_all else [],
                key="admin_selected_job_labels",
            )
            effective_selected_labels = labels_for_admin if select_all else selected_labels
            selected_jobs = [label_to_job_id[x] for x in effective_selected_labels if x in label_to_job_id]
            confirm_clear = st.checkbox("I understand this action cannot be undone")

            c_del, c_all = st.columns(2)
            with c_del:
                if st.button("Delete selected jobs", use_container_width=True):
                    if not confirm_clear:
                        st.warning("Check confirmation first.")
                    elif not selected_jobs:
                        st.warning("Select at least one history item.")
                    else:
                        st.session_state["admin_pending_delete_jobs"] = selected_jobs
                        st.session_state["admin_open_delete_dialog"] = True
                        try:
                            st.rerun()
                        except Exception:
                            st.experimental_rerun()
            with c_all:
                if st.button("Clear all history", use_container_width=True):
                    if not confirm_clear:
                        st.warning("Check confirmation first.")
                    else:
                        try:
                            resp = requests.post(f"{BACKEND_BASE_URL}/admin/clear-history", timeout=15)
                            resp.raise_for_status()
                            st.session_state.pop("last_job_id", None)
                            st.session_state.pop("last_result", None)
                            st.session_state.pop("last_results_payload", None)
                            st.success("All history cleared.")
                            try:
                                st.rerun()
                            except Exception:
                                st.experimental_rerun()
                        except Exception as exc:
                            st.error(f"Clear history failed: {exc}")
            success_msg = st.session_state.pop("admin_delete_success_msg", None)
            if success_msg:
                st.success(success_msg)

            if st.session_state.get("admin_open_delete_dialog", False):
                confirm_delete_selected_dialog()

def timeline_page():
    render_page_hero(
        "Timeline Review",
        "Track when each individual appears, how long it stays visible, and confidence by frame.",
        kicker="Timeline",
    )

    use_mock = st.session_state.get("use_mock", False)
    selected_is_video = True
    job_id: str | None = None
    df = empty_timeline_table()

    with card():
        st.markdown('<div class="ea-block-head">Video run selection</div>', unsafe_allow_html=True)
        if use_mock:
            st.caption("Demo mode")
            job_id = "mock"
            df = mock_timeline_table()
        else:
            jobs_df = fetch_jobs_from_backend()
            job_ids: list[str] = []
            job_id_to_label: dict[str, str] = {}
            label_to_job_id: dict[str, str] = {}
            if jobs_df is not None and not jobs_df.empty:
                jobs_df = jobs_df.copy()
                if "Video name" in jobs_df.columns:
                    jobs_df["is_video"] = jobs_df["Video name"].astype(str).str.lower().str.endswith(
                        (".mp4", ".mov", ".avi")
                    )
                    jobs_df = jobs_df[jobs_df["is_video"]]
                if not jobs_df.empty:
                    job_ids = jobs_df["Job ID"].dropna().astype(str).tolist()
                    for _, row in jobs_df.iterrows():
                        jid = str(row.get("Job ID", ""))
                        if not jid:
                            continue
                        vname = str(row.get("Video name", "Unknown video"))
                        submitted = format_my_time(row.get("Submitted at", ""))
                        short_id = jid[:8]
                        label = f"{vname} | {submitted} | #{short_id}"
                        job_id_to_label[jid] = label
                        label_to_job_id[label] = jid

            if not job_ids:
                st.info("No processed video runs found yet.")
                return

            default_job = st.session_state.get("last_job_id") or job_ids[0]
            if default_job not in job_ids:
                default_job = job_ids[0]
            option_labels = [job_id_to_label.get(j, j) for j in job_ids]
            default_label = job_id_to_label.get(default_job, default_job)
            selected_label = st.selectbox(
                "Select video run",
                option_labels,
                index=option_labels.index(default_label) if default_label in option_labels else 0,
                key="timeline_run_selector",
            )
            job_id = label_to_job_id.get(selected_label, default_job)

            try:
                results_resp = requests.get(f"{BACKEND_BASE_URL}/results/{job_id}", timeout=10)
                results_resp.raise_for_status()
                selected_payload = results_resp.json()
                selected_is_video = is_video_filename((selected_payload.get("job") or {}).get("filename"))
            except Exception:
                selected_is_video = True

            if not selected_is_video:
                st.info("Selected run is an image. Timeline is available for video runs only.")
                return

            rows = fetch_timeline_rows(job_id)
            if rows:
                df = pd.DataFrame(rows).rename(
                    columns={
                        "elephant_id": "Elephant ID",
                        "age_class": "Age class",
                        "first_seen": "First seen (frame)",
                        "last_seen": "Last seen (frame)",
                        "duration_frames": "Duration (frames)",
                    }
                )
            else:
                df = empty_timeline_table()

    if not df.empty and "Age class" in df.columns:
        df["Age class"] = df["Age class"].apply(normalize_age_class_label)

    with card():
        st.markdown('<div class="ea-block-head">Timeline summary</div>', unsafe_allow_html=True)
        if df.empty:
            st.info("No timeline rows available for this run.")
        else:
            durations = pd.to_numeric(df.get("Duration (frames)", pd.Series(dtype=float)), errors="coerce").fillna(0)
            k1, k2, k3 = st.columns(3)
            k1.metric("Individuals", int(df["Elephant ID"].nunique()) if "Elephant ID" in df.columns else len(df))
            k2.metric("Median visibility", f"{int(durations.median())} frames" if len(durations) else "0 frames")
            k3.metric("Longest visibility", f"{int(durations.max())} frames" if len(durations) else "0 frames")
            st.markdown('<p class="ea-note">Visibility values are based on processed frames in the selected run.</p>', unsafe_allow_html=True)

    with card():
        st.markdown('<div class="ea-block-head">Timeline table</div>', unsafe_allow_html=True)
        if df.empty:
            st.info("No timeline records to display.")
        else:
            c1, c2, c3 = st.columns([1.4, 1, 0.8])
            with c1:
                search_txt = st.text_input("Search individual ID", value="", key="timeline_search_elephant_id")
            with c2:
                age_filter = st.selectbox("Age class", ["All", "Adult", "Juvenile", "Baby"], index=0, key="timeline_age_filter")
            with c3:
                page_size = st.selectbox("Rows", [10, 20, 50, 100], index=1, key="timeline_page_size")

            filtered = df.copy()
            if age_filter != "All" and "Age class" in filtered.columns:
                filtered = filtered[filtered["Age class"] == age_filter]
            if search_txt.strip() and "Elephant ID" in filtered.columns:
                needle = search_txt.strip().lower()
                filtered = filtered[filtered["Elephant ID"].astype(str).str.lower().str.contains(needle)]
            if "Duration (frames)" in filtered.columns and not filtered.empty:
                filtered = filtered.sort_values("Duration (frames)", ascending=False).reset_index(drop=True)

            if filtered.empty:
                st.info("No matching records.")
            else:
                pages = int(np.ceil(len(filtered) / page_size))
                page = st.number_input(
                    "Page",
                    min_value=1,
                    max_value=max(1, pages),
                    value=1,
                    step=1,
                    key="timeline_page_no",
                )
                start = (int(page) - 1) * page_size
                end = start + page_size
                st.dataframe(filtered.iloc[start:end], use_container_width=True, hide_index=True)

    if not job_id or not selected_is_video:
        return

    frames_dir = DATA_DIR / "outputs" / "frames" / job_id
    frame_files = sorted(frames_dir.glob("frame_*.jpg")) if frames_dir.exists() else []
    if not frame_files:
        return

    with card():
        st.markdown('<div class="ea-block-head">Frame evidence</div>', unsafe_allow_html=True)
        if len(frame_files) == 1:
            idx = 0
            st.image(str(frame_files[0]), use_container_width=True)
        else:
            idx = st.slider("Frame index", 0, len(frame_files) - 1, 0, 1, key=f"timeline_frame_picker_{job_id}")
            st.image(str(frame_files[idx]), use_container_width=True)

        frame_index = frame_index_from_path(frame_files[idx]) or 0
        detections = fetch_job_detections(job_id) or []
        if not detections:
            return
        df_det = pd.DataFrame(detections)
        if "frame_index" in df_det.columns:
            df_det = df_det[df_det["frame_index"] == frame_index]
        if df_det.empty:
            st.info("No detections on this frame.")
            return
        df_det = df_det.rename(
            columns={
                "track_id": "Elephant ID",
                "cls_label": "Age class",
                "conf": "Confidence",
                "frame_index": "Frame",
            }
        )
        if "Age class" in df_det.columns:
            df_det["Age class"] = df_det["Age class"].apply(normalize_age_class_label)
        if "Confidence" in df_det.columns:
            df_det["Confidence"] = df_det["Confidence"].apply(format_percent)
        show_cols = [c for c in ["Elephant ID", "Age class", "Confidence", "Frame"] if c in df_det.columns]
        if show_cols:
            st.dataframe(df_det[show_cols], use_container_width=True, hide_index=True)

def map_page():
    render_page_hero(
        "Spatial Review",
        "Review detection locations, movement context, and supporting frame evidence.",
        kicker="Map",
    )
    map_section()
    annotated_frames_section()

def behaviour_habitat_page():
    render_page_hero(
        "Timeline & Habitat",
        "Review herd trend, individual timeline visibility, and habitat distribution.",
        kicker="Habitat",
    )
    charts_section()
    st.markdown("")
    timeline_and_habitat_section()


def annotated_frames_section():
    job_id = st.session_state.get("last_job_id")
    if not job_id:
        return
    frames_dir = DATA_DIR / "outputs" / "frames" / job_id
    if not frames_dir.exists():
        return
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_files:
        return
    with card():
        st.markdown("#### Annotated frames")
        st.caption("Detected elephants with track IDs (sampled frames).")
        if len(frame_files) == 1:
            st.image(str(frame_files[0]), use_container_width=True)
        else:
            idx = st.slider("Frame index", 0, len(frame_files) - 1, 0, 1)
            st.image(str(frame_files[idx]), use_container_width=True)

def home_page():
    hero_candidates = [
        PROJECT_ROOT / "streamlit" / "Home_Page.jpg",
        PROJECT_ROOT / "pexels-qwertypr-3691288-1200x675.jpg",
        PROJECT_ROOT / "Dataset" / "V2-3" / "images" / "01c527f9-elephant_535.jpg",
        PROJECT_ROOT / "Dataset" / "V2-3" / "images" / "025079b3-elephant_354.jpg",
        PROJECT_ROOT / "Dataset" / "V2-3" / "images" / "02950a67-elephant_262.jpg",
    ]
    hero_image = next((str(p) for p in hero_candidates if p.exists()), None)

    render_page_hero(
        "Elephant Analytics for Faster, Smarter Field Decisions",
        "Upload footage, detect individuals, and deliver client-ready insights in one workflow.",
        kicker="Elephant Analytics Platform",
    )

    st.markdown("")
    hero_left, hero_right = st.columns([1.2, 1])
    with hero_left:
        if hero_image:
            st.image(
                hero_image,
                use_container_width=True,
                caption="Asian elephant monitoring workflow",
            )
        else:
            st.info("Hero image not found locally.")
    with hero_right:
        with card():
            st.markdown('<div class="ea-block-head">What clients get</div>', unsafe_allow_html=True)
            st.markdown("- Accurate elephant counts")
            st.markdown("- Tracking IDs with visual evidence")
            st.markdown("- Habitat and timeline summaries")
            st.markdown("- Ready-to-present outputs")

    preview_section_header("Core value", "What this platform delivers for reporting and operations.")
    v1, v2, v3 = st.columns(3)
    with v1:
        with card():
            st.markdown('<p style="margin:0; color:#cbd5e1; font-size:0.88rem;">Review speed</p>', unsafe_allow_html=True)
            st.markdown('<p style="margin:0.2rem 0 0 0; color:#f59e0b; font-size:2rem; font-weight:800; line-height:1;">10x</p>', unsafe_allow_html=True)
            st.caption("Faster than manual frame-by-frame review.")
    with v2:
        with card():
            st.markdown('<p style="margin:0; color:#cbd5e1; font-size:0.88rem;">Output confidence</p>', unsafe_allow_html=True)
            st.markdown('<p style="margin:0.2rem 0 0 0; color:#f59e0b; font-size:2rem; font-weight:800; line-height:1;">High</p>', unsafe_allow_html=True)
            st.caption("Tracked IDs, frames, and timeline evidence.")
    with v3:
        with card():
            st.markdown('<p style="margin:0; color:#cbd5e1; font-size:0.88rem;">Deployment fit</p>', unsafe_allow_html=True)
            st.markdown('<p style="margin:0.2rem 0 0 0; color:#f59e0b; font-size:2rem; font-weight:800; line-height:1;">Local</p>', unsafe_allow_html=True)
            st.caption("Operational in field and office environments.")

    preview_section_header("3-step workflow", "Simple sequence from raw media to client output.")
    s1, s2, s3 = st.columns(3)
    with s1:
        with card():
            st.markdown('<p style="margin:0; font-weight:700;"><span style="color:#f59e0b;">Step 1</span> - Upload</p>', unsafe_allow_html=True)
            st.caption("Add video or image files.")
    with s2:
        with card():
            st.markdown('<p style="margin:0; font-weight:700;"><span style="color:#f59e0b;">Step 2</span> - Detect & track</p>', unsafe_allow_html=True)
            st.caption("Auto-detect elephants and assign IDs.")
    with s3:
        with card():
            st.markdown('<p style="margin:0; font-weight:700;"><span style="color:#f59e0b;">Step 3</span> - Report</p>', unsafe_allow_html=True)
            st.caption("Use charts, timeline, and maps in client reports.")

    preview_section_header("Quick access", "Jump directly to the section you need.")
    c1, c2, c3 = st.columns(3)
    with c1:
        with card():
            st.markdown("**Overview**")
            st.caption("Open client summary and confidence analytics.")
            if st.button("Open Overview", use_container_width=True, key="home_open_preview"):
                st.session_state["pending_page_nav"] = "Overview"
                st.rerun()
    with c2:
        with card():
            st.markdown("**Timeline**")
            st.caption("Review individual visibility and age classes.")
            if st.button("Open Timeline", use_container_width=True, key="home_open_timeline"):
                st.session_state["pending_page_nav"] = "Timeline"
                st.rerun()
    with c3:
        with card():
            st.markdown("**Map**")
            st.caption("Review location evidence and movement context.")
            if st.button("Open Map", use_container_width=True, key="home_open_map"):
                st.session_state["pending_page_nav"] = "Map"
                st.rerun()

    st.markdown("")
    c1, c2, c3 = st.columns([1, 1.6, 1])
    with c2:
        if st.button("Start a New Analysis", use_container_width=True, type="primary", key="home_start_analysis"):
            st.session_state["pending_page_nav"] = "Upload & jobs"
            st.rerun()
# ===================== ROUTER =====================

PAGES = {
    "Home": "home",
    "Upload & jobs": "upload",
    "Overview": "overview",
    "Timeline": "timeline",
    "Map": "map",
    "Timeline & habitat": "habitat",
}

if "pending_page_nav" in st.session_state:
    next_page_label = st.session_state.pop("pending_page_nav")
    st.session_state["page_nav"] = next_page_label

with st.sidebar:
    st.title("Elephant Analytics")
    if "page_nav" not in st.session_state:
        st.session_state["page_nav"] = "Home"
    page_label = st.radio("Navigation", list(PAGES.keys()), key="page_nav")

page = PAGES[page_label]
active_mode_key = str(st.session_state.get("last_pipeline_mode_key", "") or "").strip().lower()
is_fast_like_mode = active_mode_key in {"fast_trend", "balanced_2fps", "fast"}

if page == "home":
    home_page()
elif page == "overview":
    hero_section()
    if not is_fast_like_mode:
        st.markdown("")
        individual_spotlight()
        st.markdown("")
        result_quality_section()
    if st.session_state.get("preview_is_video", True):
        st.markdown("")
        preview_section_header("Herd trend", "How group size changes across the video.")
        charts_section()
    elif is_fast_like_mode:
        st.caption("Fast-like mode hides ID-heavy cards/tables and keeps trend-first outputs.")
    st.markdown("")
    preview_section_header("Timeline & habitat", "When individuals appear and where activity concentrates.")
    timeline_and_habitat_section()
    st.markdown("")
    preview_section_header("Spatial view", "Geographic context for detections and movement.")
    map_section()
elif page == "upload":
    upload_jobs_page()
elif page == "timeline":
    timeline_page()
elif page == "map":
    map_page()
elif page == "habitat":
    behaviour_habitat_page()





