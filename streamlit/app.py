import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import html
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
MODEL_UI_META = {
    "Research Model v2_s (Recommended)": {
        "title": "Research Model v2_s",
        "tag": "Recommended",
        "purpose": "Best all-round production-style analysis",
        "note": "Use for most real runs when you want the strongest overall output quality.",
    },
    "Online Baseline (Roboflow v1)": {
        "title": "Online Baseline",
        "tag": "Demo-friendly",
        "purpose": "External-style baseline for comparison",
        "note": "Useful for benchmark checks or quick side-by-side validation against your main model.",
    },
}

PIPELINE_MODE_OPTIONS = {
    "Original Pipeline (Identity Accurate)": "quality",
    "Fast Trend (2 FPS)": "fast_trend",
    "Value 1x (Tracked ~5 FPS)": "value_1x",
}
ANALYZE_RUN_MODE_OPTIONS = {
    "Fast Review mode": "fast_trend",
    "Tracking / Research mode": "quality",
}
RUN_MODE_UI_META = {
    "Fast Review mode": {
        "title": "Fast Review",
        "tag": "Fast",
        "optimize": "Optimized for faster turnaround and lighter review flow.",
        "tradeoff": "Less focused on identity continuity across the whole run.",
        "when": "Use when you want quick review, trend reading, and current-run inspection.",
    },
    "Tracking / Research mode": {
        "title": "Original / Identity Accurate",
        "tag": "Identity-accurate",
        "optimize": "Optimized for stronger identity consistency and final reporting confidence.",
        "tradeoff": "Slower processing and heavier review output.",
        "when": "Use for final review, client-facing reporting, and continuity-sensitive runs.",
    },
    "Original Pipeline (Identity Accurate)": {
        "title": "Original / Identity Accurate",
        "tag": "Identity-accurate",
        "optimize": "Optimized for stronger identity consistency and final reporting confidence.",
        "tradeoff": "Slower processing and heavier review output.",
        "when": "Use for final review, client-facing reporting, and continuity-sensitive runs.",
    },
    "Researcher mode": {
        "title": "Researcher / Identity Accurate",
        "tag": "Research",
        "optimize": "Optimized for deeper analysis, stronger continuity review, and richer evidence output.",
        "tradeoff": "More time-consuming and less lightweight for quick passes.",
        "when": "Use for research workflows, continuity-sensitive inspection, and reporting review.",
    },
    "Research mode": {
        "title": "Research / Identity Accurate",
        "tag": "Research",
        "optimize": "Optimized for deeper analysis, stronger continuity review, and richer evidence output.",
        "tradeoff": "More time-consuming and less lightweight for quick passes.",
        "when": "Use for research workflows, continuity-sensitive inspection, and reporting review.",
    },
}
SHOW_PIPELINE_MODE_CONTROL = True
DEFAULT_PIPELINE_MODE_LABEL = "Original Pipeline (Identity Accurate)"
SAMPLED_COUNT_CAVEAT = "Sampled checkpoint-based value. Not frame-perfect live inference."

st.set_page_config(
    page_title="Elephant Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "use_mock" not in st.session_state:
    st.session_state["use_mock"] = False
if "active_processing_job_id" not in st.session_state:
    st.session_state["active_processing_job_id"] = None
if "selected_history_job_id" not in st.session_state:
    st.session_state["selected_history_job_id"] = None
if "results_view_job_id" not in st.session_state:
    st.session_state["results_view_job_id"] = None

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
    [data-baseweb="tab-list"] {
        gap: 0.45rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.18);
        margin-bottom: 0.8rem;
    }
    [data-baseweb="tab"] {
        border: 1px solid rgba(148, 163, 184, 0.18) !important;
        border-radius: 12px 12px 0 0 !important;
        background: rgba(15, 23, 42, 0.14) !important;
        color: #cbd5e1 !important;
        padding: 0.5rem 0.8rem !important;
    }
    [aria-selected="true"][data-baseweb="tab"] {
        background: linear-gradient(180deg, rgba(56, 189, 248, 0.16), rgba(15, 23, 42, 0.10)) !important;
        border-color: rgba(56, 189, 248, 0.34) !important;
        color: #f8fafc !important;
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


@contextmanager
def maybe_card(enabled: bool = True):
    if enabled:
        with card():
            yield
    else:
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


def fetch_api_health() -> tuple[bool, str]:
    try:
        resp = requests.get(f"{BACKEND_BASE_URL}/health", timeout=3)
        resp.raise_for_status()
        payload = resp.json() if resp.content else {}
        if bool(payload.get("ok")):
            return True, "API online"
        return False, "API returned unexpected status"
    except Exception:
        return False, "API offline"


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


def get_fast_review_bundle(job_id: str | None) -> dict:
    if not job_id:
        return {}
    metrics = fetch_job_metrics(str(job_id)) or {}
    direct_bundle = metrics.get("fast_review_bundle")
    if isinstance(direct_bundle, dict) and direct_bundle:
        return direct_bundle
    legacy_root = metrics.get("results_bundle")
    if isinstance(legacy_root, dict) and legacy_root:
        nested = legacy_root.get("fast_review_bundle")
        if isinstance(nested, dict) and nested:
            return nested
        if "trend" in legacy_root or "review" in legacy_root:
            return legacy_root

    herd_series = metrics.get("herd_series") or []
    sampled_series = []
    for row in herd_series:
        item = {
            "frame_index": int(row.get("frame_index", 0)),
            "sampled_visible_count": int(row.get("herd_size", 0) or 0),
        }
        if row.get("time_seconds") is not None:
            item["time_seconds"] = row.get("time_seconds")
        sampled_series.append(item)
    sampled_counts = [int(row["sampled_visible_count"]) for row in sampled_series]
    return {
        "schema_version": "fallback",
        "naming": {"primary_count_metric": "sampled_visible_count"},
        "review": {
            "sampled_visible_count": sampled_counts[-1] if sampled_counts else 0,
            "run_summary": {
                "detections_count": int(metrics.get("detections_count", 0) or 0),
                "unique_tracks": int(metrics.get("unique_tracks", 0) or 0),
                "avg_herd_size": float(metrics.get("avg_herd_size", 0.0) or 0.0),
                "peak_herd_size": int(metrics.get("peak_herd_size", 0) or 0),
            },
        },
        "trend": {
            "sampled_visible_count_over_time": sampled_series,
            "min_sampled_visible_count": min(sampled_counts) if sampled_counts else 0,
            "avg_sampled_visible_count": float(np.mean(sampled_counts)) if sampled_counts else 0.0,
            "peak_sampled_visible_count": max(sampled_counts) if sampled_counts else 0,
        },
        "run_diagnostics": {
            "caveats": [
                SAMPLED_COUNT_CAVEAT,
            ]
        },
    }


def get_tracking_bundle(job_id: str | None) -> dict:
    if not job_id:
        return {}
    metrics = fetch_job_metrics(str(job_id)) or {}
    direct_bundle = metrics.get("tracking_bundle")
    if isinstance(direct_bundle, dict) and direct_bundle:
        return direct_bundle
    legacy_root = metrics.get("results_bundle")
    if isinstance(legacy_root, dict):
        nested = legacy_root.get("tracking_bundle")
        if isinstance(nested, dict):
            return nested
    return {}


def get_results_bundle(job_id: str | None) -> dict:
    # Backward-compatible alias used by existing code paths.
    return get_fast_review_bundle(job_id)


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


def get_active_processing_job_id() -> str | None:
    return st.session_state.get("active_processing_job_id")


def get_selected_history_job_id() -> str | None:
    return st.session_state.get("selected_history_job_id")


def get_results_view_job_id() -> str | None:
    return st.session_state.get("results_view_job_id")


def get_current_context_job_id() -> str | None:
    for key in ("active_processing_job_id", "results_view_job_id", "last_job_id", "selected_history_job_id"):
        value = st.session_state.get(key)
        if value:
            return str(value)
    payload = st.session_state.get("last_results_payload") or {}
    payload_job_id = ((payload.get("job") or {}).get("id"))
    return str(payload_job_id) if payload_job_id else None


def record_ui_timing(name: str, seconds: float) -> None:
    bucket = st.session_state.get("_ui_timing_last", {})
    if not isinstance(bucket, dict):
        bucket = {}
    bucket[name] = round(float(seconds), 4)
    st.session_state["_ui_timing_last"] = bucket


def get_real_summary() -> dict | None:
    job_id = get_results_view_job_id()
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


def format_display_job_id(job_id: str | None, input_type: str | None = None) -> str:
    raw = str(job_id or "").strip()
    if not raw:
        return "-"
    short = raw[:6].upper()
    if str(input_type or "").strip().lower() == "image":
        return f"IMG-{short}"
    if str(input_type or "").strip().lower() == "video":
        return f"RUN-{short}"
    return f"Job {short}"


def resolve_run_mode_meta(run_mode_choice: str, pipeline_mode_key: str | None = None) -> dict:
    direct = RUN_MODE_UI_META.get(str(run_mode_choice or "").strip())
    if direct:
        return direct
    normalized_key = str(pipeline_mode_key or "").strip().lower()
    if normalized_key == "fast_trend":
        return RUN_MODE_UI_META.get("Fast Review mode", {})
    if normalized_key in {"quality", "identity_accurate", "original"}:
        return RUN_MODE_UI_META.get("Tracking / Research mode", {})
    return {
        "title": str(run_mode_choice or "Pipeline mode"),
        "tag": "Mode",
        "optimize": "Configured for the selected analysis mode.",
        "tradeoff": "Review runtime and output detail depend on the selected mode.",
        "when": "Use this mode when it best matches the current analysis task.",
    }


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


def navigate_to_page(page_label: str) -> None:
    st.session_state["pending_page_nav"] = page_label
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()


def render_confidence_graph(df: pd.DataFrame, is_video: bool) -> None:
    if df.empty or "conf" not in df.columns:
        st.info("No confidence data available.")
        return

    chart_df = df.copy()
    chart_df["conf"] = pd.to_numeric(chart_df["conf"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    if "cls_label" in chart_df.columns:
        chart_df["age_label"] = chart_df["cls_label"].apply(normalize_age_class_label)
    else:
        chart_df["age_label"] = "Unknown"

    if not is_video:
        st.caption("Confidence across detections in the current image")
        chart_df = chart_df.reset_index(drop=True)
        chart_df["detection_no"] = np.arange(1, len(chart_df) + 1)
        chart_df["conf_pct"] = chart_df["conf"] * 100.0
        conf_bar = (
            alt.Chart(chart_df)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#38bdf8")
            .encode(
                x=alt.X("detection_no:O", axis=alt.Axis(title="Detection", labelAngle=0)),
                y=alt.Y("conf_pct:Q", axis=alt.Axis(title="Confidence (%)", labelAngle=0)),
                tooltip=[
                    alt.Tooltip("detection_no:Q", title="Detection"),
                    alt.Tooltip("age_label:N", title="Age class"),
                    alt.Tooltip("conf_pct:Q", title="Confidence (%)", format=".1f"),
                ],
            )
            .properties(height=220)
            .configure_view(strokeOpacity=0)
        )
        st.altair_chart(conf_bar, use_container_width=True)
        return

    if "frame_index" not in chart_df.columns:
        st.info("Frame-level confidence trend is not available for this run.")
        return

    st.caption("Confidence trend across sampled frames")
    trend_df = (
        chart_df.groupby("frame_index", as_index=False)
        .agg(
            detections=("conf", "count"),
            avg_conf=("conf", "mean"),
        )
        .sort_values("frame_index")
    )
    if trend_df.empty:
        st.info("No confidence trend is available for this run.")
        return

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

def hero_section(
    show_header: bool = True,
    include_preview_snapshot: bool = True,
    compact_review: bool = False,
    wrap_stats_card: bool = True,
):
    if show_header:
        render_page_hero(
            "Client Summary",
            "Clear evidence and key numbers from the latest processed job.",
            kicker="Preview",
        )

    if include_preview_snapshot:
        col_main, col_stats = st.columns([1.6, 1])
    else:
        col_main = None
        col_stats = st.container()

    job_id = get_results_view_job_id()
    use_mock = st.session_state.get("use_mock", False)
    summary = (
        mock_detection_summary()
        if use_mock
        else (get_real_summary() or empty_detection_summary())
    )
    metrics = fetch_job_metrics(job_id) if (job_id and not use_mock) else None
    results_bundle = get_results_bundle(job_id) if (job_id and not use_mock) else {}
    dominant_habitat = "-"
    if metrics:
        dominant_habitat = str(metrics.get("dominant_habitat") or "-")

    unique_count = int(summary.get("unique_elephants", 0) or 0)
    avg_herd = float(summary.get("avg_herd_size", 0.0) or 0.0)
    review_bundle = (results_bundle or {}).get("review", {})
    trend_bundle = (results_bundle or {}).get("trend", {})
    sampled_visible_count = int(review_bundle.get("sampled_visible_count", 0) or 0)
    avg_sampled_visible_count = float(trend_bundle.get("avg_sampled_visible_count", avg_herd) or avg_herd)
    mode_key_for_summary = str(st.session_state.get("last_pipeline_mode_key", "") or "").strip().lower()
    is_fast_trend_mode_summary = mode_key_for_summary in {"fast_trend", "fast"}
    if is_fast_trend_mode_summary:
        conclusion = (
            f"Fast trend mode: sampled visible count now {sampled_visible_count}. "
            f"Dominant habitat: {dominant_habitat}. Use Identity Accurate mode for stronger ID continuity checks."
        )
    elif sampled_visible_count > 0:
        conclusion = (
            f"Latest sampled visible count is {sampled_visible_count}, with average sampled visible count {avg_sampled_visible_count:.1f}. "
            f"Dominant habitat: {dominant_habitat}."
        )
    else:
        conclusion = "No sampled visible elephants are shown in the current output. Validate input quality and rerun if needed."

    display_job_id = job_id
    preview_is_video = True if use_mock else resolve_job_is_video(display_job_id, metrics=metrics)
    last_mode_key = str(st.session_state.get("last_pipeline_mode_key", "") or "").strip().lower()
    selected: Path | None = None
    if include_preview_snapshot and col_main is not None:
        with col_main:
            frame_files: list[Path] = []
            if job_id:
                frames_dir = DATA_DIR / "outputs" / "frames" / job_id
                frame_files = sorted(frames_dir.glob("frame_*.jpg")) if frames_dir.exists() else []

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
            else:
                st.info("No preview available for this run.")

            st.markdown(
                f"""
                <div class="ea-surface" style="margin-top:0.55rem;">
                    <p style="margin:0; font-size:0.76rem; letter-spacing:0.06em; text-transform:uppercase; color:#94a3b8; font-weight:700;">Summary</p>
                    <p style="margin:0.24rem 0 0 0; color:#f8fafc; font-size:0.94rem;">{conclusion}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    selected_frame_idx = frame_index_from_path(selected) if selected else None
    st.session_state["preview_frame_index"] = selected_frame_idx
    st.session_state["preview_is_video"] = preview_is_video

    with col_stats:
        with maybe_card(wrap_stats_card):
            st.markdown("#### Run summary")
            if not include_preview_snapshot:
                st.caption(conclusion)
            mode_key = str(st.session_state.get("last_pipeline_mode_key", "") or "").strip().lower()
            is_fast_trend_mode = mode_key in {"fast_trend", "fast"}
            k1, k2, k3 = st.columns(3)
            with k1:
                if is_fast_trend_mode:
                    peak_herd = int((metrics or {}).get("peak_herd_size", 0) or 0)
                    st.metric("Maximum seen at once", peak_herd)
                else:
                    st.metric("Sampled visible count", sampled_visible_count)
            with k2:
                st.metric("Avg sampled visible count", f"{avg_sampled_visible_count:.2f}")
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

            details_min_h = "132px" if compact_review else "190px"
            details_metric_size = "1.08rem" if compact_review else "1.32rem"
            st.markdown(
                f"""
                <div class="ea-surface" style="padding:0.72rem 0.8rem; min-height:{details_min_h};">
                    <p style="margin:0; color:#f8fafc; font-size:0.92rem; font-weight:760;">Run details</p>
                    <div style="display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:0.7rem; margin-top:0.55rem;">
                        <div style="border:1px solid rgba(148,163,184,0.25); border-radius:10px; padding:0.58rem 0.62rem; background:rgba(2,6,23,0.22);">
                            <p style="margin:0; color:#94a3b8; font-size:0.74rem;">Video duration</p>
                            <p style="margin:0.12rem 0 0 0; color:#93c5fd; font-size:{details_metric_size}; font-weight:780; line-height:1.05;">{video_duration_txt}</p>
                        </div>
                        <div style="border:1px solid rgba(148,163,184,0.25); border-radius:10px; padding:0.58rem 0.62rem; background:rgba(2,6,23,0.22);">
                            <p style="margin:0; color:#94a3b8; font-size:0.74rem;">Processing time</p>
                            <p style="margin:0.12rem 0 0 0; color:#93c5fd; font-size:{details_metric_size}; font-weight:780; line-height:1.05;">{processing_time_txt}</p>
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


def _resolve_reference_video_info(display_job_id: str | None, use_mock: bool) -> dict:
    info = {"path": None, "label": None}
    if use_mock or not display_job_id:
        return info

    previews_dir = DATA_DIR / "outputs" / "previews"
    if previews_dir.exists():
        candidates = sorted(
            previews_dir.glob(f"{display_job_id}*.mp4"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            info["path"] = str(candidates[0])
            info["label"] = "Reference playback"
            return info

    source_video = resolve_job_video_path(display_job_id)
    if source_video and is_video_filename(source_video):
        info["path"] = str(source_video)
        info["label"] = "Reference playback (source video)"
    return info


def _resolve_reference_video_path(display_job_id: str | None, use_mock: bool) -> str | None:
    return _resolve_reference_video_info(display_job_id, use_mock).get("path")


def _resolve_representative_frame_path(display_job_id: str | None, use_mock: bool) -> str | None:
    if use_mock or not display_job_id:
        return None
    frames_dir = DATA_DIR / "outputs" / "frames" / str(display_job_id)
    if not frames_dir.exists():
        return None
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_files:
        return None

    by_index: dict[int, Path] = {}
    indexed: list[tuple[int, Path]] = []
    for fp in frame_files:
        idx = frame_index_from_path(fp)
        if idx is None:
            continue
        by_index[idx] = fp
        indexed.append((idx, fp))
    indexed.sort(key=lambda t: t[0])

    herd_df = _get_checkpoint_series(str(display_job_id), use_mock=False)
    target_idx: int | None = None
    if not herd_df.empty and "frame_index" in herd_df.columns:
        try:
            target_idx = int(herd_df.iloc[-1].get("frame_index", 0))
        except Exception:
            target_idx = None

    if target_idx is not None and indexed:
        exact = by_index.get(target_idx)
        if exact is not None:
            return str(exact)
        nearest_idx, nearest_path = min(indexed, key=lambda t: abs(t[0] - target_idx))
        _ = nearest_idx
        return str(nearest_path)

    # Deterministic fallback: middle frame of the same run.
    return str(frame_files[len(frame_files) // 2])


def _resolve_frame_path_for_index(display_job_id: str | None, frame_index: int | None) -> str | None:
    if not display_job_id:
        return None
    frames_dir = DATA_DIR / "outputs" / "frames" / str(display_job_id)
    if not frames_dir.exists():
        return None
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_files:
        return None
    if frame_index is None:
        return str(frame_files[len(frame_files) // 2])

    indexed: list[tuple[int, Path]] = []
    for fp in frame_files:
        idx = frame_index_from_path(fp)
        if idx is not None:
            indexed.append((idx, fp))
    if not indexed:
        return str(frame_files[len(frame_files) // 2])
    indexed.sort(key=lambda t: t[0])
    exact = [p for idx, p in indexed if idx == int(frame_index)]
    if exact:
        return str(exact[0])
    nearest_idx, nearest_path = min(indexed, key=lambda t: abs(t[0] - int(frame_index)))
    _ = nearest_idx
    return str(nearest_path)


def _sync_video_review_selection(
    job_id: str,
    frame_indices: list[int],
    fps: float | None,
    source: str,
) -> None:
    if not frame_indices:
        return
    slider_key = f"review_workspace_scrub_frame_{job_id}"
    paused_key = f"paused_seconds_input_{job_id}"
    analysis_key = f"paused_frame_analysis_{job_id}"
    if source == "slider":
        frame_idx = int(st.session_state.get(slider_key, frame_indices[0]) or frame_indices[0])
        if frame_idx not in frame_indices:
            frame_idx = frame_indices[min(range(len(frame_indices)), key=lambda i: abs(frame_indices[i] - frame_idx))]
        st.session_state["preview_frame_index"] = frame_idx
        if fps and fps > 0:
            st.session_state[paused_key] = round(frame_idx / fps, 2)
        state = st.session_state.get(analysis_key, {}) or {}
        state["frame_index"] = frame_idx
        if fps and fps > 0:
            state["paused_seconds"] = float(frame_idx / fps)
            state["resolved_time_txt"] = format_seconds(frame_idx / fps)
        st.session_state[analysis_key] = state
        return

    paused_sec = float(st.session_state.get(paused_key, 0.0) or 0.0)
    if fps and fps > 0:
        target_frame = int(round(paused_sec * fps))
    else:
        target_frame = int(round(paused_sec))
    frame_idx = int(min(frame_indices, key=lambda frame: abs(frame - target_frame)))
    st.session_state["preview_frame_index"] = frame_idx
    if fps and fps > 0:
        st.session_state[paused_key] = round(frame_idx / fps, 2)
    state = st.session_state.get(analysis_key, {}) or {}
    state["frame_index"] = frame_idx
    if fps and fps > 0:
        state["paused_seconds"] = float(frame_idx / fps)
        state["resolved_time_txt"] = format_seconds(frame_idx / fps)
    st.session_state[analysis_key] = state


def _review_stat(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div style="border:1px solid rgba(148,163,184,0.14); border-radius:12px; padding:0.72rem 0.78rem; background:rgba(15,23,42,0.14);">
            <p style="margin:0; color:#7f91a8; font-size:0.7rem; letter-spacing:0.05em; text-transform:uppercase; font-weight:700;">{label}</p>
            <p style="margin:0.24rem 0 0 0; color:#f8fafc; font-size:1rem; font-weight:760; line-height:1.15;">{value}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_inspection_overview_card(
    title: str,
    description: str,
    items: list[tuple[str, str, str | None]],
    extra_footer: str | None = None,
    variant: str = "default",
) -> None:
    card_class = f"ea-inspect-card{(' ' + variant) if variant else ''}"
    item_html = []
    for label, value, tone in items:
        tone_class = f" {tone}" if tone else ""
        item_html.append(
            f'<div class="ea-inspect-item{tone_class}">'
            f'<p class="label">{html.escape(str(label))}</p>'
            f'<p class="value">{html.escape(str(value))}</p>'
            f"</div>"
        )
    footer_html = ""
    if extra_footer:
        footer_html = f'<p class="ea-inspect-foot">{html.escape(str(extra_footer))}</p>'
    items_markup = "".join(item_html)
    st.markdown(
        (
            f'<div class="{card_class}">'
            f'<p class="ea-inspect-title">{html.escape(title)}</p>'
            f'<p class="ea-inspect-copy">{html.escape(description)}</p>'
            f'<div class="ea-inspect-grid">{items_markup}</div>'
            f"{footer_html}"
            f"</div>"
        ),
        unsafe_allow_html=True,
    )


def _load_video_frame(video_path: str | None, frame_index: int | None) -> np.ndarray | None:
    if not video_path or frame_index is None or frame_index < 0:
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        return None
    return frame


def _draw_yellow_detection_card(
    image_bgr: np.ndarray,
    detections: list[dict],
) -> np.ndarray:
    frame = image_bgr.copy()
    box_color = (0, 255, 255)
    text_color = (20, 20, 20)
    occupied_bands: list[tuple[int, int, int]] = []
    normalized_dets = []
    for det in detections:
        x1 = int(float(det.get("x1", 0)))
        y1 = int(float(det.get("y1", 0)))
        x2 = int(float(det.get("x2", 0)))
        y2 = int(float(det.get("y2", 0)))
        if x2 <= x1 or y2 <= y1:
            continue
        normalized_dets.append((x1, y1, x2, y2, det))
    normalized_dets.sort(key=lambda item: (item[1], item[0]))

    for x1, y1, x2, y2, det in normalized_dets:
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        age_txt = normalize_age_class_label(det.get("cls_label"))
        track_id = det.get("track_id")
        label_txt = f"ID {int(track_id):02d} | {age_txt}" if track_id not in {None, "", "None"} else age_txt
        (tw, th), baseline = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        tag_x1 = max(0, min(frame.shape[1] - 1, x1))
        tag_h = th + baseline + 10
        preferred_y1 = max(0, y1 - tag_h - 4)
        if preferred_y1 <= 0:
            preferred_y1 = min(max(0, y2 + 4), max(0, frame.shape[0] - tag_h - 1))
        tag_y1 = preferred_y1
        tag_y2 = min(frame.shape[0] - 1, tag_y1 + tag_h)
        for occ_x1, occ_x2, occ_y1 in occupied_bands:
            overlaps_x = not (tag_x1 + tw + 14 < occ_x1 or tag_x1 > occ_x2)
            overlaps_y = abs(tag_y1 - occ_y1) < tag_h
            if overlaps_x and overlaps_y:
                candidate_y1 = min(max(0, occ_y1 + tag_h + 4), max(0, frame.shape[0] - tag_h - 1))
                tag_y1 = candidate_y1
                tag_y2 = min(frame.shape[0] - 1, tag_y1 + tag_h)
        tag_x2 = min(frame.shape[1] - 1, tag_x1 + tw + 14)
        occupied_bands.append((tag_x1, tag_x2, tag_y1))
        cv2.rectangle(frame, (tag_x1, tag_y1), (tag_x2, tag_y2), box_color, -1)
        cv2.putText(
            frame,
            label_txt,
            (tag_x1 + 7, tag_y2 - baseline - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            text_color,
            1,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _get_checkpoint_series(display_job_id: str | None, use_mock: bool) -> pd.DataFrame:
    results_bundle = get_results_bundle(str(display_job_id)) if (display_job_id and not use_mock) else {}
    trend_bundle = (results_bundle or {}).get("trend", {})
    herd_df = pd.DataFrame(trend_bundle.get("sampled_visible_count_over_time") or [])
    if use_mock:
        demo = mock_herd_series().copy()
        demo["frame_index"] = np.arange(len(demo))
        demo["time_seconds"] = np.arange(len(demo)) * 30
        demo = demo.rename(columns={"herd_size": "sampled_visible_count"})
        herd_df = demo
    if herd_df.empty:
        return herd_df
    if "frame_index" not in herd_df.columns:
        herd_df["frame_index"] = np.arange(len(herd_df))
    if "time_seconds" not in herd_df.columns:
        herd_df["time_seconds"] = np.nan
    if "sampled_visible_count" not in herd_df.columns:
        herd_df["sampled_visible_count"] = np.nan
    return herd_df.sort_values("frame_index").reset_index(drop=True)


def render_client_review_panel():
    display_job_id = get_results_view_job_id()
    use_mock = st.session_state.get("use_mock", False)
    if not display_job_id and not use_mock:
        st.info("Run one analysis to enable client review.")
        return

    is_video = st.session_state.get("preview_is_video", True)
    if not is_video and not use_mock:
        st.caption("Reference playback is available for video runs. Showing representative frame evidence.")

    video_info = _resolve_reference_video_info(display_job_id, use_mock)
    video_path = str(video_info.get("path") or "")
    video_label = str(video_info.get("label") or "").strip()
    rep_frame_path = _resolve_representative_frame_path(display_job_id, use_mock)

    if video_path and rep_frame_path:
        m1, m2 = st.columns([1.45, 1.0])
        with m1:
            st.video(video_path)
            st.caption(video_label or "Reference playback")
        with m2:
            st.image(rep_frame_path, caption="Representative frame", width="stretch")
    else:
        if video_path:
            st.video(video_path)
            st.caption(video_label or "Reference playback")
        elif use_mock:
            st.info("Demo mode: reference playback media is not attached.")
        if rep_frame_path:
            st.image(rep_frame_path, caption="Representative frame", width="stretch")

    if (not video_path) and (not rep_frame_path) and (not use_mock):
        st.info("No preview available for this run.")
        a1, a2 = st.columns([1, 1.3])
        a1.caption("No visual media found under this locked run.")
        active_processing = get_active_processing_job_id()
        if a2.button(
            "Switch to active run",
            key=f"client_switch_active_{display_job_id or 'none'}",
            disabled=not active_processing,
        ):
            st.session_state["results_view_job_id"] = active_processing
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()

    herd_df = _get_checkpoint_series(display_job_id, use_mock)
    st.caption("Reference playback. Checkpoint values are sampled and not frame-perfect live sync.")
    if herd_df.empty:
        st.caption("No sampled checkpoint series available.")
        return
    row = herd_df.iloc[-1]
    t_seconds = row.get("time_seconds")
    t_txt = format_seconds(t_seconds) if pd.notna(t_seconds) else "-"
    frame_txt = int(row.get("frame_index", 0))
    herd_txt = "-" if pd.isna(row.get("sampled_visible_count")) else int(row.get("sampled_visible_count", 0))
    c1, c2, c3 = st.columns(3)
    c1.metric("Latest sampled checkpoint", t_txt)
    c2.metric("Frame index", frame_txt)
    c3.metric("Sampled visible count", herd_txt)
    st.caption(SAMPLED_COUNT_CAVEAT)


def render_research_checkpoint_panel(show_video: bool = True):
    display_job_id = get_results_view_job_id()
    use_mock = st.session_state.get("use_mock", False)
    if not display_job_id and not use_mock:
        st.info("Run one analysis to enable checkpoint inspection.")
        return

    herd_df = _get_checkpoint_series(display_job_id, use_mock)
    if herd_df.empty:
        st.caption("No checkpoint series is available for inspection.")
        return
    key_base = f"{display_job_id or 'mock'}"
    state_key = f"research_checkpoint_idx_{key_base}"
    if state_key not in st.session_state:
        st.session_state[state_key] = 0

    idx_max = max(0, len(herd_df) - 1)
    current_idx = int(st.session_state.get(state_key, 0))
    current_idx = max(0, min(idx_max, current_idx))
    step_choice = st.segmented_control(
        "Step",
        options=["1", "5", "10"],
        default="1",
        key=f"research_step_{key_base}",
    )
    step_size = max(1, int(step_choice or "1"))
    st.caption(f"Current sampled snapshot {current_idx + 1} of {idx_max + 1} (step {step_size})")
    b1, b2, b3, b4 = st.columns([1, 1, 1, 1])
    if b1.button("First", key=f"research_first_{key_base}", disabled=current_idx <= 0):
        current_idx = 0
    if b2.button("Prev", key=f"research_prev_{key_base}", disabled=current_idx <= 0):
        current_idx = max(0, current_idx - step_size)
    if b3.button("Next", key=f"research_next_{key_base}", disabled=current_idx >= idx_max):
        current_idx = min(idx_max, current_idx + step_size)
    if b4.button("Last", key=f"research_last_{key_base}", disabled=current_idx >= idx_max):
        current_idx = idx_max
    st.progress((current_idx + 1) / max(1, idx_max + 1))

    if idx_max <= 0:
        idx = 0
        st.caption("Only one checkpoint is available for this run.")
    else:
        idx = st.slider(
            "Current sampled snapshot scrub",
            min_value=0,
            max_value=idx_max,
            value=current_idx,
            step=1,
            key=f"research_scrub_{key_base}",
        )
    st.session_state[state_key] = int(idx)
    current_idx = int(idx)
    row = herd_df.iloc[int(idx)]
    t_seconds = row.get("time_seconds")
    t_txt = format_seconds(t_seconds) if pd.notna(t_seconds) else "-"
    frame_txt = int(row.get("frame_index", 0))
    st.session_state["preview_frame_index"] = frame_txt
    herd_txt = "-" if pd.isna(row.get("sampled_visible_count")) else int(row.get("sampled_visible_count", 0))
    c1, c2, c3 = st.columns(3)
    c1.metric("Snapshot time", t_txt)
    c2.metric("Frame index", frame_txt)
    c3.metric("Sampled visible count", herd_txt)
    video_path = _resolve_reference_video_path(display_job_id, use_mock)
    if show_video and video_path:
        st.video(video_path)
    st.caption("Current sampled snapshot is sampled and not frame-perfect live sync.")
    st.caption(SAMPLED_COUNT_CAVEAT)


def render_confidence_trend_compact():
    use_mock = st.session_state.get("use_mock", False)
    if use_mock:
        return
    display_job_id = get_results_view_job_id()
    if not display_job_id:
        return
    detections = fetch_job_detections(display_job_id) or []
    if not detections:
        return
    df = pd.DataFrame(detections)
    if df.empty or "conf" not in df.columns or "frame_index" not in df.columns:
        return
    df = df.copy()
    df["conf"] = pd.to_numeric(df["conf"], errors="coerce").fillna(0.0)
    df["frame_index"] = pd.to_numeric(df["frame_index"], errors="coerce")
    df = df.dropna(subset=["frame_index"])
    if df.empty:
        return
    trend_df = (
        df.groupby("frame_index", as_index=False)
        .agg(avg_conf=("conf", "mean"))
        .sort_values("frame_index")
    )
    if trend_df.empty:
        return
    trend_df["avg_conf_pct"] = trend_df["avg_conf"] * 100.0
    trend_df["avg_conf_smooth"] = (
        trend_df["avg_conf_pct"]
        .rolling(window=max(5, min(25, len(trend_df) // 16 if len(trend_df) > 0 else 5)), center=True, min_periods=1)
        .mean()
    )
    max_points = 180
    if len(trend_df) > max_points:
        step = int(np.ceil(len(trend_df) / max_points))
        trend_df = trend_df.iloc[::step].reset_index(drop=True)

    line = (
        alt.Chart(trend_df)
        .mark_line(color="#38bdf8", strokeWidth=2.2)
        .encode(
            x=alt.X("frame_index:Q", axis=alt.Axis(title="Frame", labelAngle=0)),
            y=alt.Y("avg_conf_smooth:Q", axis=alt.Axis(title="Confidence (%)", labelAngle=0)),
            tooltip=[
                alt.Tooltip("frame_index:Q", title="Frame"),
                alt.Tooltip("avg_conf_smooth:Q", title="Smoothed confidence (%)", format=".1f"),
            ],
        )
        .properties(height=180)
        .configure_view(strokeOpacity=0)
    )
    st.markdown("##### Confidence trend")
    st.altair_chart(line, use_container_width=True)

def individual_spotlight(show_header: bool = True):
    if show_header:
        preview_section_header("Frame inspection", "Who appears in the selected sampled frame checkpoint.")
    display_job_id = get_results_view_job_id()
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

    frame_df["track_id"] = pd.to_numeric(frame_df.get("track_id"), errors="coerce")
    if frame_df["track_id"].isna().all():
        frame_df["track_id"] = np.arange(1, len(frame_df) + 1)
    else:
        frame_df["track_id"] = frame_df["track_id"].fillna(0).astype(int)
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


def render_unique_identity_cards(display_job_id: str, fps: float | None = None) -> None:
    detections = fetch_job_detections(display_job_id) or []
    if not detections:
        st.caption("No tracking identities available for this run.")
        return

    df = pd.DataFrame(detections)
    if df.empty or "track_id" not in df.columns:
        st.caption("No tracking identities available for this run.")
        return

    df["track_id"] = pd.to_numeric(df["track_id"], errors="coerce")
    df = df[df["track_id"].notna() & (df["track_id"] > 0)].copy()
    if df.empty:
        st.caption("No stable identity tracks were produced for this run.")
        return

    if "frame_index" in df.columns:
        df["frame_index"] = pd.to_numeric(df["frame_index"], errors="coerce")
    df["conf"] = pd.to_numeric(df.get("conf"), errors="coerce").fillna(0.0)
    df["age_label"] = df.get("cls_label", "Unknown").apply(normalize_age_class_label)

    cards: list[dict] = []
    for track_id, group in df.groupby("track_id"):
        age_mode = group["age_label"].mode()
        age_label = age_mode.iloc[0] if not age_mode.empty else "Unknown"
        avg_conf = float(group["conf"].mean()) if not group.empty else 0.0
        first_seen_txt = "-"
        last_seen_txt = "-"
        if "frame_index" in group.columns and group["frame_index"].notna().any():
            first_frame = int(group["frame_index"].min())
            last_frame = int(group["frame_index"].max())
            if fps and fps > 0:
                first_seen_txt = format_seconds(first_frame / fps)
                last_seen_txt = format_seconds(last_frame / fps)
            else:
                first_seen_txt = str(first_frame)
                last_seen_txt = str(last_frame)
        cards.append(
            {
                "track_id": int(track_id),
                "age_label": age_label,
                "avg_conf": avg_conf,
                "first_seen": first_seen_txt,
                "last_seen": last_seen_txt,
            }
        )

    cards = sorted(cards, key=lambda row: row["track_id"])
    st.caption(f"{len(cards)} unique elephants tracked across the run")
    cols = st.columns(2, gap="small")
    for idx, row in enumerate(cards):
        with cols[idx % 2]:
            st.markdown(
                f"""
                <div style="border:1px solid rgba(148,163,184,0.14); border-radius:14px; padding:0.78rem 0.82rem; background:rgba(15,23,42,0.12); min-height:128px;">
                    <p style="margin:0; color:#8ea4bb; font-size:0.7rem; letter-spacing:0.06em; text-transform:uppercase; font-weight:700;">ID {row['track_id']:02d}</p>
                    <p style="margin:0.24rem 0 0 0; color:#f8fafc; font-size:1.02rem; font-weight:790;">{row['age_label']}</p>
                    <p style="margin:0.26rem 0 0 0; color:#cbd5e1; font-size:0.84rem;">Avg confidence: {row['avg_conf'] * 100:.1f}%</p>
                    <p style="margin:0.18rem 0 0 0; color:#94a3b8; font-size:0.8rem;">Seen {row['first_seen']} to {row['last_seen']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_review_workspace():
    st.markdown(
        """
        <style>
        .ea-inspect-card {
            border:1px solid rgba(148,163,184,0.14);
            border-radius:16px;
            padding:0.95rem 0.96rem;
            background:linear-gradient(180deg, rgba(15,23,42,0.18), rgba(2,6,23,0.10));
        }
        .ea-inspect-card.emphasis {
            border-color:rgba(34,211,238,0.22);
            background:linear-gradient(180deg, rgba(8,47,73,0.22), rgba(2,6,23,0.12));
            box-shadow:inset 0 1px 0 rgba(125,211,252,0.05);
        }
        .ea-inspect-title { margin:0; color:#f8fafc; font-size:1rem; font-weight:800; }
        .ea-inspect-copy { margin:0.22rem 0 0 0; color:#94a3b8; font-size:0.83rem; line-height:1.45; }
        .ea-inspect-grid {
            margin-top:0.82rem;
            padding-top:0.72rem;
            border-top:1px solid rgba(148,163,184,0.12);
            display:grid;
            grid-template-columns:1fr 1fr;
            gap:0.7rem 1rem;
        }
        .ea-inspect-item .label {
            margin:0;
            color:#7f91a8;
            font-size:0.7rem;
            letter-spacing:0.05em;
            text-transform:uppercase;
            font-weight:700;
        }
        .ea-inspect-item .value {
            margin:0.2rem 0 0 0;
            color:#f8fafc;
            font-size:0.98rem;
            font-weight:730;
            line-height:1.28;
        }
        .ea-inspect-item.accent .value { color:#f59e0b; font-weight:760; }
        .ea-inspect-item.strong .value { color:#ffffff; font-weight:780; }
        .ea-inspect-item.feature .value { color:#dff7ff; font-weight:790; }
        .ea-inspect-foot {
            margin:1rem 0 0 0;
            padding-top:0.78rem;
            border-top:1px solid rgba(148,163,184,0.10);
            color:#94a3b8;
            font-size:0.82rem;
            line-height:1.42;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    display_job_id = get_results_view_job_id()
    use_mock = st.session_state.get("use_mock", False)
    video_info = _resolve_reference_video_info(display_job_id, use_mock)
    video_path = str(video_info.get("path") or "")
    source_video_path = resolve_job_video_path(display_job_id) if (display_job_id and not use_mock) else None
    representative_frame = _resolve_representative_frame_path(display_job_id, use_mock)

    bundle = get_results_bundle(display_job_id) if (display_job_id and not use_mock) else {}
    trend_bundle = (bundle or {}).get("trend", {})
    review_bundle = (bundle or {}).get("review", {})
    herd_df = _get_checkpoint_series(display_job_id, use_mock)
    metrics = fetch_job_metrics(display_job_id) if (display_job_id and not use_mock) else {}

    sampled_visible_count = int(review_bundle.get("sampled_visible_count", 0) or 0)
    avg_sampled_visible_count = float(trend_bundle.get("avg_sampled_visible_count", 0.0) or 0.0)
    dominant_habitat = str((metrics or {}).get("dominant_habitat") or "-")
    preview_is_video = True if use_mock else resolve_job_is_video(display_job_id, metrics=metrics)
    st.session_state["preview_is_video"] = preview_is_video

    processing_time_txt = "-"
    video_duration_txt = "-"
    duration_seconds = None
    processing_seconds = None
    video_fps = None
    if display_job_id and not use_mock:
        jobs_meta = fetch_jobs_from_backend()
        if jobs_meta is not None and not jobs_meta.empty:
            hit = jobs_meta[jobs_meta["Job ID"].astype(str) == str(display_job_id)]
            if not hit.empty:
                proc_sec = pd.to_numeric(hit.iloc[0].get("Duration (s)"), errors="coerce")
                if pd.notna(proc_sec):
                    processing_seconds = float(proc_sec)
                    processing_time_txt = format_duration(float(proc_sec)) or "-"
        if source_video_path:
            cap = cv2.VideoCapture(str(source_video_path))
            fps_value = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            cap.release()
            if fps_value > 0:
                video_fps = fps_value
            secs = media_duration_seconds(source_video_path)
            if secs is not None:
                duration_seconds = float(secs)
                video_duration_txt = format_seconds(secs)
    processing_ratio_txt = "-"
    if (
        preview_is_video
        and duration_seconds is not None
        and processing_seconds is not None
        and duration_seconds > 0
    ):
        processing_ratio_txt = f"{(processing_seconds / duration_seconds):.2f}x"
    model_txt = str(st.session_state.get("last_model_choice", "-") or "-")
    input_type_txt = "Video" if preview_is_video else "Image"
    display_job_label = format_display_job_id(display_job_id, input_type_txt)
    active_mode_key = str(st.session_state.get("last_pipeline_mode_key", "") or "").strip().lower()
    is_identity_mode = active_mode_key in {"quality", "identity_accurate", "original"}
    unique_elephants_txt = str(int((metrics or {}).get("unique_tracks", 0) or 0))

    # paused-frame analysis state
    analysis_key = f"paused_frame_analysis_{display_job_id or 'mock'}"
    if analysis_key not in st.session_state:
        default_frame_idx = None
        default_time_sec = None
        default_count = sampled_visible_count
        if not herd_df.empty:
            row = herd_df.iloc[-1]
            try:
                default_frame_idx = int(row.get("frame_index", 0))
            except Exception:
                default_frame_idx = None
            ts = row.get("time_seconds")
            if pd.notna(ts):
                default_time_sec = float(ts)
            sampled = row.get("sampled_visible_count")
            if not pd.isna(sampled):
                default_count = int(sampled)
        st.session_state[analysis_key] = {
            "paused_seconds": default_time_sec if default_time_sec is not None else 0.0,
            "resolved_time_txt": format_seconds(default_time_sec) if default_time_sec is not None else "-",
            "frame_index": default_frame_idx,
            "count": default_count,
            "avg_conf": None,
            "age_mix": "-",
        }

    frame_files: list[Path] = []
    selected_frame_path = representative_frame
    frame_indices: list[int] = []
    slider_key = f"review_workspace_scrub_frame_{display_job_id or 'mock'}"
    slider_pending_key = f"review_workspace_scrub_frame_pending_{display_job_id or 'mock'}"
    paused_key = f"paused_seconds_input_{display_job_id or 'mock'}"
    if display_job_id and not use_mock:
        frames_dir = DATA_DIR / "outputs" / "frames" / str(display_job_id)
        frame_files = sorted(frames_dir.glob("frame_*.jpg")) if frames_dir.exists() else []
        frame_indices = [idx for idx in [frame_index_from_path(fp) for fp in frame_files] if idx is not None]
    if frame_files:
        if display_job_id and paused_key not in st.session_state:
            default_frame_idx = frame_indices[0] if frame_indices else 0
            if video_fps and video_fps > 0:
                st.session_state[paused_key] = round(default_frame_idx / video_fps, 2)
        if slider_key not in st.session_state:
            default_selected_frame = int(st.session_state.get("preview_frame_index") or (frame_indices[0] if frame_indices else 0))
            if frame_indices:
                default_selected_frame = min(frame_indices, key=lambda frame: abs(frame - default_selected_frame))
            st.session_state[slider_key] = default_selected_frame
        if slider_pending_key in st.session_state:
            pending_frame = int(st.session_state.pop(slider_pending_key))
            if frame_indices:
                pending_frame = min(frame_indices, key=lambda frame: abs(frame - pending_frame))
            st.session_state[slider_key] = pending_frame
            st.session_state["preview_frame_index"] = pending_frame
        selected_scrub_frame = int(st.session_state.get(slider_key, frame_indices[0] if frame_indices else 0))
        st.session_state["preview_frame_index"] = int(selected_scrub_frame)
        nearest_path = _resolve_frame_path_for_index(display_job_id, int(selected_scrub_frame))
        if nearest_path:
            selected_frame_path = nearest_path
    elif representative_frame:
        st.session_state["preview_frame_index"] = frame_index_from_path(Path(representative_frame))

    analysis = st.session_state.get(analysis_key, {})
    current_preview_idx = int(st.session_state.get("preview_frame_index") or 0)
    current_preview_rgb = None
    analyzed_preview_rgb = None
    if preview_is_video and source_video_path:
        dets = fetch_job_detections(display_job_id) or []
        selected_frame_dets = []
        if dets:
            det_df = pd.DataFrame(dets)
            if "frame_index" in det_df.columns:
                det_df["frame_index"] = pd.to_numeric(det_df["frame_index"], errors="coerce")
                selected_frame_dets = det_df[det_df["frame_index"] == float(current_preview_idx)].to_dict("records")
        raw_frame = _load_video_frame(source_video_path, current_preview_idx)
        if raw_frame is not None:
            current_preview_rgb = _draw_yellow_detection_card(raw_frame, selected_frame_dets)
            if (
                bool(analysis.get("show_analysis_frame"))
                and analysis.get("frame_index") is not None
                and int(analysis.get("frame_index")) == current_preview_idx
            ):
                analyzed_preview_rgb = current_preview_rgb

    def _run_paused_frame_analysis(paused_value: float) -> None:
        resolved_frame_idx = int(st.session_state.get("preview_frame_index") or 0)
        resolved_time_txt = format_seconds(paused_value)
        resolved_count = "-"
        avg_conf_txt = "-"
        age_mix_txt = "-"

        if display_job_id and not use_mock and resolved_frame_idx is not None:
            dets = fetch_job_detections(display_job_id) or []
            if dets:
                det_df = pd.DataFrame(dets)
                if "frame_index" in det_df.columns:
                    det_df["frame_index"] = pd.to_numeric(det_df["frame_index"], errors="coerce")
                    frame_df = det_df[det_df["frame_index"] == float(resolved_frame_idx)].copy()
                    if frame_df.empty and det_df["frame_index"].notna().any():
                        nearest_label = (det_df["frame_index"] - float(resolved_frame_idx)).abs().idxmin()
                        nearest_val = int(det_df.loc[nearest_label]["frame_index"])
                        frame_df = det_df[det_df["frame_index"] == float(nearest_val)].copy()
                        resolved_frame_idx = nearest_val
                    if not frame_df.empty:
                        frame_df["conf"] = pd.to_numeric(frame_df.get("conf"), errors="coerce").fillna(0.0)
                        avg_conf_txt = f"{(frame_df['conf'].mean() * 100):.1f}%"
                        if "cls_label" in frame_df.columns:
                            ages = frame_df["cls_label"].apply(normalize_age_class_label).value_counts()
                            age_mix_txt = ", ".join([f"{k}:{int(v)}" for k, v in ages.items()]) if not ages.empty else "-"
                        resolved_count = int(len(frame_df))

        st.session_state[analysis_key] = {
            "paused_seconds": float(paused_value),
            "resolved_time_txt": resolved_time_txt,
            "frame_index": resolved_frame_idx,
            "count": resolved_count,
            "avg_conf": avg_conf_txt,
            "age_mix": age_mix_txt,
            "show_analysis_frame": True,
        }
        if resolved_frame_idx is not None:
            st.session_state["preview_frame_index"] = int(resolved_frame_idx)

    if preview_is_video and duration_seconds is not None:
        hero_left, hero_right = st.columns([1.65, 1.0], gap="large")
        with hero_left:
            st.markdown("#### Preview frame")
            st.caption("Scrub by timestamp to inspect the run like a player while the live inspector updates on the right.")
            if frame_files:
                selected_scrub_frame = int(st.session_state.get(slider_key, frame_indices[0] if frame_indices else 0))
                resolved_preview_path = _resolve_frame_path_for_index(display_job_id, int(selected_scrub_frame))
                if resolved_preview_path:
                    selected_frame_path = resolved_preview_path

            st.markdown(
                '<p style="margin:0 0 0.2rem 0; color:#7f91a8; font-size:0.72rem; letter-spacing:0.05em; text-transform:uppercase; font-weight:700;">Player view</p>',
                unsafe_allow_html=True,
            )
            if current_preview_rgb is not None:
                st.image(current_preview_rgb, caption="Current review frame", width="stretch")
            elif analyzed_preview_rgb is not None:
                st.image(analyzed_preview_rgb, caption="Analyzed paused frame", width="stretch")
            elif selected_frame_path:
                st.image(selected_frame_path, caption="Current review frame", width="stretch")
            elif use_mock:
                st.info("Demo mode: reference playback media is not attached.")
            else:
                st.info("No visual evidence available for this run.")
            if frame_files:
                st.markdown(
                    '<p style="margin:0.38rem 0 0.12rem 0; color:#8ea4bb; font-size:0.72rem; letter-spacing:0.06em; text-transform:uppercase; font-weight:700;">Timeline</p>',
                    unsafe_allow_html=True,
                )
                selected_scrub_frame = st.select_slider(
                    "Timeline",
                    options=frame_indices,
                    value=int(st.session_state.get(slider_key, frame_indices[0] if frame_indices else 0)),
                    format_func=(lambda frame: format_seconds(frame / video_fps) if video_fps and video_fps > 0 else format_seconds(frame)),
                    key=slider_key,
                    on_change=_sync_video_review_selection if (display_job_id and frame_indices) else None,
                    args=(str(display_job_id), frame_indices, video_fps, "slider") if (display_job_id and frame_indices) else None,
                )
                st.session_state["preview_frame_index"] = int(selected_scrub_frame)
                resolved_preview_path = _resolve_frame_path_for_index(display_job_id, int(selected_scrub_frame))
                if resolved_preview_path:
                    selected_frame_path = resolved_preview_path
                current_selected_time = float(st.session_state.get(paused_key, 0.0) or 0.0)
                t1, t2, t3 = st.columns([1, 1.2, 1])
                with t1:
                    st.markdown(f'<p style="margin:0; color:#7f91a8; font-size:0.84rem;">{format_seconds(0)}</p>', unsafe_allow_html=True)
                with t2:
                    st.markdown(
                        f'<p style="margin:0; color:#dbeafe; font-size:0.84rem; font-weight:700; text-align:center;">Current timestamp: {format_seconds(current_selected_time)}</p>',
                        unsafe_allow_html=True,
                    )
                with t3:
                    st.markdown(
                        f'<p style="margin:0; color:#7f91a8; font-size:0.84rem; text-align:right;">{format_seconds(duration_seconds or current_selected_time)}</p>',
                        unsafe_allow_html=True,
                    )

        with hero_right:
            if not is_identity_mode:
                analysis = st.session_state.get(analysis_key, {})
                st.markdown(
                    """
                    <div style="border:1px solid rgba(34,211,238,0.22); border-radius:16px; padding:0.95rem 0.96rem; background:linear-gradient(180deg, rgba(8,47,73,0.26), rgba(2,6,23,0.14)); box-shadow:inset 0 1px 0 rgba(125,211,252,0.05);">
                        <p style="margin:0; color:#f8fafc; font-size:1rem; font-weight:800;">Live inspector</p>
                        <p style="margin:0.24rem 0 0 0; color:#94a3b8; font-size:0.83rem; line-height:1.45;">Track the selected moment, nudge time precisely, and analyze the current frame without leaving this panel.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown("")
                st.markdown(
                    '<p style="margin:0 0 0.4rem 0; color:#cbd5e1; font-size:0.84rem; font-weight:720;">Selected-frame status</p>',
                    unsafe_allow_html=True,
                )
                i1, i2 = st.columns(2, gap="small")
                with i1:
                    _review_stat("Selected time", format_seconds(float(st.session_state.get(paused_key, 0.0) or 0.0)))
                with i2:
                    _review_stat("Analyzed time", str(analysis.get("resolved_time_txt", "-")))
                i3, i4 = st.columns(2, gap="small")
                with i3:
                    _review_stat("Elephant count", str(analysis.get("count", sampled_visible_count)))
                with i4:
                    _review_stat("Avg confidence", str(analysis.get("avg_conf", "-")))
                i5, i6 = st.columns(2, gap="small")
                with i5:
                    _review_stat("Age mix", str(analysis.get("age_mix", "-")))
                with i6:
                    _review_stat("Habitat", dominant_habitat)

                max_seconds = duration_seconds if duration_seconds > 0 else 7200.0
                text_paused_key = f"paused_seconds_text_{display_job_id or 'mock'}"
                if text_paused_key not in st.session_state:
                    st.session_state[text_paused_key] = str(
                        round(float((st.session_state.get(analysis_key) or {}).get("paused_seconds", 0.0) or 0.0), 2)
                    )
                st.markdown("")
                st.markdown(
                    """
                    <div style="border:1px solid rgba(148,163,184,0.14); border-radius:14px; padding:0.82rem 0.86rem; background:linear-gradient(180deg, rgba(15,23,42,0.18), rgba(2,6,23,0.10));">
                        <p style="margin:0; color:#f8fafc; font-size:0.9rem; font-weight:770;">Time controls</p>
                        <p style="margin:0.18rem 0 0 0; color:#94a3b8; font-size:0.81rem; line-height:1.4;">Enter a precise second, then run the current-frame inspection.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                paused_value_raw = st.text_input(
                    "Selected time (seconds)",
                    key=text_paused_key,
                    help=f"Enter a value between 0 and {max_seconds:.2f} seconds.",
                )
                if st.button("Analyze current frame", use_container_width=True, key=f"analyze_paused_frame_{display_job_id or 'mock'}"):
                    try:
                        parsed_paused = float(str(paused_value_raw).strip())
                    except Exception:
                        st.error("Enter a valid time in seconds before analyzing.")
                        parsed_paused = None
                    if parsed_paused is not None:
                        original_paused = parsed_paused
                        parsed_paused = min(float(max_seconds), max(0.0, parsed_paused))
                        if original_paused != parsed_paused:
                            st.warning(f"Selected time is outside the video duration. Using {parsed_paused:.2f}s instead.")
                        st.session_state[paused_key] = parsed_paused
                        if display_job_id and frame_indices:
                            if video_fps and video_fps > 0:
                                target_frame = int(round(parsed_paused * video_fps))
                            else:
                                target_frame = int(round(parsed_paused))
                            frame_idx = int(min(frame_indices, key=lambda frame: abs(frame - target_frame)))
                            st.session_state[slider_pending_key] = frame_idx
                            st.session_state["preview_frame_index"] = frame_idx
                            st.session_state[paused_key] = round(frame_idx / video_fps, 2) if video_fps and video_fps > 0 else parsed_paused
                        _run_paused_frame_analysis(parsed_paused)
                        st.rerun()

            st.markdown("")
            st.markdown("#### Run overview")
            overview_copy = "Tracking-aware context for the current identity-accurate review pass." if is_identity_mode else "Core run context for the current video inspection pass."
            st.caption(overview_copy)
            st.markdown(
                f"""
                <div style="border:1px solid rgba(148,163,184,0.12); border-radius:16px; padding:0.95rem 0.96rem; background:linear-gradient(180deg, rgba(15,23,42,0.16), rgba(2,6,23,0.08));">
                    <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.7rem 1rem;">
                        <div>
                            <p style="margin:0; color:#7f91a8; font-size:0.7rem; letter-spacing:0.05em; text-transform:uppercase; font-weight:700;">Video duration</p>
                            <p style="margin:0.2rem 0 0 0; color:#f8fafc; font-size:0.97rem; font-weight:730;">{video_duration_txt}</p>
                        </div>
                        <div>
                            <p style="margin:0; color:#7f91a8; font-size:0.7rem; letter-spacing:0.05em; text-transform:uppercase; font-weight:700;">Processing time</p>
                            <p style="margin:0.2rem 0 0 0; color:#ffffff; font-size:0.97rem; font-weight:760;">{processing_time_txt}</p>
                        </div>
                        <div>
                            <p style="margin:0; color:#7f91a8; font-size:0.7rem; letter-spacing:0.05em; text-transform:uppercase; font-weight:700;">Processing ratio</p>
                            <p style="margin:0.2rem 0 0 0; color:#f8fafc; font-size:0.97rem; font-weight:730;">{processing_ratio_txt}</p>
                        </div>
                        <div>
                            <p style="margin:0; color:#7f91a8; font-size:0.7rem; letter-spacing:0.05em; text-transform:uppercase; font-weight:700;">Model</p>
                            <p style="margin:0.2rem 0 0 0; color:#f8fafc; font-size:0.97rem; font-weight:730;">{model_txt}</p>
                        </div>
                        <div>
                            <p style="margin:0; color:#7f91a8; font-size:0.7rem; letter-spacing:0.05em; text-transform:uppercase; font-weight:700;">Job label</p>
                            <p style="margin:0.2rem 0 0 0; color:#f8fafc; font-size:0.97rem; font-weight:730;">{display_job_label}</p>
                        </div>
                        <div>
                            <p style="margin:0; color:#7f91a8; font-size:0.7rem; letter-spacing:0.05em; text-transform:uppercase; font-weight:700;">Avg sampled visible count</p>
                            <p style="margin:0.2rem 0 0 0; color:#e2e8f0; font-size:0.97rem; font-weight:730;">{avg_sampled_visible_count:.2f}</p>
                        </div>
                    </div>
                    <div style="margin-top:0.82rem; padding-top:0.72rem; border-top:1px solid rgba(148,163,184,0.14); display:grid; grid-template-columns:1fr 1fr; gap:0.7rem 1rem;">
                        <div>
                            <p style="margin:0; color:#7f91a8; font-size:0.7rem; letter-spacing:0.05em; text-transform:uppercase; font-weight:700;">Dominant habitat</p>
                            <p style="margin:0.2rem 0 0 0; color:#dff7ff; font-size:1.03rem; font-weight:790;">{dominant_habitat}</p>
                        </div>
                        <div>
                            <p style="margin:0; color:#7f91a8; font-size:0.7rem; letter-spacing:0.05em; text-transform:uppercase; font-weight:700;">Input type</p>
                            <p style="margin:0.2rem 0 0 0; color:#f59e0b; font-size:0.97rem; font-weight:760;">{input_type_txt}</p>
                        </div>
                        {"<div><p style='margin:0; color:#7f91a8; font-size:0.7rem; letter-spacing:0.05em; text-transform:uppercase; font-weight:700;'>Total elephants in video</p><p style='margin:0.2rem 0 0 0; color:#ffffff; font-size:1.03rem; font-weight:790;'>" + unique_elephants_txt + "</p></div>" if is_identity_mode else ""}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if is_identity_mode:
                st.markdown("")
                st.markdown("#### Tracked identities")
                st.caption("Unique elephant IDs recovered across the whole run.")
                render_unique_identity_cards(display_job_id, fps=video_fps)
            else:
                st.caption("Sampled values are checkpoints rather than frame-perfect live counts.")
        return

    row1_left, row1_right = st.columns([1.7, 1.0], gap="large", vertical_alignment="top")
    with row1_left:
        st.markdown("#### Preview frame")
        st.caption("Review the current image result and supporting evidence.")
        if selected_frame_path:
            st.image(selected_frame_path, caption="Analyzed image result", width="stretch")
        elif use_mock:
            st.info("Demo mode: reference playback media is not attached.")
        else:
            st.info("No visual evidence available for this run.")

    with row1_right:
        st.markdown("<div style='height:7.25rem;'></div>", unsafe_allow_html=True)
        render_inspection_overview_card(
            "Run overview",
            "Static result context for the current image inspection pass.",
            [
                ("Processing time", processing_time_txt, "strong"),
                ("Model", model_txt, None),
                ("Job label", display_job_label, None),
                ("Avg sampled visible count", f"{avg_sampled_visible_count:.2f}", None),
                ("Dominant habitat", dominant_habitat, "feature"),
                ("Input type", input_type_txt, "accent"),
            ],
        )
        st.caption("Frame-first review is active for this run. Video-specific paused playback controls are hidden.")


def result_quality_section(wrap_cards: bool = True):
    t0 = time.perf_counter()
    preview_section_header("Run diagnostics", "Confidence, observed composition, and sampling caveats.")
    use_mock = st.session_state.get("use_mock", False)
    display_job_id = get_results_view_job_id()
    is_video = st.session_state.get("preview_is_video", True)

    if use_mock:
        st.info("Run one backend job to view quality analytics.")
        record_ui_timing("diagnostics_prep_s", time.perf_counter() - t0)
        return
    if not display_job_id:
        st.info("No processed run selected.")
        record_ui_timing("diagnostics_prep_s", time.perf_counter() - t0)
        return
    bundle = get_results_bundle(display_job_id)
    diagnostics_bundle = (bundle or {}).get("run_diagnostics", {})
    caveats = diagnostics_bundle.get("caveats") or []

    detections = fetch_job_detections(display_job_id) or []
    if not detections:
        st.info("No detection data available for quality analytics.")
        record_ui_timing("diagnostics_prep_s", time.perf_counter() - t0)
        return

    df = pd.DataFrame(detections)
    if df.empty or "conf" not in df.columns:
        st.info("No confidence data available for quality analytics.")
        record_ui_timing("diagnostics_prep_s", time.perf_counter() - t0)
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

    with maybe_card(wrap_cards):
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

    with maybe_card(wrap_cards):
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

    if caveats:
        with maybe_card(wrap_cards):
            st.markdown("#### Sampling caveats")
            for caveat in caveats:
                st.markdown(f"- {caveat}")

    with st.expander("Confidence analysis", expanded=False):
        render_confidence_graph(df, is_video=is_video)
    record_ui_timing("diagnostics_prep_s", time.perf_counter() - t0)


def charts_section(compact: bool = False, wrap_card: bool = True):
    t0 = time.perf_counter()
    if not st.session_state.get("preview_is_video", True):
        record_ui_timing("trend_chart_prep_s", time.perf_counter() - t0)
        return

    use_mock = st.session_state.get("use_mock", False)
    if use_mock:
        herd_df = mock_herd_series()
        herd_df = herd_df.rename(columns={"herd_size": "sampled_visible_count"})
    else:
        job_id = get_results_view_job_id()
        bundle = get_results_bundle(job_id) if job_id else {}
        trend_bundle = (bundle or {}).get("trend", {})
        series = trend_bundle.get("sampled_visible_count_over_time")
        herd_df = pd.DataFrame(series or [])
    with maybe_card(wrap_card):
        header_cols = st.columns([3, 1])
        with header_cols[0]:
            st.markdown("#### Sampled visible count trend")
        with header_cols[1]:
            if use_mock:
                herd_mode = "Time"
            else:
                options = ["Frame"]
                if "time_seconds" in herd_df.columns:
                    options = ["Time", "Frame"]
                if compact:
                    herd_mode = "Time" if "Time" in options else "Frame"
                else:
                    job_id = get_results_view_job_id() or "none"
                    herd_mode = st.radio(
                        "X axis",
                        options,
                        horizontal=True,
                        label_visibility="collapsed",
                        key=f"herd_x_mode_{job_id}",
                    )

        st.caption(f"Smoothed trend from sampled checkpoints. {SAMPLED_COUNT_CAVEAT}")
        if herd_df.empty:
            st.info("No data yet. Enable mock data or process a video.")
            return

        herd_df = herd_df.copy()
        if use_mock:
            herd_df["frame_index"] = np.arange(len(herd_df))
        herd_df = herd_df.sort_values("frame_index").reset_index(drop=True)
        window = max(5, min(25, len(herd_df) // 20 if len(herd_df) > 0 else 5))
        herd_df["herd_smooth"] = herd_df["sampled_visible_count"].rolling(window=window, center=True, min_periods=1).mean()
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
                alt.Tooltip("sampled_visible_count:Q", title="Sampled visible count"),
                alt.Tooltip("herd_smooth:Q", title="Smoothed", format=".2f"),
            ]
        else:
            x_def = alt.X("frame_index:Q", axis=alt.Axis(title="Frame", labelAngle=0))
            tooltips = [
                alt.Tooltip("frame_index:Q", title="Frame"),
                alt.Tooltip("time_seconds:Q", title="Time (s)", format=".1f"),
                alt.Tooltip("sampled_visible_count:Q", title="Sampled visible count"),
                alt.Tooltip("herd_smooth:Q", title="Smoothed", format=".2f"),
            ]

        raw_line = (
            alt.Chart(chart_df)
            .mark_line(opacity=0.35, color="#9db7da")
            .encode(x=x_def, y=alt.Y("sampled_visible_count:Q", axis=alt.Axis(title=None, labelAngle=0)))
        )
        smooth_line = (
            alt.Chart(chart_df)
            .mark_line(color="#e6f0ff", strokeWidth=2.6)
            .encode(x=x_def, y=alt.Y("herd_smooth:Q", axis=alt.Axis(title=None, labelAngle=0)), tooltip=tooltips)
        )
        chart_height = 240 if compact else 320
        st.altair_chart((raw_line + smooth_line).properties(height=chart_height).configure_view(strokeOpacity=0), use_container_width=True)

        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("Min sampled visible count", int(herd_df["sampled_visible_count"].min()))
        with k2:
            st.metric("Avg sampled visible count", f"{float(herd_df['sampled_visible_count'].mean()):.2f}")
        with k3:
            st.metric("Peak sampled visible count", int(herd_df["sampled_visible_count"].max()))

        if not compact:
            compact_df = (
                herd_df.groupby("sampled_visible_count", as_index=False)
                .size()
                .rename(columns={"size": "Samples", "sampled_visible_count": "Sampled visible count"})
                .sort_values("Sampled visible count")
            )
            st.markdown("##### Sampled visible count distribution")
            st.dataframe(compact_df, use_container_width=True, hide_index=True)
    record_ui_timing("trend_chart_prep_s", time.perf_counter() - t0)


def timeline_section():
    t0 = time.perf_counter()
    use_mock = st.session_state.get("use_mock", False)
    job_id = get_results_view_job_id()
    if not st.session_state.get("preview_is_video", True) and not use_mock:
        record_ui_timing("timeline_prep_s", time.perf_counter() - t0)
        return
    tracking = get_tracking_bundle(job_id)
    tracking_available = bool((tracking.get("availability") or {}).get("has_tracking_rows"))
    if not tracking_available and not use_mock:
        st.info("Tracking-aware timeline is not available for this run yet. Fast review remains available.")
        record_ui_timing("timeline_prep_s", time.perf_counter() - t0)
        return

    rows: list[dict] = []
    if use_mock:
        df = mock_timeline_table()
    else:
        try:
            resp = requests.get(f"{BACKEND_BASE_URL}/timeline/{job_id}", timeout=10)
            resp.raise_for_status()
            rows = resp.json().get("timeline", [])
        except Exception:
            rows = []
        if not rows:
            st.info("No timeline rows available for this run.")
            record_ui_timing("timeline_prep_s", time.perf_counter() - t0)
            return
        df = pd.DataFrame(rows).rename(
            columns={
                "elephant_id": "Elephant ID",
                "age_class": "Age class",
                "first_seen": "First seen (frame)",
                "last_seen": "Last seen (frame)",
                "duration_frames": "Duration (frames)",
            }
        )

    if "Age class" in df.columns:
        df["Age class"] = df["Age class"].apply(normalize_age_class_label)
    st.markdown("#### Timeline")
    age_filter = st.selectbox(
        "Age class filter",
        ["All", "Adult", "Juvenile", "Baby"],
        index=0,
        key=f"results_timeline_age_{job_id or 'mock'}",
    )
    if age_filter != "All" and "Age class" in df.columns:
        df = df[df["Age class"] == age_filter]
    st.dataframe(df, use_container_width=True, hide_index=True)
    record_ui_timing("timeline_prep_s", time.perf_counter() - t0)


def habitat_section():
    t0 = time.perf_counter()
    use_mock = st.session_state.get("use_mock", False)
    job_id = get_results_view_job_id()
    st.markdown("#### Habitat")
    if use_mock:
        dist = pd.DataFrame(
            [
                {"habitat": "Grassland", "percentage": 48, "count": 18, "avg_confidence": 0.84},
                {"habitat": "Oil Palm Plantation", "percentage": 32, "count": 12, "avg_confidence": 0.79},
                {"habitat": "Field Edge", "percentage": 14, "count": 5, "avg_confidence": 0.74},
                {"habitat": "Forest", "percentage": 6, "count": 2, "avg_confidence": 0.71},
            ]
        )
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Dominant habitat", "Grassland")
        k2.metric("Time in dominant", "48%")
        k3.metric("Habitat switches", "6")
        k4.metric("Coverage", "100%")
        with st.expander("Habitat charts", expanded=True):
            left, right = st.columns([1.2, 1])
            with left:
                habitat_bar = (
                    alt.Chart(dist)
                    .mark_bar(cornerRadiusEnd=7, color="#5eead4")
                    .encode(
                        x=alt.X("percentage:Q", axis=alt.Axis(title="Coverage (%)", labelAngle=0)),
                        y=alt.Y("habitat:N", axis=alt.Axis(title=None, labelAngle=0), sort="-x"),
                    )
                    .properties(height=180)
                    .configure_view(strokeOpacity=0)
                )
                st.markdown("##### Habitat coverage")
                st.altair_chart(habitat_bar, use_container_width=True)
            with right:
                timeline_df = pd.DataFrame(
                    [
                        {"start": 0, "end": 22, "habitat": "Grassland", "confidence": 0.86},
                        {"start": 23, "end": 34, "habitat": "Oil Palm Plantation", "confidence": 0.79},
                        {"start": 35, "end": 41, "habitat": "Field Edge", "confidence": 0.74},
                        {"start": 42, "end": 44, "habitat": "Grassland", "confidence": 0.82},
                    ]
                )
                timeline_chart = (
                    alt.Chart(timeline_df)
                    .mark_bar(cornerRadius=4)
                    .encode(
                        x=alt.X("start:Q", axis=alt.Axis(title="Frame index", labelAngle=0)),
                        x2="end:Q",
                        y=alt.Y("habitat:N", axis=alt.Axis(title=None, labelAngle=0)),
                        color=alt.Color("habitat:N", legend=None),
                    )
                    .properties(height=180)
                    .configure_view(strokeOpacity=0)
                )
                st.markdown("##### Habitat transition view")
                st.altair_chart(timeline_chart, use_container_width=True)
        with st.expander("Habitat summary data", expanded=False):
            st.dataframe(dist.rename(columns={"habitat": "Habitat", "percentage": "Time (%)", "count": "Samples", "avg_confidence": "Avg confidence"}), use_container_width=True, hide_index=True)
        record_ui_timing("habitat_prep_s", time.perf_counter() - t0)
        return

    metrics = fetch_job_metrics(job_id) if job_id else None
    habitat_distribution = (metrics or {}).get("habitat_distribution", [])
    habitat_timeline = (metrics or {}).get("habitat_timeline", [])
    if not habitat_distribution:
        st.info("Habitat summary is not available for this run.")
        record_ui_timing("habitat_prep_s", time.perf_counter() - t0)
        return
    dist_df = pd.DataFrame(habitat_distribution).rename(
        columns={
            "habitat": "Habitat",
            "percentage": "Time (%)",
            "avg_confidence": "Avg confidence",
            "count": "Samples",
        }
    )
    dominant_habitat = str((metrics or {}).get("dominant_habitat") or "-")
    dominant_share = float((metrics or {}).get("dominant_habitat_share", 0.0) or 0.0)
    habitat_switches = int((metrics or {}).get("habitat_switches", 0) or 0)
    habitat_coverage = float((metrics or {}).get("habitat_coverage", 0.0) or 0.0)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Dominant habitat", dominant_habitat)
    k2.metric("Time in dominant", f"{dominant_share:.0f}%")
    k3.metric("Habitat switches", habitat_switches)
    k4.metric("Coverage", f"{habitat_coverage:.0f}%")
    chart_df = pd.DataFrame(habitat_distribution).copy()
    with st.expander("Habitat charts", expanded=True):
        if not chart_df.empty:
            chart_df["label"] = chart_df["habitat"].astype(str)
            chart_df["percentage"] = pd.to_numeric(chart_df["percentage"], errors="coerce").fillna(0.0)
            left, right = st.columns([1.2, 1])
            with left:
                habitat_bar = (
                    alt.Chart(chart_df)
                    .mark_bar(cornerRadiusEnd=7, color="#5eead4")
                    .encode(
                        x=alt.X("percentage:Q", axis=alt.Axis(title="Coverage (%)", labelAngle=0)),
                        y=alt.Y("label:N", axis=alt.Axis(title=None, labelAngle=0), sort="-x"),
                        tooltip=[
                            alt.Tooltip("label:N", title="Habitat"),
                            alt.Tooltip("percentage:Q", title="Coverage (%)", format=".1f"),
                            alt.Tooltip("count:Q", title="Samples"),
                            alt.Tooltip("avg_confidence:Q", title="Avg confidence", format=".2f"),
                        ],
                    )
                    .properties(height=max(180, 48 * len(chart_df)))
                    .configure_view(strokeOpacity=0)
                )
                st.markdown("##### Habitat coverage")
                st.altair_chart(habitat_bar, use_container_width=True)

            with right:
                if habitat_timeline:
                    timeline_df = pd.DataFrame(habitat_timeline)
                    if not timeline_df.empty:
                        timeline_chart = (
                            alt.Chart(timeline_df)
                            .mark_bar(cornerRadius=4)
                            .encode(
                                x=alt.X("start:Q", axis=alt.Axis(title="Frame index", labelAngle=0)),
                                x2="end:Q",
                                y=alt.Y("habitat:N", axis=alt.Axis(title=None, labelAngle=0)),
                                color=alt.Color("habitat:N", legend=None),
                                tooltip=[
                                    alt.Tooltip("habitat:N", title="Habitat"),
                                    alt.Tooltip("start:Q", title="Start"),
                                    alt.Tooltip("end:Q", title="End"),
                                    alt.Tooltip("confidence:Q", title="Confidence", format=".2f"),
                                ],
                            )
                            .properties(height=max(180, 42 * timeline_df["habitat"].nunique()))
                            .configure_view(strokeOpacity=0)
                        )
                        st.markdown("##### Habitat transition view")
                        st.altair_chart(timeline_chart, use_container_width=True)
                else:
                    st.info("No habitat transition view is available for this run.")

    with st.expander("Habitat summary data", expanded=False):
        st.dataframe(dist_df, use_container_width=True, hide_index=True)
    record_ui_timing("habitat_prep_s", time.perf_counter() - t0)


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
    t0 = time.perf_counter()
    use_mock = st.session_state.get("use_mock", False)
    preview_is_video = st.session_state.get("preview_is_video", True)
    job_id = get_results_view_job_id()
    points_df = None
    if job_id:
        points_df = cached_gps_points(job_id)
    if points_df is not None:
        points_df = downsample_points(points_df, max_points=200)
    if (points_df is None or points_df.empty) and not use_mock:
        st.markdown("#### Detection map")
        if preview_is_video:
            st.caption("GPS output is not available for this run, so the map is hidden.")
        else:
            st.caption("Image/frame-first run without GPS context. Map is hidden to keep the page focused on visual evidence.")
        record_ui_timing("map_prep_s", time.perf_counter() - t0)
        return

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
    record_ui_timing("map_prep_s", time.perf_counter() - t0)

def upload_jobs_page():
    st.markdown(
        """
        <style>
        .ea-ws-label { margin:0; color:#94a3b8; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.08em; font-weight:700; }
        .ea-ws-v { margin:0.08rem 0 0 0; color:#f8fafc; font-size:1.05rem; font-weight:720; line-height:1.2; }
        .ea-ws-muted { color:#9fb1c7; font-size:0.85rem; }
        .ea-upload-step { margin:0 0 0.85rem 0; color:#f8fafc; font-size:1.02rem; font-weight:760; }
        .ea-upload-lock {
            border: 1px solid rgba(245,158,11,0.28);
            border-radius: 14px;
            padding: 0.82rem 0.88rem;
            background: linear-gradient(180deg, rgba(245,158,11,0.09), rgba(15,23,42,0.14));
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
        }
        .ea-upload-lock .name { margin:0.16rem 0 0 0; color:#f8fafc; font-size:1.04rem; font-weight:760; line-height:1.2; }
        .ea-upload-lock .meta { margin:0.12rem 0 0 0; color:#aab8cb; font-size:0.84rem; }
        .ea-upload-note {
            border:1px solid rgba(96,165,250,0.16);
            border-radius:12px;
            padding:0.72rem 0.8rem;
            background:rgba(37,99,235,0.10);
            color:#b8d3ff;
            font-size:0.88rem;
        }
        .ea-compare-card {
            border:1px solid rgba(148,163,184,0.18);
            border-radius:14px;
            padding:0.82rem 0.88rem;
            background:linear-gradient(180deg, rgba(15,23,42,0.26), rgba(2,6,23,0.16));
            min-height:138px;
        }
        .ea-compare-card.active {
            border-color: rgba(94,234,212,0.62);
            box-shadow: 0 0 0 1px rgba(94,234,212,0.14) inset, 0 10px 24px rgba(8,47,73,0.18);
            background: linear-gradient(180deg, rgba(12,74,110,0.18), rgba(15,23,42,0.16));
        }
        .ea-compare-card .topline { display:flex; justify-content:space-between; align-items:center; gap:0.6rem; }
        .ea-compare-card .title { margin:0; color:#f8fafc; font-size:1rem; font-weight:790; letter-spacing:-0.01em; }
        .ea-compare-card .tag {
            display:inline-block;
            padding:0.18rem 0.52rem;
            border-radius:999px;
            font-size:0.67rem;
            letter-spacing:0.06em;
            text-transform:uppercase;
            font-weight:800;
            color:#ecfeff;
            background:rgba(14,165,233,0.26);
            border:1px solid rgba(125,211,252,0.24);
            white-space:nowrap;
        }
        .ea-compare-card .purpose { margin:0.42rem 0 0 0; color:#e2e8f0; font-size:0.92rem; font-weight:650; }
        .ea-compare-card .note { margin:0.34rem 0 0 0; color:#94a3b8; font-size:0.84rem; line-height:1.38; }
        .ea-mode-summary {
            border:1px solid rgba(148,163,184,0.18);
            border-radius:14px;
            padding:0.84rem 0.9rem;
            background:linear-gradient(180deg, rgba(15,23,42,0.26), rgba(2,6,23,0.16));
            margin-top:0.48rem;
        }
        .ea-mode-summary .eyebrow { margin:0; color:#f59e0b; font-size:0.7rem; letter-spacing:0.08em; text-transform:uppercase; font-weight:780; }
        .ea-mode-summary .title { margin:0.16rem 0 0 0; color:#f8fafc; font-size:1rem; font-weight:800; }
        .ea-mode-summary .line { margin:0.26rem 0 0 0; color:#cbd5e1; font-size:0.85rem; line-height:1.4; }
        .ea-mode-mini {
            border:1px solid rgba(148,163,184,0.14);
            border-radius:12px;
            padding:0.62rem 0.72rem;
            background:rgba(15,23,42,0.10);
            min-height:108px;
        }
        .ea-mode-mini.active {
            border-color: rgba(94,234,212,0.42);
            background: linear-gradient(180deg, rgba(12,74,110,0.14), rgba(15,23,42,0.10));
        }
        .ea-mode-mini .t { margin:0; color:#f8fafc; font-size:0.9rem; font-weight:760; }
        .ea-mode-mini .d { margin:0.28rem 0 0 0; color:#9fb1c7; font-size:0.8rem; line-height:1.32; }
        .ea-guide-lite {
            border:1px solid rgba(148,163,184,0.14);
            border-radius:14px;
            padding:0.76rem 0.82rem;
            background:rgba(15,23,42,0.10);
        }
        .ea-ws-kpi {
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 12px;
            padding: 0.7rem 0.76rem;
            background: rgba(2,6,23,0.16);
        }
        .ea-ws-kpi .k { margin:0; color:#94a3b8; font-size:0.76rem; letter-spacing:0.04em; text-transform:uppercase; }
        .ea-ws-kpi .v { margin:0.14rem 0 0 0; color:#f8fafc; font-size:1.3rem; font-weight:780; line-height:1.1; }
        .ea-inspect-card {
            border:1px solid rgba(148,163,184,0.14);
            border-radius:16px;
            padding:0.95rem 0.96rem;
            background:linear-gradient(180deg, rgba(15,23,42,0.18), rgba(2,6,23,0.10));
        }
        .ea-inspect-card.emphasis {
            border-color:rgba(34,211,238,0.22);
            background:linear-gradient(180deg, rgba(8,47,73,0.22), rgba(2,6,23,0.12));
            box-shadow:inset 0 1px 0 rgba(125,211,252,0.05);
        }
        .ea-inspect-title { margin:0; color:#f8fafc; font-size:1rem; font-weight:800; }
        .ea-inspect-copy { margin:0.22rem 0 0 0; color:#94a3b8; font-size:0.83rem; line-height:1.45; }
        .ea-inspect-grid {
            margin-top:0.82rem;
            padding-top:0.72rem;
            border-top:1px solid rgba(148,163,184,0.12);
            display:grid;
            grid-template-columns:1fr 1fr;
            gap:0.7rem 1rem;
        }
        .ea-inspect-item .label {
            margin:0;
            color:#7f91a8;
            font-size:0.7rem;
            letter-spacing:0.05em;
            text-transform:uppercase;
            font-weight:700;
        }
        .ea-inspect-item .value {
            margin:0.2rem 0 0 0;
            color:#f8fafc;
            font-size:0.98rem;
            font-weight:730;
            line-height:1.28;
        }
        .ea-inspect-item.accent .value { color:#f59e0b; font-weight:760; }
        .ea-inspect-item.strong .value { color:#ffffff; font-weight:780; }
        .ea-inspect-item.feature .value { color:#dff7ff; font-weight:790; }
        .ea-inspect-foot {
            margin:1rem 0 0 0;
            padding-top:0.78rem;
            border-top:1px solid rgba(148,163,184,0.10);
            color:#94a3b8;
            font-size:0.82rem;
            line-height:1.42;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    render_page_hero(
        "Analyze",
        "Upload media, configure run settings, and execute processing with lightweight readiness status.",
        kicker="Analyze",
    )

    if "upload_uploader_rev" not in st.session_state:
        st.session_state["upload_uploader_rev"] = 0
    if "upload_locked_file" not in st.session_state:
        st.session_state["upload_locked_file"] = None
    if "upload_locked" not in st.session_state:
        st.session_state["upload_locked"] = False

    analyze_mode_options = {
        "Fast Review": "fast_trend",
        "Original / Identity Accurate": "quality",
    }
    analyze_mode_labels = list(analyze_mode_options.keys())
    default_run_mode = analyze_mode_labels[0]
    reset_clicked = False

    # Section A — Run Setup
    with card():
        st.markdown('<div class="ea-block-head">Upload Workspace</div>', unsafe_allow_html=True)
        col_upload, col_cfg = st.columns([1.25, 1.0])

        with col_upload:
            st.markdown('<p class="ea-upload-step">1. Upload media</p>', unsafe_allow_html=True)
            if st.session_state["upload_locked"] and st.session_state["upload_locked_file"]:
                locked = st.session_state["upload_locked_file"]
                file_col, action_col = st.columns([5.2, 1.1])
                with file_col:
                    st.markdown(
                        f"""
                        <div class="ea-upload-lock">
                            <p class="ea-ws-label">Selected file (locked)</p>
                            <p class="name">{locked.get('name', '-')}</p>
                            <p class="meta">{locked.get('size_mb', '-')} MB</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with action_col:
                    reset_clicked = st.button(
                        "Reset",
                        use_container_width=True,
                        key="analyze_reset_btn",
                    )
                st.markdown(
                    '<div class="ea-upload-note">1 file selected. Upload is locked for this run. Click Reset to choose another file.</div>',
                    unsafe_allow_html=True,
                )
                uploaded = {
                    "name": locked.get("name"),
                    "type": locked.get("type") or "application/octet-stream",
                    "bytes": locked.get("bytes", b""),
                }
            else:
                up = st.file_uploader(
                    "Drag and drop media or click to browse",
                    type=["mp4", "mov", "avi", "jpg", "jpeg", "png"],
                    key=f"upload_single_{st.session_state['upload_uploader_rev']}",
                )
                uploaded = None
                render_inspection_overview_card(
                    "Upload hint",
                    "Choose one image or video for the current run.",
                    [
                        ("Selection", "Single locked input", "feature"),
                        ("Behavior", "Config stays tied to one file", None),
                    ],
                    extra_footer="After selection, use Reset to replace the current file.",
                )
                if up is not None:
                    file_bytes = up.getvalue()
                    st.session_state["upload_locked"] = True
                    st.session_state["upload_locked_file"] = {
                        "name": up.name,
                        "type": up.type,
                        "bytes": file_bytes,
                        "size_mb": round(len(file_bytes) / (1024 * 1024), 2),
                    }
                    try:
                        st.rerun()
                    except Exception:
                        st.experimental_rerun()
            st.caption(f"Supported: MP4, MOV, AVI, JPG, JPEG, PNG | Max {MAX_VIDEO_MB} MB")

        with col_cfg:
            st.markdown('<p class="ea-upload-step">2. Configure run</p>', unsafe_allow_html=True)
            model_choice = st.selectbox("Detection profile", list(VISIBLE_MODEL_OPTIONS.keys()), index=0, key="analyze_model")
            run_mode_choice = st.selectbox(
                "Pipeline mode",
                analyze_mode_labels,
                index=analyze_mode_labels.index(default_run_mode),
                key="analyze_run_mode",
            )
            pipeline_mode_key = analyze_mode_options[run_mode_choice]
            mode_meta = resolve_run_mode_meta(run_mode_choice, pipeline_mode_key)
            st.markdown(
                f"""
                <div class="ea-mode-summary">
                    <p class="eyebrow">{mode_meta.get('tag', 'Mode')}</p>
                    <p class="title">{mode_meta.get('title', run_mode_choice)}</p>
                    <p class="line"><strong style="color:#cbd5e1;">Optimized for:</strong> {mode_meta.get('optimize', '-')}</p>
                    <p class="line"><strong style="color:#cbd5e1;">Tradeoff:</strong> {mode_meta.get('tradeoff', '-')}</p>
                    <p class="line"><strong style="color:#cbd5e1;">When to use:</strong> {mode_meta.get('when', '-')}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown('<p class="ea-ws-label" style="margin:0.82rem 0 0.42rem 0;">Model guide</p>', unsafe_allow_html=True)
            for option_label in VISIBLE_MODEL_OPTIONS.keys():
                meta = MODEL_UI_META.get(option_label, {})
                active_class = " active" if option_label == model_choice else ""
                st.markdown(
                    f"""
                    <div class="ea-compare-card{active_class}" style="min-height:112px; margin-top:0.42rem;">
                        <div class="topline">
                            <p class="title">{meta.get('title', option_label)}</p>
                            <span class="tag">{meta.get('tag', 'Profile')}</span>
                        </div>
                        <p class="purpose">{meta.get('purpose', '')}</p>
                        <p class="note">{meta.get('note', '')}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.caption("Processing is synchronous. Large videos take longer.")
            run_clicked = st.button("Run Analysis", type="primary", use_container_width=True, key="analyze_run_btn")

    if reset_clicked:
        st.session_state["upload_locked"] = False
        st.session_state["upload_locked_file"] = None
        st.session_state["upload_uploader_rev"] += 1
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()

    if run_clicked:
        locked = st.session_state.get("upload_locked_file")
        if not locked:
            st.error("Please upload one media file before running analysis.")
        else:
            try:
                with st.spinner("Analyzing media. This may take longer for large videos."):
                    files = {"file": (locked.get("name"), locked.get("bytes", b""), locked.get("type") or "application/octet-stream")}
                    upload_resp = requests.post(f"{BACKEND_BASE_URL}/upload", files=files, timeout=120)
                    upload_resp.raise_for_status()
                    job_id = str(upload_resp.json().get("job_id"))
                    st.session_state["active_processing_job_id"] = job_id

                    process_resp = requests.post(
                        f"{BACKEND_BASE_URL}/process/{job_id}",
                        params={
                            "model_key": VISIBLE_MODEL_OPTIONS[model_choice],
                            "pipeline_mode": pipeline_mode_key,
                        },
                        timeout=600,
                    )
                    process_resp.raise_for_status()

                    results_resp = requests.get(f"{BACKEND_BASE_URL}/results/{job_id}", timeout=60)
                    if results_resp.ok:
                        st.session_state["last_results_payload"] = results_resp.json()
                    if st.session_state.get("active_processing_job_id") == job_id:
                        st.session_state["active_processing_job_id"] = None

                st.session_state["last_model_choice"] = model_choice
                st.session_state["last_pipeline_mode_choice"] = run_mode_choice
                st.session_state["last_pipeline_mode_key"] = pipeline_mode_key
                st.session_state["last_job_id"] = job_id
                st.session_state["results_view_job_id"] = job_id
                st.session_state["selected_history_job_id"] = job_id
                st.success("Analysis completed.")
                navigate_to_page("Results")
            except Exception as exc:
                st.error(f"Run failed: {exc}")

    context_job_id = get_current_context_job_id()
    jobs_df = fetch_jobs_from_backend()
    job_row = None
    if context_job_id and jobs_df is not None and not jobs_df.empty and "Job ID" in jobs_df.columns:
        match = jobs_df[jobs_df["Job ID"].astype(str) == str(context_job_id)]
        if not match.empty:
            job_row = match.iloc[0]

    results_payload = st.session_state.get("last_results_payload") if context_job_id else None
    if context_job_id and not results_payload:
        try:
            rr = requests.get(f"{BACKEND_BASE_URL}/results/{context_job_id}", timeout=20)
            rr.raise_for_status()
            results_payload = rr.json()
            st.session_state["last_results_payload"] = results_payload
        except Exception:
            results_payload = None

    metrics_payload = fetch_job_metrics(str(context_job_id)) if context_job_id else None
    job_meta = (results_payload or {}).get("job", {})
    filename = str(job_meta.get("filename") or (job_row.get("Video name") if job_row is not None else "") or "")
    media_type = "Video" if is_video_filename(filename) else ("Image" if filename else "-")
    status_txt = str(job_meta.get("status") or (job_row.get("Status") if job_row is not None else "-") or "-")
    submitted_txt = format_my_time(job_row.get("Submitted at")) if job_row is not None and "Submitted at" in job_row else "-"
    duration_txt = format_duration(job_row.get("Duration (s)")) if job_row is not None and "Duration (s)" in job_row else "-"
    representative_frame = _resolve_representative_frame_path(context_job_id, use_mock=False) if context_job_id else None
    bundle = get_results_bundle(str(context_job_id)) if context_job_id else {}
    review_bundle = (bundle or {}).get("review", {})
    trend_bundle = (bundle or {}).get("trend", {})
    sampled_visible_count = int(review_bundle.get("sampled_visible_count", 0) or 0)
    avg_sampled_visible_count = float(trend_bundle.get("avg_sampled_visible_count", 0.0) or 0.0)
    dominant_habitat = str((metrics_payload or {}).get("dominant_habitat") or "-")
    has_result_context = bool(context_job_id and (representative_frame or metrics_payload or results_payload))
    context_job_label = format_display_job_id(context_job_id, media_type)

    # Section B — Current Run Context
    with card():
        st.markdown('<div class="ea-block-head">Current Run Context</div>', unsafe_allow_html=True)
        meta_rows = [
            ("Media type", media_type),
            ("Job status", status_txt),
            ("Model", st.session_state.get("last_model_choice", "-")),
            ("Run mode", st.session_state.get("last_pipeline_mode_choice", "-")),
            ("Processing time", duration_txt or "-"),
            ("Run timestamp", submitted_txt or "-"),
        ]
        for idx, (label, value) in enumerate(meta_rows):
            left_col, right_col = st.columns([1.0, 2.1], gap="small")
            with left_col:
                st.markdown(
                    f'<p style="margin:0.22rem 0 0.32rem 0; color:#7f91a8; font-size:0.73rem; letter-spacing:0.04em; text-transform:uppercase; font-weight:700;">{label}</p>',
                    unsafe_allow_html=True,
                )
            with right_col:
                st.markdown(
                    f'<p style="margin:0.22rem 0 0.32rem 0; color:#f8fafc; font-size:0.98rem; font-weight:730; line-height:1.34;">{value}</p>',
                    unsafe_allow_html=True,
                )
            if idx < len(meta_rows) - 1:
                st.markdown('<div style="height:1px; background:rgba(148,163,184,0.08); margin:0.04rem 0;"></div>', unsafe_allow_html=True)

    # Section C — Primary Output Preview
    with card():
        st.markdown('<div class="ea-block-head">Current Result Preview</div>', unsafe_allow_html=True)
        if not has_result_context:
            st.markdown(
                '<p class="ea-ws-muted">Run one analysis to attach visual evidence and current run context here.</p>',
                unsafe_allow_html=True,
            )
        else:
            left, right = st.columns([1.45, 1.0])
            with left:
                if representative_frame:
                    st.image(representative_frame, caption="Current run review frame", width="stretch")
                else:
                    st.info("This run has no saved review frame yet.")
            with right:
                st.markdown(
                    f"""
                    <div style="margin-top:0.05rem; border:1px solid rgba(148,163,184,0.18); border-radius:16px; padding:0.95rem 0.96rem; background:linear-gradient(180deg, rgba(15,23,42,0.24), rgba(2,6,23,0.15));">
                        <div style="display:grid; grid-template-columns:1.1fr 1fr; gap:0.9rem; align-items:end;">
                            <div>
                                <p style="margin:0; color:#97a9bf; font-size:0.7rem; letter-spacing:0.08em; text-transform:uppercase; font-weight:700;">Sampled visible count</p>
                                <p style="margin:0.24rem 0 0 0; color:#ffffff; font-size:2.0rem; font-weight:840; line-height:0.95;">{sampled_visible_count}</p>
                            </div>
                            <div>
                                <p style="margin:0; color:#97a9bf; font-size:0.7rem; letter-spacing:0.08em; text-transform:uppercase; font-weight:700;">Avg sampled visible count</p>
                                <p style="margin:0.24rem 0 0 0; color:#e2e8f0; font-size:1.22rem; font-weight:790; line-height:1.0;">{avg_sampled_visible_count:.2f}</p>
                            </div>
                        </div>
                        <div style="margin-top:0.82rem; padding-top:0.72rem; border-top:1px solid rgba(148,163,184,0.14);">
                            <p style="margin:0; color:#8ea4bb; font-size:0.7rem; letter-spacing:0.08em; text-transform:uppercase; font-weight:700;">Dominant habitat</p>
                            <p style="margin:0.18rem 0 0 0; color:#dff7ff; font-size:1.08rem; font-weight:800; line-height:1.04;">{dominant_habitat}</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                detail_rows = [
                    ("Active job", context_job_label),
                    ("Status", status_txt or "-"),
                    ("Processing time", duration_txt or "-"),
                ]
                st.markdown('<p style="margin:0.85rem 0 0.4rem 0; color:#f8fafc; font-size:0.9rem; font-weight:770;">Run details</p>', unsafe_allow_html=True)
                for idx, (label, value) in enumerate(detail_rows):
                    lcol, rcol = st.columns([1.0, 1.6], gap="small")
                    with lcol:
                        st.markdown(
                            f'<p style="margin:0.2rem 0 0.32rem 0; color:#7f91a8; font-size:0.73rem; letter-spacing:0.04em; text-transform:uppercase; font-weight:700;">{label}</p>',
                            unsafe_allow_html=True,
                        )
                    with rcol:
                        st.markdown(
                            f'<p title="{context_job_id or "-"}" style="margin:0.2rem 0 0.32rem 0; color:#f8fafc; font-size:0.98rem; font-weight:730; line-height:1.34;">{value}</p>',
                            unsafe_allow_html=True,
                        )
                    if idx < len(detail_rows) - 1:
                        st.markdown('<div style="height:1px; background:rgba(148,163,184,0.08); margin:0.04rem 0;"></div>', unsafe_allow_html=True)
                st.markdown(
                    f'<p class="ea-ws-muted" style="margin-top:0.55rem;">Dominant habitat: <strong>{dominant_habitat}</strong></p>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<p class="ea-ws-muted">Analyze now keeps the latest run attached so the page does not feel result-empty after processing.</p>',
                    unsafe_allow_html=True,
                )
                st.markdown("<div style='height:0.45rem;'></div>", unsafe_allow_html=True)
                if st.button("Open detailed results", use_container_width=True, key="analyze_preview_open_results"):
                    if context_job_id:
                        st.session_state["results_view_job_id"] = context_job_id
                    navigate_to_page("Results")

    # Section E — Next Actions
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
    ]
    hero_image = next((str(p) for p in hero_candidates if p.exists()), None)
    st.markdown(
        """
        <style>
        .ea-home-kicker { margin:0; font-size:0.74rem; letter-spacing:0.1em; text-transform:uppercase; color:#f59e0b; font-weight:740; }
        .ea-home-title { margin:0.24rem 0 0.52rem 0; font-size:clamp(3.5rem, 6.7vw, 5.35rem); line-height:0.93; letter-spacing:-0.035em; color:#f8fafc; font-weight:900; }
        .ea-home-sub { margin:0; color:#aab8cb; font-size:1rem; line-height:1.45; max-width:44ch; }
        .ea-home-head { margin:2.1rem 0 0.42rem 0; color:#f8fafc; font-size:1.92rem; font-weight:810; letter-spacing:-0.012em; }
        .ea-home-section-note { margin:0 0 0.92rem 0; color:#99aac0; font-size:0.9rem; line-height:1.38; }
        .ea-home-soft-card {
            border: 1px solid rgba(148,163,184,0.12);
            border-radius: 12px;
            padding: 0.82rem 0.9rem;
            background: rgba(2,6,23,0.14);
        }
        .ea-home-soft-card .t { margin:0; color:#f8fafc; font-size:1.14rem; font-weight:760; letter-spacing:-0.01em; }
        .ea-home-soft-card .d { margin:0.22rem 0 0 0; color:#9fb1c7; font-size:0.88rem; line-height:1.35; }
        [data-testid="stImage"] img {
            border-radius: 14px;
            border: 1px solid rgba(148,163,184,0.14);
        }
        .ea-home-flow {
            margin-top: 0.38rem;
            border-top: 1px solid rgba(148,163,184,0.16);
            border-bottom: 1px solid rgba(148,163,184,0.16);
            padding: 0.95rem 0.18rem;
        }
        .ea-home-flow-grid {
            display: grid;
            grid-template-columns: 1fr auto 1fr auto 1fr;
            align-items: center;
            gap: 0.42rem;
        }
        .ea-home-flow-step .n { margin:0; font-size:0.74rem; letter-spacing:0.08em; text-transform:uppercase; color:#f59e0b; font-weight:760; }
        .ea-home-flow-step .t { margin:0.1rem 0 0 0; color:#f8fafc; font-size:1.18rem; font-weight:785; letter-spacing:-0.01em; }
        .ea-home-flow-step .d { margin:0.16rem 0 0 0; color:#9fb1c7; font-size:0.86rem; line-height:1.34; max-width:34ch; }
        .ea-home-flow-sep { width: 36px; height: 1px; background: rgba(148,163,184,0.24); margin: 0 0.3rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.38, 1.12])
    with left:
        st.markdown(
            """
            <p class="ea-home-kicker">Elephant Analytics Platform</p>
            <h1 class="ea-home-title">AI Elephant Intelligence</h1>
            <p class="ea-home-sub">Turn field media into clear counts, track visibility, and client-ready summaries.</p>
            """,
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Start analysis", type="primary", use_container_width=True, key="home_start_analysis"):
                st.session_state["pending_page_nav"] = "Analyze"
                st.rerun()
        with c2:
            if st.button("View latest results", use_container_width=True, key="home_view_latest_results"):
                st.session_state["pending_page_nav"] = "Results"
                st.rerun()
        if st.button("How it works", key="home_how_it_works"):
            st.session_state["pending_page_nav"] = "Analyze"
            st.rerun()
    with right:
        if hero_image:
            st.image(hero_image, use_container_width=True)
        else:
            st.info("Add a hero image at `streamlit/Home_Page.jpg`.")

    st.markdown('<h3 class="ea-home-head">Platform strengths</h3>', unsafe_allow_html=True)
    st.markdown('<p class="ea-home-section-note">Core analytical advantages across media review and reporting.</p>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown('<div class="ea-home-soft-card"><p class="t">Faster review</p><p class="d">Shorter review cycles.</p></div>', unsafe_allow_html=True)
    with s2:
        st.markdown('<div class="ea-home-soft-card"><p class="t">Evidence-based</p><p class="d">Visual proof for each claim.</p></div>', unsafe_allow_html=True)
    with s3:
        st.markdown('<div class="ea-home-soft-card"><p class="t">Track visibility</p><p class="d">See movement over time.</p></div>', unsafe_allow_html=True)
    with s4:
        st.markdown('<div class="ea-home-soft-card"><p class="t">Spatial context</p><p class="d">Map insight from GPS media.</p></div>', unsafe_allow_html=True)

    st.markdown('<h3 class="ea-home-head">What the platform delivers</h3>', unsafe_allow_html=True)
    st.markdown('<p class="ea-home-section-note">Client-facing outputs generated from each processed run.</p>', unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown('<div class="ea-home-soft-card"><p class="t">Evidence frames</p><p class="d">Clear proof for reporting.</p></div>', unsafe_allow_html=True)
    with d2:
        st.markdown('<div class="ea-home-soft-card"><p class="t">Herd and track insights</p><p class="d">Trend and visibility summaries.</p></div>', unsafe_allow_html=True)
    with d3:
        st.markdown('<div class="ea-home-soft-card"><p class="t">Habitat and route context</p><p class="d">Context when GPS is available.</p></div>', unsafe_allow_html=True)

    st.markdown('<h3 class="ea-home-head">3-step workflow</h3>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="ea-home-flow">
            <div class="ea-home-flow-grid">
                <div class="ea-home-flow-step">
                    <p class="n">01</p>
                    <p class="t">Upload</p>
                    <p class="d">Bring in video or image files from field operations.</p>
                </div>
                <div class="ea-home-flow-sep" aria-hidden="true"></div>
                <div class="ea-home-flow-step">
                    <p class="n">02</p>
                    <p class="t">Analyze</p>
                    <p class="d">Choose model and mode, then process in one run.</p>
                </div>
                <div class="ea-home-flow-sep" aria-hidden="true"></div>
                <div class="ea-home-flow-step">
                    <p class="n">03</p>
                    <p class="t">Present</p>
                    <p class="d">Present evidence, counts, and context in a polished format.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def results_page():
    render_page_hero(
        "Results",
        "Review the latest run with a clear summary first, then open advanced analysis on demand.",
        kicker="Results",
    )
    use_mock = st.session_state.get("use_mock", False)
    active_job = get_results_view_job_id() or get_current_context_job_id()
    if active_job and not get_results_view_job_id():
        st.session_state["results_view_job_id"] = active_job
    if not use_mock and not active_job:
        with card():
            st.info("No run is locked in Results. Open a run from History or explicitly switch to the active processing run.")
            c1, c2 = st.columns(2)
            if c1.button("Go to History", type="primary", key="results_go_history_btn"):
                st.session_state["pending_page_nav"] = "History"
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()
            active_processing = get_active_processing_job_id()
            if c2.button("Switch to active run", key="results_switch_active_btn", disabled=not active_processing):
                st.session_state["results_view_job_id"] = active_processing
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()
        return

    active_mode_key = str(st.session_state.get("last_pipeline_mode_key", "") or "").strip().lower()
    is_fast_like_mode = active_mode_key in {"fast_trend", "balanced_2fps", "fast"}

    # 1) Review workspace (always visible)
    with card():
        preview_section_header("Review Workspace", "Frame-first review workspace with compact run summary and current visual evidence.")
        render_review_workspace()
        st.caption(SAMPLED_COUNT_CAVEAT)

    # 2) Trend (always visible, lightweight)
    if st.session_state.get("preview_is_video", True):
        with card():
            preview_section_header("Trend", "Herd and sampled visibility trend with compact confidence context.")
            charts_section(compact=True, wrap_card=False)
            render_confidence_trend_compact()
            if is_fast_like_mode:
                st.caption("Fast review mode prioritizes sampled trend/context outputs. Tracking-aware depth may be limited.")

    # 3) Deep analysis tabs (contained area)
    with card():
        preview_section_header("Deep Analysis", "Open focused tabs for detailed comparison.")
        tab_ind, tab_hab_map, tab_diag = st.tabs(["Individuals & Timeline", "Habitat & Map", "Diagnostics"])
        with tab_ind:
            st.caption("Review unique elephants, current-frame identities, and timeline visibility for tracked runs.")
            individual_spotlight(show_header=False)
            if st.session_state.get("preview_is_video", True):
                st.markdown("")
                timeline_section()
        with tab_hab_map:
            st.caption("Inspect habitat distribution, habitat change over time, and spatial context when GPS is available.")
            habitat_section()
            st.markdown("")
            map_section()
        with tab_diag:
            st.caption("Check confidence quality, observed composition, and supporting diagnostic charts for this run.")
            result_quality_section(wrap_cards=False)


def history_page():
    render_page_hero(
        "History",
        "Browse previous analyses, reopen results, and manage stored runs.",
        kicker="History",
    )
    jobs_df = fetch_jobs_from_backend()
    if jobs_df is None:
        jobs_df = pd.DataFrame(columns=["Job ID", "Video name", "Status", "Submitted at", "Duration (s)"])

    if not jobs_df.empty:
        jobs_df = jobs_df.copy()
        jobs_df["Submitted at"] = jobs_df["Submitted at"].apply(format_my_time)
        if "Duration (s)" in jobs_df.columns:
            jobs_df["Duration"] = jobs_df["Duration (s)"].apply(format_duration)
        else:
            jobs_df["Duration"] = "-"

    with card():
        st.markdown('<div class="ea-block-head">Analysis archive</div>', unsafe_allow_html=True)
        if jobs_df.empty:
            st.info("No history yet. Complete one analysis to build your evidence archive.")
            return

        job_ids = jobs_df["Job ID"].dropna().astype(str).tolist()
        if not job_ids:
            return
        default_job = get_selected_history_job_id() or get_results_view_job_id()
        if default_job not in job_ids:
            default_job = job_ids[0]

        nav_df = jobs_df.copy()
        input_types = nav_df["Video name"].apply(lambda name: "Video" if is_video_filename(str(name)) else "Image")
        nav_df["Run"] = [
            format_display_job_id(str(job_id), input_type)
            for job_id, input_type in zip(nav_df["Job ID"].astype(str), input_types)
        ]
        if "history_search_query" not in st.session_state:
            st.session_state["history_search_query"] = ""
        if "history_confirm_delete_all" not in st.session_state:
            st.session_state["history_confirm_delete_all"] = False
        if "history_pending_delete_job" not in st.session_state:
            st.session_state["history_pending_delete_job"] = None
        search_query = st.text_input(
            "Search history",
            value=st.session_state.get("history_search_query", ""),
            key="history_search_query",
            placeholder="Search by run label, filename, status, or date",
        ).strip()
        query_lower = search_query.lower()
        if query_lower:
            nav_df = nav_df[
                nav_df.apply(
                    lambda row: query_lower in " ".join(
                        [
                            str(row.get("Run", "")).lower(),
                            str(row.get("Video name", "")).lower(),
                            str(row.get("Status", "")).lower(),
                            str(row.get("Submitted at", "")).lower(),
                        ]
                    ),
                    axis=1,
                )
            ].copy()
        if nav_df.empty:
            st.warning("No runs match the current search.")
            return

        filtered_job_ids = nav_df["Job ID"].astype(str).tolist()
        if default_job not in filtered_job_ids:
            default_job = filtered_job_ids[0]
        selected_job = str(st.session_state.get("selected_history_job_id") or default_job)
        if selected_job not in filtered_job_ids:
            selected_job = filtered_job_ids[0]
        st.session_state["selected_history_job_id"] = selected_job

        action_left, action_mid, action_right = st.columns([1.2, 1.0, 1.0], gap="small")
        if action_left.button("Review selected result", type="primary", use_container_width=True, key="history_review_selected_btn"):
            st.session_state["results_view_job_id"] = selected_job
            st.session_state["selected_history_job_id"] = selected_job
            st.session_state["pending_page_nav"] = "Results"
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
        if action_mid.button("Delete selected", use_container_width=True, key="history_arm_delete_selected_btn"):
            st.session_state["history_pending_delete_job"] = selected_job
            st.session_state["history_confirm_delete_all"] = False
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
        if action_right.button("Delete all", use_container_width=True, key="history_arm_delete_all_btn"):
            st.session_state["history_confirm_delete_all"] = True
            st.session_state["history_pending_delete_job"] = None
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()

        pending_delete_job = st.session_state.get("history_pending_delete_job")
        if pending_delete_job:
            pending_label = nav_df.loc[nav_df["Job ID"].astype(str) == str(pending_delete_job), "Run"]
            pending_label = pending_label.iloc[0] if not pending_label.empty else format_display_job_id(str(pending_delete_job), None)
            st.warning(f"Delete selected is permanent. Delete {pending_label}?")
            d1, d2 = st.columns([1.0, 1.0], gap="small")
            if d1.button("Confirm delete selected", use_container_width=True, key="history_delete_selected_btn"):
                try:
                    resp = requests.post(
                        f"{BACKEND_BASE_URL}/admin/delete-jobs",
                        json={"job_ids": [str(pending_delete_job)]},
                        timeout=30,
                    )
                    resp.raise_for_status()
                    if st.session_state.get("results_view_job_id") == str(pending_delete_job):
                        st.session_state["results_view_job_id"] = None
                    if st.session_state.get("selected_history_job_id") == str(pending_delete_job):
                        st.session_state["selected_history_job_id"] = None
                    if st.session_state.get("last_job_id") == str(pending_delete_job):
                        st.session_state["last_job_id"] = None
                    st.session_state["history_pending_delete_job"] = None
                    st.success("Selected run deleted.")
                    try:
                        st.rerun()
                    except Exception:
                        st.experimental_rerun()
                except Exception as exc:
                    st.error(f"Delete failed: {exc}")
            if d2.button("Cancel", use_container_width=True, key="history_cancel_delete_selected_btn"):
                st.session_state["history_pending_delete_job"] = None
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()
        if st.session_state.get("history_confirm_delete_all"):
            st.warning("Delete all is permanent. This removes all runs, frames, uploads, and stored outputs.")
            a1, a2 = st.columns([1.0, 1.0], gap="small")
            if a1.button("Confirm delete all", use_container_width=True, key="history_delete_all_btn"):
                try:
                    resp = requests.post(f"{BACKEND_BASE_URL}/admin/clear-history", timeout=60)
                    resp.raise_for_status()
                    st.session_state["results_view_job_id"] = None
                    st.session_state["selected_history_job_id"] = None
                    st.session_state["last_job_id"] = None
                    st.session_state["history_confirm_delete_all"] = False
                    st.success("All history deleted.")
                    try:
                        st.rerun()
                    except Exception:
                        st.experimental_rerun()
                except Exception as exc:
                    st.error(f"Delete all failed: {exc}")
            if a2.button("Cancel", use_container_width=True, key="history_cancel_delete_all_btn"):
                st.session_state["history_confirm_delete_all"] = False
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()

        st.caption("Quick navigation")
        nav_rows = nav_df[["Job ID", "Run", "Video name"]].head(8).to_dict("records")
        chunk_size = 4
        for start in range(0, len(nav_rows), chunk_size):
            chunk = nav_rows[start:start + chunk_size]
            cols = st.columns(len(chunk))
            for idx, row in enumerate(chunk):
                run_id = str(row.get("Job ID", ""))
                run_label = str(row.get("Run", run_id))
                media_name = str(row.get("Video name", "") or "")
                button_label = f"{run_label}\n{media_name}"
                if cols[idx].button(
                    button_label,
                    use_container_width=True,
                    type="primary" if run_id == selected_job else "secondary",
                    key=f"history_nav_{run_id}",
                ):
                    st.session_state["selected_history_job_id"] = run_id
                    try:
                        st.rerun()
                    except Exception:
                        st.experimental_rerun()

        show_df = nav_df.copy()
        show_df["Job ID"] = show_df["Run"]
        show_cols = [c for c in ["Job ID", "Video name", "Status", "Submitted at", "Duration"] if c in show_df.columns]
        st.markdown("")
        st.dataframe(show_df[show_cols], use_container_width=True, hide_index=True)
# ===================== ROUTER =====================

PAGES = {
    "Home": "home",
    "Analyze": "analyze",
    "Results": "results",
    "History": "history",
}

if "pending_page_nav" in st.session_state:
    next_page_label = st.session_state.pop("pending_page_nav")
    st.session_state["page_nav"] = next_page_label if next_page_label in PAGES else "Home"

if "page_nav" not in st.session_state:
    st.session_state["page_nav"] = "Home"
elif st.session_state.get("page_nav") not in PAGES:
    st.session_state["page_nav"] = "Home"

with st.sidebar:
    st.title("Elephant Analytics")
    st.caption("Navigate the analysis flow")
    page_label = st.radio("Navigation", list(PAGES.keys()), key="page_nav")
    st.caption("Open Results for review, History for archive, and Spatial Review when local tiles are connected.")

page = PAGES[page_label]

if page == "home":
    home_page()
elif page == "analyze":
    upload_jobs_page()
elif page == "results":
    results_page()
elif page == "history":
    history_page()





