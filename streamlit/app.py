import streamlit as st
import pandas as pd
import numpy as np
import json
import os
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

BACKEND_BASE_URL = "http://127.0.0.1:8000"
MAX_VIDEO_MB = 500
TILE_SERVER_URL = os.getenv("EA_TILE_URL", "")

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
        --color-forest-green: #2d5016;
        --color-forest-green-light: #4f7f29;
        --color-warm-orange: #e07b39;
        --color-warm-orange-light: #f59550;
        --color-border: #e5e5e0;
        --color-text-primary: #222222;
        --color-text-secondary: #555555;
        --color-text-muted: #888888;
        --radius-md: 12px;
        --radius-lg: 18px;
        --shadow-card: 0 8px 18px rgba(15,23,42,0.06);
        --shadow-card-hover: 0 16px 30px rgba(15,23,42,0.10);
    }
    html, body, [class^="css"] {
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
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


def downsample_points(df: pd.DataFrame, max_points: int = 300) -> pd.DataFrame:
    if df is None or df.empty or len(df) <= max_points:
        return df
    step = max(1, len(df) // max_points)
    return df.iloc[::step].reset_index(drop=True)

# ===================== PAGE SECTIONS =====================

def hero_section():
    col1, col2 = st.columns([1.4, 1])
    with col1:
        st.markdown(
            "<span style='font-size:0.75rem;letter-spacing:0.08em;text-transform:uppercase;color:#e07b39;'>"
            "Wildlife AI for conservation</span>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='ea-hero-title'>Turn elephant footage into actionable insights.</div>", unsafe_allow_html=True)
        st.markdown(
            "<p class='ea-hero-subtitle'>"
            "This tool detects elephants, tracks individuals, classifies age groups "
            "(baby, juvenile, adult), measures herd size, analyses behaviour patterns, "
            "and identifies habitat types from drone videos."
            "</p>",
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("🧪 Try the demo dashboard", use_container_width=True):
                st.session_state["demo_clicked"] = True
        with c2:
            st.download_button(
                "⬇️ Download SEGP report (PDF)",
                data=b"",
                file_name="elephant_analytics_report.pdf",
                help="Placeholder – attach your PDF later.",
                use_container_width=True,
            )

        st.markdown(
            "<p style='font-size:0.75rem;color:var(--color-text-muted);margin-top:0.6rem;'>"
            "✅ Built with ecologists for real fieldwork."
            "</p>",
            unsafe_allow_html=True,
        )

    with col2:
        with card():
            st.image(
                "https://images.unsplash.com/photo-1696251803608-e8893f7fcdf3"
                "?auto=format&fit=crop&w=900&q=80",
                caption="Elephant conservation – demo image",
                use_container_width=True,
            )
            st.markdown("##### Real-time analysis")
            if st.session_state.get("use_mock", False):
                summary = mock_detection_summary()
            else:
                summary = get_real_summary() or empty_detection_summary()
            c1, c2 = st.columns(2)
            c1.metric("Elephants detected", summary["unique_elephants"])
            c2.metric("Avg. herd size", summary["avg_herd_size"])
            st.info("✨ Placeholder where you could embed a GIF / animation later.")

def stats_cards():
    if st.session_state.get("use_mock", False):
        s = mock_detection_summary()
    else:
        s = get_real_summary() or empty_detection_summary()
    cards = [
        ("🐘", str(s["unique_elephants"]), "elephants detected (unique)", None),
        ("🧬", f"{s['age_distribution']['adult']}% · {s['age_distribution']['juvenile']}% · {s['age_distribution']['baby']}%",
         "adults · juveniles · babies", None),
        ("👥", f"{s['avg_herd_size']:.1f}", "average herd size", "+0.3 vs last clip"),
        ("🗺️", s["main_habitats"], "main habitats", None),
        ("⏱️", s["video_duration"], "total video duration", None),
    ]

    st.markdown(
        "<p style='text-align:center;font-weight:600;color:var(--color-text-primary);margin-bottom:0.3rem;'>"
        "From pixels to numbers that matter.</p>",
        unsafe_allow_html=True,
    )

    cols = st.columns(5)
    for col, (icon, value, label, trend) in zip(cols, cards):
        with col:
            with card():
                st.markdown(f"<div style='font-size:1.8rem;margin-bottom:0.1rem;'>{icon}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:1.4rem;font-weight:650;'>{value}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:0.8rem;color:var(--color-text-secondary);'>{label}</div>", unsafe_allow_html=True)
                if trend:
                    st.markdown(
                        f"<div style='font-size:0.75rem;color:#16a34a;margin-top:0.35rem;'>📈 {trend}</div>",
                        unsafe_allow_html=True,
                    )

def charts_section():
    use_mock = st.session_state.get("use_mock", False)
    if use_mock:
        herd_df = mock_herd_series()
    else:
        job_id = st.session_state.get("last_job_id")
        metrics = fetch_job_metrics(job_id) if job_id else None
        series = metrics.get("herd_series") if metrics else None
        herd_df = pd.DataFrame(series or [])

    behaviour_series = mock_behaviour_distribution() if use_mock else empty_behaviour_distribution()
    behaviour = behaviour_series.reset_index()
    behaviour.columns = ["behaviour", "percentage"]

    behaviour_chart = (
        alt.Chart(behaviour)
        .mark_bar()
        .encode(
            x=alt.X("behaviour:N", axis=alt.Axis(title=None, labelAngle=0)),
            y=alt.Y("percentage:Q", axis=alt.Axis(title=None)),
            tooltip=[
                alt.Tooltip("behaviour:N", title="Behaviour"),
                alt.Tooltip("percentage:Q", title="%"),
            ],
        )
        .properties(height=320)
        .configure_view(strokeOpacity=0)
        .configure_axisX(labelAngle=0)
    )

    col1, col2 = st.columns(2)

    with col1:
        with card():
            header_cols = st.columns([3, 1])
            with header_cols[0]:
                st.markdown("#### Herd size over time")
            with header_cols[1]:
                if use_mock:
                    herd_mode = "Time"
                else:
                    options = ["Frame"]
                    if "time_seconds" in herd_df.columns:
                        options = ["Time", "Frame"]
                    job_id = st.session_state.get("last_job_id") or "none"
                    herd_mode = st.radio(
                        "X axis",
                        options,
                        horizontal=True,
                        label_visibility="collapsed",
                        key=f"herd_x_mode_{job_id}",
                    )
            st.caption("How many elephants are visible at each moment in the video.")
            if herd_df.empty:
                st.info("No data yet. Enable mock data or process a video.")
            else:
                if use_mock:
                    times_order = herd_df["time"].tolist()
                    herd_chart = (
                        alt.Chart(herd_df)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X(
                                "time:N",
                                sort=times_order,
                                axis=alt.Axis(
                                    title="Time (mm:ss)",
                                    labelAngle=0,
                                    labelOverlap="greedy",
                                ),
                            ),
                            y=alt.Y("herd_size:Q", axis=alt.Axis(title=None)),
                            tooltip=[
                                alt.Tooltip("time:N", title="Time"),
                                alt.Tooltip("herd_size:Q", title="Herd size"),
                            ],
                        )
                        .properties(height=320)
                        .configure_view(strokeOpacity=0)
                        .configure_axisX(labelAngle=0)
                    )
                else:
                    herd_df = herd_df.copy()
                    if "time_seconds" in herd_df.columns:
                        herd_df["time_label"] = herd_df["time_seconds"].apply(format_seconds)
                    if herd_mode == "Time" and "time_seconds" in herd_df.columns:
                        time_order = herd_df["time_seconds"].tolist()
                        herd_chart = (
                            alt.Chart(herd_df)
                            .mark_line(point=True)
                            .encode(
                                x=alt.X(
                                    "time_label:N",
                                    sort=time_order,
                                    axis=alt.Axis(title="Time (mm:ss)", labelAngle=0),
                                ),
                                y=alt.Y("herd_size:Q", axis=alt.Axis(title=None)),
                                tooltip=[
                                    alt.Tooltip("time_seconds:Q", title="Time (s)", format=".1f"),
                                    alt.Tooltip("frame_index:Q", title="Frame"),
                                    alt.Tooltip("herd_size:Q", title="Herd size"),
                                ],
                            )
                            .properties(height=320)
                            .configure_view(strokeOpacity=0)
                            .configure_axisX(labelAngle=0)
                        )
                    else:
                        herd_chart = (
                            alt.Chart(herd_df)
                            .mark_line(point=True)
                            .encode(
                                x=alt.X(
                                    "frame_index:Q",
                                    axis=alt.Axis(title="Frame"),
                                ),
                                y=alt.Y("herd_size:Q", axis=alt.Axis(title=None)),
                                tooltip=[
                                    alt.Tooltip("frame_index:Q", title="Frame"),
                                    alt.Tooltip("time_seconds:Q", title="Time (s)", format=".1f"),
                                    alt.Tooltip("herd_size:Q", title="Herd size"),
                                ],
                            )
                            .properties(height=320)
                            .configure_view(strokeOpacity=0)
                        )

                st.altair_chart(herd_chart, use_container_width=True)
                table_df = herd_df.copy()
                if use_mock:
                    table_df = table_df.rename(
                        columns={"time": "Time (mm:ss)", "herd_size": "Herd size"}
                    )
                    table_cols = ["Time (mm:ss)", "Herd size"]
                else:
                    if "time_label" in table_df.columns:
                        table_df["Time (mm:ss)"] = table_df["time_label"]
                    table_df = table_df.rename(
                        columns={"frame_index": "Frame", "herd_size": "Herd size"}
                    )
                    if herd_mode == "Time" and "Time (mm:ss)" in table_df.columns:
                        table_cols = ["Time (mm:ss)", "Frame", "Herd size"]
                    else:
                        table_cols = ["Frame", "Time (mm:ss)", "Herd size"]
                    table_cols = [c for c in table_cols if c in table_df.columns]
                st.dataframe(table_df[table_cols], use_container_width=True, hide_index=True)

    with col2:
        with card():
            st.markdown("#### Behaviour distribution")
            st.caption("How often elephants are grazing, travelling, resting, etc.")
            if behaviour.empty:
                st.info("No data yet. Enable mock data or process a video.")
            else:
                st.altair_chart(behaviour_chart, use_container_width=True)

def timeline_and_habitat_section():
    with card():
        tab_timeline, tab_habitat = st.tabs(["Timeline table", "Habitat breakdown"])
        use_mock = st.session_state.get("use_mock", False)
        df = empty_timeline_table()
        if use_mock:
            df = mock_timeline_table()
        else:
            job_id = st.session_state.get("last_job_id")
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
            if age_filter != "All":
                if "Age class" in df_show.columns:
                    df_show = df_show[df_show["Age class"] == age_filter]
            st.dataframe(df_show, use_container_width=True, hide_index=True)

        with tab_habitat:
            if not use_mock:
                st.info("No habitat data yet. Enable mock data to preview.")
                return
            habitat_series = pd.Series(
                {"Grassland": 48, "Oil Palm": 32, "Field": 14, "Forest": 6},
                name="percentage",
            )

            col1, col2 = st.columns([1.25, 1])
            with col1:
                st.subheader("Time spent by habitat type")

                # ---- Pie chart (matplotlib) ----
                labels = habitat_series.index.tolist()
                values = habitat_series.values.tolist()

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.pie(
                    values,
                    labels=labels,
                    autopct="%1.0f%%",
                    startangle=90,
                    labeldistance=1.08,
                    pctdistance=0.78,
                    wedgeprops={"linewidth": 1, "edgecolor": "white"},
                )
                ax.axis("equal")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            with col2:
                st.markdown("##### Summary")
                st.write(
                    "Most time is spent in **grassland** and **oil palm** habitats. "
                    "Forest appears less frequently, suggesting elephants prefer "
                    "open areas with abundant vegetation for grazing and foraging."
                )
                st.markdown("##### Habitat legend")
                for name, val in habitat_series.items():
                    st.write(f"- **{name}** – {val}%")

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

    if TILE_SERVER_URL:
        m = folium.Map(
            location=center,
            zoom_start=zoom,
            tiles=TILE_SERVER_URL,
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
    col1, col2 = st.columns(2)

    with col1:
        with card():
            st.markdown("#### Map of detections (Folium)")
            st.caption("Map of elephant detections based on GPS (if available).")

            job_id = st.session_state.get("last_job_id")
            points_df = None
            if job_id:
                points_df = cached_gps_points(job_id)
            if points_df is not None:
                points_df = downsample_points(points_df, max_points=200)
            if points_df is None:
                # Example detection points (Malaysia / SG region)
                points_df = pd.DataFrame(
                    {
                        "lat": [3.1390, 3.1398, 3.1384, 3.1406, 1.3521],
                        "lon": [101.6869, 101.6882, 101.6855, 101.6848, 103.8198],
                    }
                )
            else:
                st.caption("Showing GPS track from the latest processed video.")

            # Optional: Example track (just for demo). Later you can replace with real GPS extracted from video.
            track_df = points_df.copy() if not points_df.empty else None

            m = build_folium_map(points_df, track_df=track_df)
            map_key = f"map-{job_id}" if job_id else "map-demo"
            st_folium(m, height=420, use_container_width=True, key=map_key)

            st.info("Showing start/end markers for the GPS track when available.")

    with col2:
        with card():
            st.markdown("#### Individual timeline")
            st.caption("Selected: ELE_001")
            st.progress(90, text="Visibility 0:12 → 8:23")

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Distance travelled", "2.4 km")
            with col_b:
                st.metric("Main habitat", "Grassland")

            st.markdown("---")
            st.markdown("##### Movement pattern")
            st.write(
                "This adult elephant remained with the herd throughout the observation "
                "period, travelling from grassland through oil palm areas. Movement was "
                "steady with frequent grazing stops."
            )

def upload_jobs_page():
    st.markdown("### Upload & jobs")

    with card():
        st.subheader("Upload drone footage")

        col_drop, col_meta = st.columns([1.2, 1])
        with col_drop:
            uploaded_file = st.file_uploader(
                "Drop MP4 / MOV / AVI here or click to browse",
                type=["mp4", "mov", "avi", "jpg", "jpeg", "png"],
            )
            st.caption(f"Maximum file size: {MAX_VIDEO_MB} MB")

        with col_meta:
            gps_input = st.text_input("Optional GPS coordinates", "")
            model_choice = st.selectbox(
                "Model version",
                [
                    "elephant_v4_all (default)",
                    "elephant_v3_all",
                    "elephant_v1_project5",
                ],
            )
            use_mock = st.checkbox("Use mock data for demo", value=st.session_state.get("use_mock", False))
            st.session_state["use_mock"] = use_mock

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            start = st.button("🚀 Start analysis", use_container_width=True)
        with col_btn2:
            clear = st.button("🧹 Clear", use_container_width=True)

        st.caption(
            "Analysis may take several minutes depending on video length and quality. "
            "Supported formats: MP4, MOV, AVI • Recommended: 1080p or higher, 30fps."
        )

        if clear:
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()

        if start:
            if uploaded_file is None and not use_mock:
                st.error("Please upload a video or enable mock data.")
            else:
                with st.spinner("Processing video…"):
                    time.sleep(1.5)
                    if use_mock:
                        result = {
                            "video_name": getattr(uploaded_file, "name", "demo_video.mp4"),
                            "summary": mock_detection_summary(),
                        }
                    else:
                        try:
                            file_bytes = uploaded_file.getvalue()
                            files = {
                                "file": (
                                    uploaded_file.name,
                                    file_bytes,
                                    uploaded_file.type or "video/mp4",
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
                                timeout=600,
                            )
                            process_resp.raise_for_status()
                            result = process_resp.json()

                            results_resp = requests.get(
                                f"{BACKEND_BASE_URL}/results/{job_id}",
                                timeout=60,
                            )
                            if results_resp.ok:
                                st.session_state["last_results_payload"] = results_resp.json()
                        except Exception as exc:
                            st.error(f"Backend error: {exc}")
                            return

                st.success("Analysis complete.")
                st.session_state["last_result"] = result
                st.json(result)

    if "last_result" in st.session_state:
        with card():
            st.markdown("#### Latest result")
            st.json(st.session_state["last_result"])
            if "last_results_payload" in st.session_state:
                st.markdown("#### Latest GPS payload")
                st.json(st.session_state["last_results_payload"])
        job_id = st.session_state.get("last_job_id")
        if job_id:
            detections = fetch_job_detections(job_id) or []
            if detections:
                frames_dir = Path("data") / "outputs" / "frames" / job_id
                frame_path = frames_dir / "frame_000000.jpg"
                if frame_path.exists():
                    with card():
                        st.markdown("#### Latest annotated result")
                        st.image(str(frame_path), use_container_width=True)
                        df_det = pd.DataFrame(detections)
                        df_det = df_det.rename(
                            columns={
                                "track_id": "Elephant ID",
                                "cls_label": "Age class",
                                "conf": "Confidence",
                                "frame_index": "Frame",
                            }
                        )
                        df_show = df_det[["Elephant ID", "Age class", "Confidence", "Frame"]]
                        st.dataframe(df_show, use_container_width=True, hide_index=True)

    st.markdown("#### Recent jobs")
    if not use_mock:
        with st.expander("Admin"):
            confirm_clear = st.checkbox("Confirm clear history (DB only)")
            if st.button("Clear history", use_container_width=True):
                if confirm_clear:
                    try:
                        resp = requests.post(f"{BACKEND_BASE_URL}/admin/clear-history", timeout=10)
                        resp.raise_for_status()
                        st.session_state.pop("last_job_id", None)
                        st.session_state.pop("last_result", None)
                        st.session_state.pop("last_results_payload", None)
                        st.success("History cleared.")
                    except Exception as exc:
                        st.error(f"Clear history failed: {exc}")
                else:
                    st.warning("Check confirmation first.")
    jobs_df = fetch_jobs_from_backend() if not use_mock else None
    if jobs_df is None:
        jobs_df = mock_jobs_table() if use_mock else pd.DataFrame(
            columns=["Job ID", "Video name", "Status", "Submitted at", "Duration"]
        )
    else:
        jobs_df["Submitted at"] = jobs_df["Submitted at"].apply(format_my_time)
        jobs_df["Duration (s)"] = jobs_df["Duration (s)"].apply(format_duration)
        jobs_df = jobs_df.rename(columns={"Duration (s)": "Duration"})

    def status_badge_html(status: str) -> str:
        cls = {
            "complete": "ea-badge-complete",
            "processing": "ea-badge-processing",
            "queued": "ea-badge-queued",
            "failed": "ea-badge-failed",
        }.get(status, "")
        label = status.capitalize()
        return f"<span class='ea-pill {cls}'>{label}</span>"

    styled_rows = []
    for _, row in jobs_df.iterrows():
        styled_rows.append(
            {
                "Job ID": row["Job ID"],
                "Video name": row["Video name"],
                "Status": status_badge_html(row["Status"]),
                "Submitted at": row["Submitted at"],
                "Duration": row["Duration"],
            }
        )
    styled_df = pd.DataFrame(styled_rows)

    with card():
        st.write(styled_df.to_html(escape=False, index=False), unsafe_allow_html=True)

def timeline_page():
    st.markdown("### Timeline analysis")
    st.caption(
        "Individual elephant tracking: when each elephant appears, "
        "their age class, and whether they are alone or in a herd."
    )
    if not st.session_state.get("use_mock", False):
        jobs_df = fetch_jobs_from_backend()
        job_ids = jobs_df["Job ID"].tolist() if jobs_df is not None else []
        if job_ids:
            default_job = st.session_state.get("last_job_id") or job_ids[0]
            job_id = st.selectbox("Select job", job_ids, index=job_ids.index(default_job))
            try:
                resp = requests.get(f"{BACKEND_BASE_URL}/timeline/{job_id}", timeout=10)
                resp.raise_for_status()
                payload = resp.json()
                rows = payload.get("timeline", [])
            except Exception as exc:
                st.error(f"Timeline error: {exc}")
                rows = []
        else:
            rows = []
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
        else:
            df = empty_timeline_table()
        with card():
            st.dataframe(df, use_container_width=True, hide_index=True)
        if job_ids:
            frames_dir = Path("data") / "outputs" / "frames" / job_id
            frame_files = sorted(frames_dir.glob("frame_*.jpg")) if frames_dir.exists() else []
            if frame_files:
                with card():
                    st.markdown("#### Annotated result")
                    st.caption("Detected elephants with IDs on the selected frame.")
                    frame_indices = []
                    for path in frame_files:
                        parts = path.stem.split("_", 1)
                        if len(parts) == 2 and parts[1].isdigit():
                            frame_indices.append(int(parts[1]))
                        else:
                            frame_indices.append(0)
                    idx = st.slider("Frame index", 0, len(frame_files) - 1, 0, 1)
                    st.image(str(frame_files[idx]), use_container_width=True)
                    frame_index = frame_indices[idx]
                    detections = fetch_job_detections(job_id) or []
                    if detections:
                        df_det = pd.DataFrame(detections)
                        df_det = df_det[df_det["frame_index"] == frame_index]
                        df_det = df_det.rename(
                            columns={
                                "track_id": "Elephant ID",
                                "cls_label": "Age class",
                                "conf": "Confidence",
                                "frame_index": "Frame",
                            }
                        )
                        df_show = df_det[["Elephant ID", "Age class", "Confidence", "Frame"]]
                        st.dataframe(df_show, use_container_width=True, hide_index=True)
    else:
        df = mock_timeline_table()
        with card():
            st.dataframe(df, use_container_width=True, hide_index=True)

def map_page():
    st.markdown("### Geographic distribution")
    st.caption("Spatial distribution of detections and movement patterns.")
    map_section()
    annotated_frames_section()

def behaviour_habitat_page():
    st.markdown("### Behaviour & habitat analysis")
    charts_section()
    st.markdown("")
    timeline_and_habitat_section()


def annotated_frames_section():
    job_id = st.session_state.get("last_job_id")
    if not job_id:
        return
    frames_dir = Path("data") / "outputs" / "frames" / job_id
    if not frames_dir.exists():
        return
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_files:
        return
    with card():
        st.markdown("#### Annotated frames")
        st.caption("Detected elephants with track IDs (sampled frames).")
        idx = st.slider("Frame index", 0, len(frame_files) - 1, 0, 1)
        st.image(str(frame_files[idx]), use_container_width=True)

def about_page():
    st.markdown("### About project")
    st.write(
        """
This dashboard is part of your SEGP elephant-analytics project.

**High-level pipeline:**

1. **Upload video** from drones (or select example footage).
2. **Backend / HPC** runs YOLO detection + tracking + age / behaviour / habitat models.
3. Backend writes detection results to JSON / database.
4. This **Streamlit UI** reads the processed results and turns them into:
   - herd-size time series
   - individual timelines
   - behaviour & habitat breakdown
   - GPS movement maps
        """
    )
    st.markdown("---")
    st.write("You can edit this page to describe your team, data sources, and ethics.")

# ===================== ROUTER =====================

PAGES = {
    "Overview": "overview",
    "Upload & jobs": "upload",
    "Timeline": "timeline",
    "Map": "map",
    "Behaviour & habitat": "behaviour",
    "About project": "about",
}

with st.sidebar:
    st.title("🐘 Elephant Analytics")
    page_label = st.radio("Navigation", list(PAGES.keys()))

page = PAGES[page_label]

if page == "overview":
    hero_section()
    st.markdown("----")
    stats_cards()
    st.markdown("----")
    charts_section()
    st.markdown("----")
    timeline_and_habitat_section()
    st.markdown("----")
    map_section()
elif page == "upload":
    upload_jobs_page()
elif page == "timeline":
    timeline_page()
elif page == "map":
    map_page()
elif page == "behaviour":
    behaviour_habitat_page()
elif page == "about":
    about_page()
