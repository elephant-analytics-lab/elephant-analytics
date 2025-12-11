import streamlit as st
import pandas as pd
import numpy as np
import time
import io
import requests
import json
from datetime import timedelta

# ---------- Config ----------
BACKEND_PREDICT_URL = "https://your-backend.example.com/predict"  # replace with your real API
MAX_VIDEO_MB = 500

st.set_page_config(page_title="Elephant Detector (Frontend)", layout="wide")

# ---------- Upload Section ----------
st.title("Elephant detection from drone footage")
st.write(
    "Upload a drone video. The app will send it to the backend for analysis and display results such as locations, counts, timestamps, behaviours, and habitat type."
)

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader(
        f"Upload video file (mp4, mov, avi). Max: {MAX_VIDEO_MB} MB",
        type=["mp4", "mov", "avi"],
        accept_multiple_files=False,
        help="GPS metadata should be present in the filename or accompanying JSON if available."
    )
    gps_input = st.text_input("(Optional) GPS location for clip (lat, lon)", "")

with col2:
    st.markdown("### Controls")
    process_button = st.button("Process / Analyze", disabled=(uploaded_file is None))
    st.button("Clear", on_click=lambda: st.experimental_rerun())

# ---------- Example/mock data toggle ----------
use_mock = st.checkbox("Use mock results (no backend)", value=True)

output_placeholder = st.empty()
progress = st.empty()

# ---------- Mock response generator ----------
def make_mock_response():
    return {
        "video_name": getattr(uploaded_file, "name", "demo_video.mp4"),
        "duration_s": 185,
        "gps_start": gps_input or "-1.2921,36.8219",
        "detections": [
            {"id": 1, "timestamp_s": 10, "bbox": [100, 50, 200, 150], "age": "adult", "behaviour": "grazing", "lat": -1.2921, "lon": 36.8219},
            {"id": 2, "timestamp_s": 12, "bbox": [250, 60, 350, 170], "age": "juvenile", "behaviour": "traveling", "lat": -1.2922, "lon": 36.8220},
            {"id": 1, "timestamp_s": 40, "bbox": [110, 55, 210, 160], "age": "adult", "behaviour": "travelling", "lat": -1.29215, "lon": 36.82195},
        ],
        "summary": {
            "total_elephants_unique": 2,
            "counts_by_age": {"adult": 1, "juvenile": 1, "baby": 0},
            "herd_sizes": [2],
            "habitat": "savanna",
        },
    }

# ---------- Processing ----------
if process_button:
    if uploaded_file is None and not use_mock:
        st.error("No file uploaded and mock disabled. Upload a video or enable mock mode.")
    else:
        with st.spinner("Processing video..."):
            for i in range(3):
                progress.info(f"Step {i+1}/3: uploading/processing...")
                time.sleep(0.5)

            if use_mock:
                response = make_mock_response()
            else:
                files = {"video": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                data = {"gps_start": gps_input}
                try:
                    r = requests.post(BACKEND_PREDICT_URL, files=files, data=data, timeout=600)
                    r.raise_for_status()
                    response = r.json()
                except Exception as e:
                    st.error(f"Error contacting backend: {e}")
                    response = None

        if response:
            progress.success("Processing complete — displaying results")

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Unique elephants", response["summary"]["total_elephants_unique"]) 
            s2.metric("Adults", response["summary"]["counts_by_age"].get("adult", 0))
            s3.metric("Juveniles", response["summary"]["counts_by_age"].get("juvenile", 0))
            s4.metric("Babies", response["summary"]["counts_by_age"].get("baby", 0))

            st.subheader("Detections (timeline)")
            df = pd.DataFrame(response["detections"]) 
            df["time_hhmmss"] = pd.to_timedelta(df["timestamp_s"], unit="s")
            st.dataframe(df[["id", "time_hhmmss", "age", "behaviour", "lat", "lon"]])

            st.subheader("Map of detection locations")
            map_df = df[["lat", "lon"]].dropna()
            if not map_df.empty:
                st.map(map_df)
            else:
                st.write("No GPS coordinates found for detections")

            st.subheader("Herd size timeline (simple)")
            timeline = df.groupby("timestamp_s").size().reset_index(name="count")
            st.line_chart(timeline.set_index("timestamp_s"))

            st.subheader("Behaviour distribution")
            behaviour_counts = df["behaviour"].value_counts()
            st.bar_chart(behaviour_counts)

            st.download_button(
	    	"Download results (JSON)",
		data=json.dumps(response, indent=4),
		file_name="analysis_results.json",
		mime="application/json"
	    )

            if uploaded_file is not None:
                st.subheader("Uploaded video preview")
                st.video(uploaded_file.getvalue())

            st.session_state["last_response"] = response

# ---------- If idle ----------
if not process_button:
    st.info("Upload a video and press 'Process / Analyze'. Use mock results for demos.")

# ---------- Developer Notes ----------
st.sidebar.markdown("---")
st.sidebar.header("Developer Notes")
st.sidebar.write(
    "This frontend expects a backend API POST /predict that accepts a video file and optional GPS string. "
    "Backend should return a JSON containing: video_name, duration_s, detections (list), summary (dict)."
)
