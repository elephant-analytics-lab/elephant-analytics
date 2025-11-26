# streamlit_elephant_frontend.py
import streamlit as st
import pandas as pd
import time, io, json, requests

# ---------- Config ----------
BACKEND_BASE_URL = "http://127.0.0.1:8000"  # change if your API is elsewhere
CREATE_URL   = f"{BACKEND_BASE_URL}/api/jobs"
STATUS_URL   = f"{BACKEND_BASE_URL}/api/jobs/{{job_id}}"
RESULTS_URL  = f"{BACKEND_BASE_URL}/api/jobs/{{job_id}}/results"
ARTIFACT_URL = f"{BACKEND_BASE_URL}/api/jobs/{{job_id}}/artifact/{{name}}"

MAX_VIDEO_MB  = 500
POLL_INTERVAL = 1.0
MAX_POLL_SEC  = 900

# ---------- Backend helpers ----------
def submit_job(uploaded_file, gps_text: str | None):
    """POST /api/jobs with the video and optional GPS; return job_id."""
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(),
                      uploaded_file.type or "application/octet-stream")}
    data = {}
    if gps_text and gps_text.strip():
        try:
            lat, lon = [x.strip() for x in gps_text.split(",", 1)]
            data["lat"] = float(lat); data["lon"] = float(lon)
        except Exception:
            pass
    r = requests.post(CREATE_URL, files=files, data=data, timeout=180)
    r.raise_for_status()
    return r.json()["job_id"]

def wait_until_done(job_id: str, progress_slot):
    """Poll GET /api/jobs/{job_id} until status is done/error/timeout."""
    elapsed = 0.0
    while elapsed < MAX_POLL_SEC:
        r = requests.get(STATUS_URL.format(job_id=job_id), timeout=10)
        r.raise_for_status()
        s = r.json()
        pct = s.get("progress", 0)
        msg = s.get("message", "")
        progress_slot.info(f"Status: {s['status']} ({pct}%) {msg}")
        if s["status"] == "done":
            return True, s
        if s["status"] == "error":
            return False, s
        time.sleep(POLL_INTERVAL); elapsed += POLL_INTERVAL
    return False, {"status": "timeout", "message": "processing timeout"}

def fetch_results(job_id: str):
    """GET /api/jobs/{job_id}/results (metadata + artifact URLs)."""
    r = requests.get(RESULTS_URL.format(job_id=job_id), timeout=120)
    r.raise_for_status()
    return r.json()

# ---------- Mock (optional) ----------
def make_mock_response(uploaded_file, gps_input):
    return {
        "video_name": getattr(uploaded_file, "name", "demo_video.mp4"),
        "duration_s": 185,
        "summary": {
            "total_elephants_unique": 2,
            "counts_by_age": {"adult": 1, "juvenile": 1, "baby": 0},
            "herd_sizes": [2],
            "habitat": "savanna",
            "fps": 25, "width": 1280, "height": 720
        },
        "detections": [
            {"id": 1, "timestamp_s": 10, "bbox": [100, 50, 200, 150],
             "age": "adult", "behaviour": "grazing", "lat": None, "lon": None},
            {"id": 2, "timestamp_s": 12, "bbox": [250, 60, 350, 170],
             "age": "juvenile", "behaviour": "traveling", "lat": None, "lon": None},
        ],
    }

# ---------- UI ----------
def run_app():
    st.set_page_config(page_title="Elephant Detector (Frontend)", layout="wide")

    st.title("Elephant detection from drone footage")
    st.write("Upload a drone video. The app will send it to the backend for analysis and display results.")

    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded_file = st.file_uploader(
            f"Upload video file (mp4, mov, avi). Max: {MAX_VIDEO_MB} MB",
            type=["mp4", "mov", "avi"],
            accept_multiple_files=False,
            help="Optional GPS: type as 'lat, lon' below."
        )
        gps_input = st.text_input("(Optional) GPS location for clip (lat, lon)", "")
    with col2:
        st.markdown("### Controls")
        process_button = st.button("Process / Analyze", disabled=(uploaded_file is None))
        st.button("Clear", on_click=lambda: st.experimental_rerun())

    use_mock = st.checkbox("Use mock results (no backend)", value=False)

    progress = st.empty()

    if process_button:
        if uploaded_file is None and not use_mock:
            st.error("No file uploaded and mock disabled.")
            return

        with st.spinner("Processing video..."):
            if use_mock:
                response = make_mock_response(uploaded_file, gps_input)
                annotated_url = dets_url = tracks_url = None
                dets_df = pd.DataFrame(response["detections"])
                tracks_df = pd.DataFrame()
                fps = float(response["summary"].get("fps", 25))
            else:
                try:
                    # 1) submit job
                    job_id = submit_job(uploaded_file, gps_input)
                    progress.info(f"Job submitted: {job_id}")

                    # 2) poll until done
                    ok, status_obj = wait_until_done(job_id, progress)
                    if not ok:
                        raise RuntimeError(f"Backend status: {status_obj.get('status')} {status_obj.get('message','')}")

                    # 3) fetch results JSON
                    results_json = fetch_results(job_id)
                    response = {"summary": results_json.get("summary", {}),
                                "video_name": results_json.get("video_name", "")}

                    # Artifact URLs (served by backend)
                    annotated_url = BACKEND_BASE_URL + results_json["annotated_video_url"]
                    dets_url      = BACKEND_BASE_URL + results_json["detections_url"]
                    tracks_url    = BACKEND_BASE_URL + results_json["tracks_url"]

                    # Download CSVs for charts
                    dets_bytes   = requests.get(dets_url, timeout=120).content
                    tracks_bytes = requests.get(tracks_url, timeout=120).content
                    dets_df   = pd.read_csv(io.BytesIO(dets_bytes))
                    tracks_df = pd.read_csv(io.BytesIO(tracks_bytes))

                    # Derive timestamps by frame using fps from summary (fallback 25)
                    fps = float(response["summary"].get("fps", 25.0))
                    if "frame" in dets_df.columns and "timestamp_s" not in dets_df.columns:
                        dets_df["timestamp_s"] = dets_df["frame"] / max(1.0, fps)
                    if "frame" in tracks_df.columns and "timestamp_s" not in tracks_df.columns:
                        tracks_df["timestamp_s"] = tracks_df["frame"] / max(1.0, fps)

                    # Build a lightweight "detections" list for the existing UI
                    if not tracks_df.empty and {"x1","y1","x2","y2"}.issubset(tracks_df.columns):
                        response["detections"] = [
                            {"id": int(getattr(r, "track_id", 0)),
                             "timestamp_s": float(getattr(r, "timestamp_s", getattr(r, "frame", 0) / max(1.0, fps))),
                             "bbox": [int(r.x1), int(r.y1), int(r.x2), int(r.y2)],
                             "age": "", "behaviour": "", "lat": None, "lon": None}
                            for r in tracks_df.itertuples(index=False)
                        ]
                    else:
                        response["detections"] = []

                except Exception as e:
                    st.error(f"Error contacting backend: {e}")
                    return

        progress.success("Processing complete — displaying results")

        # ----- Summary metrics -----
        summary = response.get("summary", {})
        # Support either 'total_elephants_unique' or 'unique_elephants'
        uniques = summary.get("total_elephants_unique", summary.get("unique_elephants", 0))
        adults = summary.get("counts_by_age", {}).get("adult", 0)
        juveniles = summary.get("counts_by_age", {}).get("juvenile", 0)
        babies = summary.get("counts_by_age", {}).get("baby", 0)

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Unique elephants", int(uniques))
        s2.metric("Adults", int(adults))
        s3.metric("Juveniles", int(juveniles))
        s4.metric("Babies", int(babies))

        # ----- Timeline table -----
        st.subheader("Detections (timeline)")
        det_list = response.get("detections", [])
        if det_list:
            df = pd.DataFrame(det_list)
            df["time_hhmmss"] = pd.to_timedelta(df["timestamp_s"], unit="s")
            st.dataframe(df[["id", "time_hhmmss", "bbox", "age", "behaviour", "lat", "lon"]])
        else:
            st.info("No per-detection list available (using CSV summaries only).")

        # ----- Simple charts from CSVs -----
        st.subheader("Herd size timeline (by frame)")
        if not tracks_df.empty and "timestamp_s" in tracks_df.columns:
            herds = tracks_df.groupby("timestamp_s").track_id.nunique().reset_index(name="count")
            st.line_chart(herds.set_index("timestamp_s"))
        else:
            st.write("No track data for timeline.")

        st.subheader("Behaviour distribution")
        if det_list and "behaviour" in pd.DataFrame(det_list).columns:
            behaviour_counts = pd.DataFrame(det_list)["behaviour"].value_counts()
            st.bar_chart(behaviour_counts)
        else:
            st.write("No behaviour labels in this run.")

        # ----- Artifacts (downloads + video) -----
        if not use_mock:
            st.subheader("Outputs")
            try:
                c1, c2 = st.columns(2)
                with c1:
                    st.download_button("Download dets.csv", dets_bytes, "dets.csv", "text/csv")
                with c2:
                    st.download_button("Download tracks.csv", tracks_bytes, "tracks.csv", "text/csv")
            except Exception:
                st.warning("CSV artifacts not available for download.")

            st.subheader("Annotated video")
            st.video(annotated_url)

        # Keep last response in session
        st.session_state["last_response"] = response

    else:
        st.info("Upload a video and press 'Process / Analyze'. Uncheck mock to use the backend.")

    # Developer notes
    st.sidebar.markdown("---")
    st.sidebar.header("Developer Notes")
    st.sidebar.write(
        "POST /api/jobs -> returns job_id; poll GET /api/jobs/{id}; fetch GET /api/jobs/{id}/results "
        "to get artifact URLs (annotated.mp4, dets.csv, tracks.csv)."
    )

if __name__ == "__main__":
    run_app()
