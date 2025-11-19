import streamlit as st

st.set_page_config(page_title="Elephant Detection Demo", page_icon="🐘", layout="wide")

# --- Sidebar ---
st.sidebar.title("🐘 Elephant Detection")
st.sidebar.write("Prototype interface for the Software Engineering project.")
st.sidebar.markdown("---")
st.sidebar.write("Created by Raihan.")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📤 Upload Video", "📊 Job Status", "📁 Results"])

# ---------------------------------
# TAB 1 — UPLOAD
# ---------------------------------
with tab1:
    st.header("Upload a Video for Detection")

    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=["mp4", "mov", "avi"],
        help="Upload any wildlife video to detect elephants."
    )

    st.subheader("Detection Settings")
    confidence = st.slider("Detection Confidence", 0.0, 1.0, 0.5)
    draw_boxes = st.checkbox("Draw Bounding Boxes", value=True)
    track_elephants = st.checkbox("Enable Tracking", value=True)

    if st.button("Submit Job"):
        if uploaded_video:
            st.success("Video submitted successfully! (Demo only)")
            st.info("In real system: FastAPI will receive this, Celery will queue it, YOLO will detect elephants.")
        else:
            st.error("Please upload a video first.")

    st.markdown("---")
    st.caption("This is only a mock interface. Backend will be connected later.")

# ---------------------------------
# TAB 2 — JOB STATUS
# ---------------------------------
with tab2:
    st.header("Check Job Status")

    job_id = st.text_input("Enter Job ID")

    if st.button("Check Status"):
        if job_id.strip() == "":
            st.error("Please enter a Job ID.")
        else:
            st.info(f"Job ID {job_id}: Currently processing... (Demo only)")

    st.markdown("---")
    st.caption("Once backend is ready, this will show real job progress.")

# ---------------------------------
# TAB 3 — RESULTS
# ---------------------------------
with tab3:
    st.header("Detection Results")

    job_id_results = st.text_input("Enter Job ID to load results")

    if st.button("Load Results"):
        if job_id_results.strip() == "":
            st.error("Please enter a Job ID.")
        else:
            st.success(f"Results for Job {job_id_results} loaded! (Demo only)")
            st.write("Fake results preview:")
            st.image("https://upload.wikimedia.org/wikipedia/commons/3/37/African_Bush_Elephant.jpg", caption="Example Elephant Detection")

    st.markdown("---")
    st.caption("Later, this will show output video and detection stats.")


