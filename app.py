import os
import time
import cv2
import pandas as pd
import streamlit as st
import altair as alt
from datetime import datetime
from ultralytics import YOLO
import re  # for robust speed parsing

# ===== AUTO-CREATE FOLDER + CSV =====
os.makedirs("overspeed_captures", exist_ok=True)
logbook_file = "traffic_logbook.csv"

if not os.path.exists(logbook_file):
    df_empty = pd.DataFrame(columns=["timestamp", "vehicle_type", "speed_kmph"])
    df_empty.to_csv(logbook_file, index=False)

# ===== Load YOLO model once =====
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # nano model for speed

model = load_model()

# ===== Initialize session state =====
if "detecting" not in st.session_state:
    st.session_state.detecting = False
if "uploaded_video" not in st.session_state:
    st.session_state.uploaded_video = None
if "cap" not in st.session_state:
    st.session_state.cap = None
if "frame_num" not in st.session_state:
    st.session_state.frame_num = 0
if "vehicle_tracks" not in st.session_state:
    st.session_state.vehicle_tracks = {}
if "vehicle_id_counter" not in st.session_state:
    st.session_state.vehicle_id_counter = 0
if "vehicle_speeds" not in st.session_state:
    st.session_state.vehicle_speeds = {}  # {vehicle_id: [speed1, speed2, ...]}

# ===== Streamlit UI =====
st.set_page_config(page_title="Traffic Monitoring System", layout="wide")
st.title("🚦 Traffic Monitoring System")

tab1, tab2, tab3 = st.tabs(["📡 Detection", "📊 Dashboard", "📷 Overspeed Gallery"])

# ====================== Detection Tab ==========================
with tab1:
    st.subheader("YOLO Vehicle Detection + Speed Estimation")
    status_text = "🟢 Running" if st.session_state.detecting else "🔴 Stopped"
    st.markdown(f"**Status:** {status_text}")

    mode = st.radio("Select Input Source", ["Webcam", "Upload Video"])
    stframe = st.empty()

    speed_limit = 60  # km/h

    # ===== Speed Calibration Slider =====
    meters_per_pixel = st.slider(
        "Speed Calibration (meters per pixel)",
        min_value=0.01,
        max_value=0.2,
        value=0.05,
        step=0.005,
        help="Adjust this based on camera distance and field of view for accurate speeds."
    )
    st.markdown(f"**Current calibration:** {meters_per_pixel} meters/pixel")

    # Webcam
    if mode == "Webcam":
        if st.button("▶️ Start Webcam Detection") and not st.session_state.detecting:
            st.session_state.cap = cv2.VideoCapture(0)
            st.session_state.detecting = True

    # Upload Video
    else:
        uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        if uploaded_file:
            st.session_state.uploaded_video = uploaded_file
            st.session_state.frame_num = 0
        if st.session_state.uploaded_video and not st.session_state.detecting:
            tfile = "temp_video.mp4"
            with open(tfile, "wb") as f:
                f.write(st.session_state.uploaded_video.read())
            cap = cv2.VideoCapture(tfile)
            st.session_state.cap = cap
            st.session_state.detecting = True
            progress_bar = st.progress(0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Detection Loop
    if st.session_state.detecting and st.session_state.cap:
        cap = st.session_state.cap
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_time = 1 / fps

        progress_bar = st.empty() if mode == "Upload Video" else None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if mode == "Upload Video" else None

        while st.session_state.detecting:
            ret, frame = cap.read()
            if not ret:
                st.success("✅ Detection finished.")
                st.session_state.detecting = False
                st.session_state.cap = None
                break

            results = model(frame)
            new_tracks = {}

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]

                    # Centroid
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    centroid = (cx, cy)

                    # Track matching
                    matched_id = None
                    min_dist = float("inf")
                    for vid, prev_c in st.session_state.vehicle_tracks.items():
                        dist = ((cx - prev_c[0])**2 + (cy - prev_c[1])**2)**0.5
                        if dist < min_dist and dist < 50:
                            min_dist = dist
                            matched_id = vid
                    if matched_id is None:
                        st.session_state.vehicle_id_counter += 1
                        matched_id = st.session_state.vehicle_id_counter

                    # Speed calculation
                    if matched_id in st.session_state.vehicle_tracks:
                        prev_c = st.session_state.vehicle_tracks[matched_id]
                        pixel_distance = ((cx - prev_c[0])**2 + (cy - prev_c[1])**2)**0.5
                        distance_m = pixel_distance * meters_per_pixel
                        speed_mps = distance_m / frame_time
                        speed_kmph_raw = speed_mps * 3.6
                    else:
                        speed_kmph_raw = 0

                    # ---- 3-frame moving average ----
                    if matched_id not in st.session_state.vehicle_speeds:
                        st.session_state.vehicle_speeds[matched_id] = []
                    st.session_state.vehicle_speeds[matched_id].append(speed_kmph_raw)
                    st.session_state.vehicle_speeds[matched_id] = st.session_state.vehicle_speeds[matched_id][-3:]
                    speed_kmph = sum(st.session_state.vehicle_speeds[matched_id]) / len(st.session_state.vehicle_speeds[matched_id])

                    new_tracks[matched_id] = centroid

                    # Draw box + speed
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {int(speed_kmph)} km/h", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Save logbook
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    pd.DataFrame([[timestamp, label, int(speed_kmph)]],
                                 columns=["timestamp", "vehicle_type", "speed_kmph"]).to_csv(
                                 logbook_file, mode="a", header=False, index=False)

                    # Overspeed capture
                    if speed_kmph > speed_limit:
                        filename = f"overspeed_captures/{timestamp.replace(':','-')}_{label}_{int(speed_kmph)}.jpg"
                        cv2.imwrite(filename, frame)

            st.session_state.vehicle_tracks = new_tracks
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

            # Progress
            if mode == "Upload Video" and progress_bar:
                st.session_state.frame_num += 1
                progress_bar.progress(min(st.session_state.frame_num / total_frames, 1.0))

    if st.session_state.detecting and st.button("⏹️ Stop Detection"):
        st.session_state.detecting = False
        if st.session_state.cap:
            st.session_state.cap.release()
            st.session_state.cap = None
        st.success("✅ Detection stopped.")

# ====================== Dashboard Tab ==========================
with tab2:
    st.subheader("Traffic Monitoring Dashboard")
    try:
        st.autorefresh(interval=5000, key="refresh_dashboard")
    except AttributeError:
        st.info("Manual refresh required (Streamlit >=1.26).")

    if os.path.exists(logbook_file):
        df = pd.read_csv(logbook_file)
        if not df.empty:
            st.write("### Latest Entries")
            st.dataframe(df.tail(10))

            if "vehicle_type" in df.columns:
                counts = df["vehicle_type"].value_counts().reset_index()
                counts.columns = ["vehicle_type", "count"]
                chart_counts = alt.Chart(counts).mark_bar().encode(
                    x="vehicle_type", y="count", tooltip=["vehicle_type", "count"]
                )
                st.altair_chart(chart_counts, use_container_width=True)

            if "speed_kmph" in df.columns:
                chart_speed = alt.Chart(df).mark_bar().encode(
                    x=alt.X("speed_kmph:Q", bin=True),
                    y="count()",
                    tooltip=["count()"]
                )
                st.write("### Speed Distribution")
                st.altair_chart(chart_speed, use_container_width=True)

            st.download_button(
                "📥 Download Full Logbook (CSV)",
                df.to_csv(index=False).encode("utf-8"),
                file_name="traffic_logbook.csv",
                mime="text/csv"
            )
        else:
            st.warning("⚠️ Logbook is empty. Run detection first.")
    else:
        st.error("⚠️ No logbook found. Run detection first.")

    if st.button("🗑️ Clear All Data"):
        pd.DataFrame(columns=["timestamp", "vehicle_type", "speed_kmph"]).to_csv(logbook_file, index=False)
        for f in os.listdir("overspeed_captures"):
            os.remove(os.path.join("overspeed_captures", f))
        st.session_state.frame_num = 0
        st.success("✅ Logbook reset and overspeed captures cleared!")

# ====================== Overspeed Gallery Tab ==========================
with tab3:
    st.subheader("📷 Overspeed Captures Gallery")
    try:
        st.autorefresh(interval=7000, key="refresh_gallery")
    except AttributeError:
        st.info("Manual refresh required (Streamlit >=1.26).")

    image_files = []
    for f in os.listdir("overspeed_captures"):
        if f.endswith(".jpg"):
            m = re.search(r"_(\d+)\.jpg$", f)
            if m:
                speed = int(m.group(1))
                if speed > speed_limit:
                    image_files.append(f)
    image_files = sorted(image_files)

    if image_files:
        for img_file in image_files:
            img_path = os.path.join("overspeed_captures", img_file)
            st.image(img_path, caption=img_file, use_container_width=True)
            if st.button(f"🗑️ Delete {img_file}"):
                os.remove(img_path)
                st.experimental_rerun()
    else:
        st.info("No overspeed captures yet. Run detection and exceed 60 km/h to see images.")
