import os
import time
import cv2
import pandas as pd
import streamlit as st
import altair as alt
from datetime import datetime
from ultralytics import YOLO
import shutil
from streamlit_autorefresh import st_autorefresh

# ===== AUTO-CREATE FOLDER + CSV =====
os.makedirs("overspeed_captures", exist_ok=True)
logbook_file = "traffic_logbook.csv"

if not os.path.exists(logbook_file):
    df_empty = pd.DataFrame(columns=["timestamp", "vehicle_type", "speed_kmph"])
    df_empty.to_csv(logbook_file, index=False)

# ===== Load YOLO model once =====
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # nano for faster inference

model = load_model()

# ===== Fake speed function =====
def estimate_speed():
    return round(20 + 80 * (time.time() % 1), 2)

# ===== Streamlit UI =====
st.set_page_config(page_title="Traffic Monitoring System", layout="wide")
st.title("🚦 Traffic Monitoring System")

tab1, tab2, tab3 = st.tabs(["📡 Detection", "📊 Dashboard", "📷 Overspeed Gallery"])

# -------------------------------------------------------------------
# ====================== Detection Tab ==========================
with tab1:
    st.subheader("YOLO Vehicle Detection + Speed Estimation")

    if "detecting" not in st.session_state:
        st.session_state.detecting = False

    # Status indicator
    status_text = "🟢 Running" if st.session_state.detecting else "🔴 Stopped"
    st.markdown(f"**Status:** {status_text}")

    mode = st.radio("Select Input Source", ["Webcam", "Upload Video"])

    stframe = st.empty()

    # ===== Webcam Mode =====
    if mode == "Webcam":
        if st.button("▶️ Start Webcam Detection") and not st.session_state.detecting:
            st.session_state.cap = cv2.VideoCapture(0)
            st.session_state.detecting = True

    # ===== Upload Video Mode =====
    else:
        uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        if uploaded_file and not st.session_state.detecting:
            tfile = "temp_video.mp4"
            with open(tfile, "wb") as f:
                f.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile)
            st.session_state.cap = cap
            st.session_state.detecting = True
            progress_bar = st.progress(0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_num = 0

            while st.session_state.detecting:
                ret, frame = cap.read()
                if not ret:
                    st.success("✅ Video processing completed.")
                    st.session_state.detecting = False
                    break

                results = model(frame)

                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        label = model.names[cls_id]
                        speed = estimate_speed()

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} {speed} km/h", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        # Save logbook entry
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        new_data = pd.DataFrame([[timestamp, label, speed]],
                                                columns=["timestamp", "vehicle_type", "speed_kmph"])
                        new_data.to_csv(logbook_file, mode="a", header=False, index=False)

                        # Save overspeed capture
                        if speed > 60:
                            filename = f"overspeed_captures/{timestamp.replace(':','-')}_{label}_{speed}.jpg"
                            cv2.imwrite(filename, frame)

                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

                frame_num += 1
                progress_bar.progress(min(frame_num / total_frames, 1.0))

            cap.release()
            st.session_state.detecting = False

    # ===== Stop Detection Button =====
    if st.session_state.detecting and st.button("⏹️ Stop Detection"):
        st.session_state.detecting = False
        if "cap" in st.session_state and st.session_state.cap:
            st.session_state.cap.release()
            del st.session_state.cap
        st.success("✅ Detection stopped.")

# -------------------------------------------------------------------
# ====================== Dashboard Tab ==========================
with tab2:
    st.subheader("Traffic Monitoring Dashboard")
    st_autorefresh(interval=5000, key="refresh_dashboard")

    if os.path.exists(logbook_file):
        try:
            df = pd.read_csv(logbook_file)
            expected_cols = ["timestamp", "vehicle_type", "speed_kmph"]
            if list(df.columns) != expected_cols:
                df = pd.DataFrame(columns=expected_cols)
                df.to_csv(logbook_file, index=False)
        except:
            df = pd.DataFrame(columns=["timestamp", "vehicle_type", "speed_kmph"])

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
                label="📥 Download Full Logbook (CSV)",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="traffic_logbook.csv",
                mime="text/csv"
            )
        else:
            st.warning("⚠️ Logbook is empty. Run detection first.")
    else:
        st.error("⚠️ No logbook found. Run detection first.")

    # Clear Data Button
    if st.button("🗑️ Clear All Data"):
        df_empty = pd.DataFrame(columns=["timestamp", "vehicle_type", "speed_kmph"])
        df_empty.to_csv(logbook_file, index=False)
        if os.path.exists("overspeed_captures"):
            for f in os.listdir("overspeed_captures"):
                os.remove(os.path.join("overspeed_captures", f))
        st.success("✅ Logbook reset and overspeed captures cleared!")

# -------------------------------------------------------------------
# ====================== Overspeed Gallery Tab ==========================
with tab3:
    st.subheader("📷 Overspeed Captures Gallery")
    st_autorefresh(interval=7000, key="refresh_gallery")

    image_files = sorted([f for f in os.listdir("overspeed_captures") if f.endswith(".jpg")])
    if image_files:
        cols = st.columns(3)
        for idx, img_file in enumerate(image_files):
            img_path = os.path.join("overspeed_captures", img_file)
            with cols[idx % 3]:
                st.image(img_path, caption=img_file, use_container_width=True)
    else:
        st.info("No overspeed captures yet. Run detection and exceed speed limit to see images.")
