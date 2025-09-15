import os
import time
import cv2
import pandas as pd
import streamlit as st
import altair as alt
from datetime import datetime
from ultralytics import YOLO
import shutil

# ===== AUTO-CREATE FOLDER + CSV =====
os.makedirs("overspeed_captures", exist_ok=True)
logbook_file = "traffic_logbook.csv"

if not os.path.exists(logbook_file):
    df = pd.DataFrame(columns=["timestamp", "vehicle_type", "speed_kmph"])
    df.to_csv(logbook_file, index=False)

# ===== Load YOLO model once =====
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # smallest model for faster inference

model = load_model()

# ===== Fake speed function (replace with real tracking later) =====
def estimate_speed():
    return round(20 + 80 * (time.time() % 1), 2)  # random 20–100 km/h

# ===== Streamlit UI =====
st.set_page_config(page_title="Traffic Monitoring System", layout="wide")
st.title("🚦 Traffic Monitoring System")

tab1, tab2, tab3 = st.tabs(["📡 Detection", "📊 Dashboard", "📷 Overspeed Gallery"])

# -------------------------------------------------------------------
with tab1:
    st.subheader("YOLO Vehicle Detection + Speed Estimation")

    mode = st.radio("Select Input Source", ["Webcam", "Upload Video"])

    if mode == "Upload Video":
        uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        if uploaded_file:
            tfile = "temp_video.mp4"
            with open(tfile, "wb") as f:
                f.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile)
        else:
            cap = None
    else:
        run_webcam = st.checkbox("Run Webcam Detection")
        cap = cv2.VideoCapture(0) if run_webcam else None

    stframe = st.empty()

    if cap:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("End of video stream.")
                break

            results = model(frame)

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    speed = estimate_speed()

                    # Draw detection
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

            # Convert frame BGR -> RGB for Streamlit
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

            if mode == "Upload Video":
                time.sleep(0.03)  # slow playback for Streamlit

        cap.release()

# -------------------------------------------------------------------
with tab2:
    st.subheader("Traffic Monitoring Dashboard")

    if os.path.exists(logbook_file):
        df = pd.read_csv(logbook_file)

        if not df.empty:
            st.write("### Latest Entries")
            st.dataframe(df.tail(10))

            # Vehicle counts
            counts = df["vehicle_type"].value_counts().reset_index()
            counts.columns = ["vehicle_type", "count"]
            chart_counts = alt.Chart(counts).mark_bar().encode(
                x="vehicle_type", y="count", tooltip=["vehicle_type", "count"]
            )
            st.altair_chart(chart_counts, use_container_width=True)

            # Speed distribution
            chart_speed = alt.Chart(df).mark_bar().encode(
                x=alt.X("speed_kmph:Q", bin=True),
                y="count()",
                tooltip=["count()"]
            )
            st.write("### Speed Distribution")
            st.altair_chart(chart_speed, use_container_width=True)

            # ===== Download Button =====
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

    # ===== Clear Data Button =====
    if st.button("🗑️ Clear All Data"):
        if os.path.exists(logbook_file):
            os.remove(logbook_file)
        if os.path.exists("overspeed_captures"):
            shutil.rmtree("overspeed_captures")
            os.makedirs("overspeed_captures", exist_ok=True)

        st.success("✅ Logbook and overspeed captures cleared! Refresh to start fresh.")

# -------------------------------------------------------------------
with tab3:
    st.subheader("📷 Overspeed Captures Gallery")

    image_files = sorted([f for f in os.listdir("overspeed_captures") if f.endswith(".jpg")])

    if image_files:
        cols = st.columns(3)
        for idx, img_file in enumerate(image_files):
            img_path = os.path.join("overspeed_captures", img_file)
            with cols[idx % 3]:
                st.image(img_path, caption=img_file, use_container_width=True)
    else:
        st.info("No overspeed captures yet. Run detection and exceed speed limit to see images.")
