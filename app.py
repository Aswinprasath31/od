import streamlit as st
import pandas as pd
import cv2
import os
import csv
from datetime import datetime
import altair as alt
from ultralytics import YOLO
import tempfile
import urllib.request

# =========================
# CONFIG
# =========================
CSV_FILE = "traffic_logbook.csv"
CAPTURE_FOLDER = "overspeed_captures"
SPEED_LIMIT = 60  # Default, user can adjust in sidebar
DEMO_VIDEO_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/highway.mp4"
DEMO_VIDEO_PATH = "demo_highway.mp4"

os.makedirs(CAPTURE_FOLDER, exist_ok=True)

# =========================
# LOGGING FUNCTION
# =========================
def log_vehicle(vehicle_id, vehicle_type, speed, overspeed, image_path=""):
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Timestamp", "Vehicle_ID", "Vehicle_Type", "Speed_km/h", "Overspeed", "Image_Path"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                         vehicle_id, vehicle_type, speed, overspeed, image_path])

# =========================
# VEHICLE SPEED ESTIMATION (dummy)
# =========================
def estimate_speed(box, fps=30):
    return int(40 + (box[2] - box[0]) / 5)  # dummy

# =========================
# DETECTION FUNCTION
# =========================
def run_detection(source):
    st.info("🚗 Running vehicle detection...")

    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(source)
    vehicle_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]

                if label not in ["car", "bus", "truck", "motorbike"]:
                    continue

                vehicle_id += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                speed = estimate_speed([x1, y1, x2, y2])
                overspeed = "YES" if speed > SPEED_LIMIT else "NO"
                image_path = ""

                if overspeed == "YES":
                    filename = f"{CAPTURE_FOLDER}/overspeed_{vehicle_id}_{datetime.now().strftime('%H%M%S')}.jpg"
                    cv2.imwrite(filename, frame[y1:y2, x1:x2])
                    image_path = filename

                log_vehicle(vehicle_id, label, speed, overspeed, image_path)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, channels="RGB", use_container_width=True)

        if st.session_state.get("stop_detection", False):
            break

    cap.release()

# =========================
# DASHBOARD UI
# =========================
def dashboard():
    st.subheader("📊 Traffic Monitoring Dashboard")

    if not os.path.exists(CSV_FILE):
        st.warning("⚠️ No traffic logbook found. Start detection first.")
        return

    df = pd.read_csv(CSV_FILE)
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    st.sidebar.header("🔍 Filters")
    vehicle_types = df["Vehicle_Type"].dropna().unique().tolist()
    selected_types = st.sidebar.multiselect("Vehicle Type", vehicle_types, default=vehicle_types)

    overspeed_filter = st.sidebar.selectbox("Overspeed Only?", ["All", "YES", "NO"])
    if not df.empty:
        speed_min, speed_max = st.sidebar.slider("Speed Range (km/h)",
                                                 min_value=0,
                                                 max_value=int(df["Speed_km/h"].max()),
                                                 value=(0, int(df["Speed_km/h"].max())))
    else:
        speed_min, speed_max = 0, 200

    global SPEED_LIMIT
    SPEED_LIMIT = st.sidebar.number_input("⚙️ Speed Limit (km/h)", min_value=10, max_value=200, value=60, step=5)

    filtered_df = df[df["Vehicle_Type"].isin(selected_types)]
    if overspeed_filter != "All":
        filtered_df = filtered_df[filtered_df["Overspeed"] == overspeed_filter]
    filtered_df = filtered_df[(filtered_df["Speed_km/h"] >= speed_min) & (filtered_df["Speed_km/h"] <= speed_max)]

    if not filtered_df.empty:
        filtered_df["Speed_Status"] = filtered_df["Overspeed"].apply(lambda x: "Overspeed" if x == "YES" else "Normal")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Vehicles", len(df))
    with col2:
        st.metric("Overspeeding", (df["Overspeed"] == "YES").sum())
    with col3:
        avg_speed = round(df["Speed_km/h"].mean(), 2) if not df.empty else 0
        st.metric("Average Speed", avg_speed)

    if not filtered_df.empty:
        chart1 = alt.Chart(filtered_df).mark_bar().encode(
            x="Vehicle_Type:N", y="count()", color="Vehicle_Type:N"
        ).properties(title="Vehicle Count by Type")
        st.altair_chart(chart1, use_container_width=True)

        chart2 = alt.Chart(filtered_df).mark_bar().encode(
            x=alt.X("Speed_km/h:Q", bin=alt.Bin(maxbins=20)),
            y="count()",
            color=alt.Color("Speed_Status:N", scale=alt.Scale(domain=["Normal", "Overspeed"],
                                                              range=["green", "red"]))
        ).properties(title="Speed Distribution")
        st.altair_chart(chart2, use_container_width=True)

    st.subheader("📑 Vehicle Logbook (last 30)")
    st.dataframe(filtered_df.tail(30), use_container_width=True)

    if not filtered_df.empty:
        st.download_button(
            label="💾 Download Filtered Logbook",
            data=filtered_df.to_csv(index=False).encode("utf-8"),
            file_name="filtered_logbook.csv",
            mime="text/csv"
        )

    st.subheader("🚨 Overspeed Evidence Gallery")
    overspeed_df = filtered_df[filtered_df["Overspeed"] == "YES"].dropna(subset=["Image_Path"])
    if not overspeed_df.empty:
        cols = st.columns(3)
        for i, row in overspeed_df.tail(9).iterrows():
            with cols[i % 3]:
                if os.path.exists(row["Image_Path"]):
                    st.image(row["Image_Path"], caption=f"{row['Vehicle_ID']} | {row['Speed_km/h']} km/h", use_container_width=True)

        st.download_button(
            label="💾 Download Overspeed Evidence Report",
            data=overspeed_df.to_csv(index=False).encode("utf-8"),
            file_name="overspeed_report.csv",
            mime="text/csv"
        )
    else:
        st.info("No overspeeding vehicles yet 🚘💨")

# =========================
# MAIN APP
# =========================
st.title("🚦 Traffic Monitoring System")

mode = st.sidebar.radio("Choose Mode:", ["Dashboard", "Run Detection"])

if mode == "Run Detection":
    st.sidebar.subheader("🎥 Detection Source")
    source_choice = st.sidebar.radio("Select source:", ["Webcam", "Upload Video", "Demo Video"])

    if source_choice == "Webcam":
        if st.button("▶ Start Webcam Detection"):
            st.session_state["stop_detection"] = False
            run_detection(0)
        if st.button("⏹ Stop Detection"):
            st.session_state["stop_detection"] = True

    elif source_choice == "Upload Video":
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
                tmpfile.write(uploaded_file.read())
                tmp_path = tmpfile.name
            if st.button("▶ Start Video Detection"):
                run_detection(tmp_path)

    elif source_choice == "Demo Video":
        if not os.path.exists(DEMO_VIDEO_PATH):
            st.info("📥 Downloading demo video...")
            urllib.request.urlretrieve(DEMO_VIDEO_URL, DEMO_VIDEO_PATH)
        if st.button("▶ Start Demo Video Detection"):
            run_detection(DEMO_VIDEO_PATH)

else:
    dashboard()
