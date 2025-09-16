import os
import re
import cv2
import pandas as pd
import streamlit as st
import altair as alt
from datetime import datetime
from ultralytics import YOLO

# External OCR
try:
    import easyocr
except Exception:
    easyocr = None

# ===== Auto-create paths =====
os.makedirs("overspeed_captures", exist_ok=True)
logbook_file = "traffic_logbook.csv"

# Ensure CSV has plate_number column
if not os.path.exists(logbook_file):
    pd.DataFrame(columns=["timestamp", "vehicle_type", "speed_kmph", "plate_number"]).to_csv(logbook_file, index=False)

# ===== Load models (cached) =====
@st.cache_resource
def load_yolo_model(model_name="yolov8n.pt"):
    try:
        return YOLO(model_name)
    except Exception as e:
        return None

# Primary detection model (vehicles)
model = load_yolo_model("yolov8n.pt")  # change if you use a custom weights file

# Try load a license-plate specific yolo model if you have it locally.
# Common custom names: "yolov8n-license-plate.pt" or "license_plate.pt" — change path if available.
@st.cache_resource
def load_plate_model():
    # Try a few common names; if none available return None
    for candidate in ("yolov8n-license-plate.pt", "yolov8n-lp.pt", "license_plate.pt"):
        try:
            if os.path.exists(candidate):
                return YOLO(candidate)
        except Exception:
            pass
    # If you don't have a local plate model, return None (we'll fallback to EasyOCR directly)
    return None

plate_model = load_plate_model()

# EasyOCR reader (fallback for plate OCR / direct OCR)
@st.cache_resource
def load_easyocr_reader():
    if easyocr is None:
        return None
    try:
        # english is usually fine for many plates; add other langs if needed
        reader = easyocr.Reader(['en'], gpu=False)  # set gpu=True if available
        return reader
    except Exception:
        return None

ocr_reader = load_easyocr_reader()

# ===== Initialize Streamlit session state =====
defaults = {
    "detecting": False,
    "uploaded_video": None,
    "cap": None,
    "frame_num": 0,
    "vehicle_tracks": {},
    "vehicle_id_counter": 0,
    "vehicle_speeds": {}
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ===== UI config =====
st.set_page_config(page_title="Traffic Monitoring + Plate OCR", layout="wide")
st.title("🚦 Traffic Monitoring + Number-Plate OCR")

tab1, tab2, tab3 = st.tabs(["📡 Detection", "📊 Dashboard", "📷 Overspeed Gallery"])
speed_limit = 60  # km/h

# -------------------------
# Helper functions
# -------------------------
def sanitize_plate(text: str) -> str:
    """Sanitize plate text to a safe filename segment (alphanumeric + underscore)."""
    if not text:
        return "UNKNOWN"
    text = str(text).upper()
    # keep only alnum and replace others with underscore
    return re.sub(r"[^A-Z0-9]", "_", text)

def safe_extract_speed_plate_from_filename(filename: str):
    """
    Extract speed and plate from filename if possible.
    filename format we use: <timestamp>_<vehicle>_<speed>_<plate>.jpg
    but timestamps or vehicle names may include underscores; so use regex to get last groups.
    """
    speed = 0
    plate = ""
    try:
        # match last numeric group (speed) and optional plate before .jpg
        m = re.search(r"_(\d+)(?:_([A-Z0-9_]+))?\.jpg$", filename, flags=re.IGNORECASE)
        if m:
            speed = int(m.group(1))
            plate = m.group(2) if m.group(2) else ""
    except Exception:
        speed = 0
        plate = ""
    return speed, plate

def ocr_on_image(crop_bgr):
    """Run OCR on a BGR image via EasyOCR reader if available. Returns best text or ''."""
    if ocr_reader is None:
        return ""
    try:
        # EasyOCR expects RGB numpy array
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        result = ocr_reader.readtext(crop_rgb)
        # pick the longest text result (heuristic)
        if not result:
            return ""
        texts = [r[1] for r in result if r and len(r[1].strip()) > 0]
        if not texts:
            return ""
        # sort by length descending
        texts.sort(key=lambda t: len(t), reverse=True)
        return texts[0]
    except Exception:
        return ""

# -------------------------
# Detection Tab
# -------------------------
with tab1:
    st.subheader("YOLO Vehicle Detection + Speed Estimation + Plate OCR")
    status_text = "🟢 Running" if st.session_state.detecting else "🔴 Stopped"
    st.markdown(f"**Status:** {status_text}")

    mode = st.radio("Select Input Source", ["Webcam", "Upload Video"])
    stframe = st.empty()

    # Speed calibration slider
    meters_per_pixel = st.slider(
        "Speed Calibration (meters per pixel)",
        min_value=0.01,
        max_value=0.2,
        value=0.05,
        step=0.005,
        help="Adjust based on camera distance/FOV so pixel movement -> meters is accurate."
    )
    st.markdown(f"**Current calibration:** {meters_per_pixel} meters/pixel")

    # Start webcam
    if mode == "Webcam":
        if st.button("▶️ Start Webcam Detection") and not st.session_state.detecting:
            st.session_state.cap = cv2.VideoCapture(0)
            st.session_state.detecting = True

    # Upload video
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
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress_bar = st.progress(0)

    # Detection loop
    if st.session_state.detecting and st.session_state.cap:
        cap = st.session_state.cap
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_time = 1.0 / fps if fps > 0 else 1.0 / 30
        progress_bar = st.empty() if mode == "Upload Video" else None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if mode == "Upload Video" else None

        while st.session_state.detecting:
            ret, frame = cap.read()
            if not ret:
                st.success("✅ Detection finished.")
                st.session_state.detecting = False
                st.session_state.cap = None
                break

            # run YOLO vehicle detection (if model available)
            vehicle_results = None
            if model is not None:
                try:
                    vehicle_results = model(frame)
                except Exception:
                    vehicle_results = None

            # fallback: if model is None, skip detections
            if vehicle_results is None:
                stframe.image(frame[:, :, ::-1], channels="RGB")
                st.warning("Vehicle detection model not loaded. Install ultralytics and provide model weights.")
                break

            new_tracks = {}
            for res in vehicle_results:  # results per image (usually 1)
                for box in res.boxes:
                    try:
                        cls_id = int(box.cls[0])
                        label = model.names[cls_id] if model and hasattr(model, "names") else f"class_{cls_id}"
                    except Exception:
                        label = "vehicle"

                    # bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(frame.shape[1]-1, x2); y2 = min(frame.shape[0]-1, y2)

                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    centroid = (cx, cy)

                    # tracking: simple nearest-centroid matching
                    matched_id = None
                    min_dist = float("inf")
                    for vid, prev_c in st.session_state.vehicle_tracks.items():
                        dist = ((cx - prev_c[0])**2 + (cy - prev_c[1])**2) ** 0.5
                        if dist < min_dist and dist < 50:  # pixel threshold
                            min_dist = dist
                            matched_id = vid

                    if matched_id is None:
                        st.session_state.vehicle_id_counter += 1
                        matched_id = st.session_state.vehicle_id_counter

                    # compute speed
                    if matched_id in st.session_state.vehicle_tracks:
                        prev_c = st.session_state.vehicle_tracks[matched_id]
                        pixel_distance = ((cx - prev_c[0])**2 + (cy - prev_c[1])**2) ** 0.5
                        distance_m = pixel_distance * meters_per_pixel
                        speed_kmph_raw = (distance_m / frame_time) * 3.6
                    else:
                        speed_kmph_raw = 0.0

                    # moving average smoothing (3 frames)
                    if matched_id not in st.session_state.vehicle_speeds:
                        st.session_state.vehicle_speeds[matched_id] = []
                    st.session_state.vehicle_speeds[matched_id].append(speed_kmph_raw)
                    st.session_state.vehicle_speeds[matched_id] = st.session_state.vehicle_speeds[matched_id][-3:]
                    speed_kmph = sum(st.session_state.vehicle_speeds[matched_id]) / len(st.session_state.vehicle_speeds[matched_id])

                    new_tracks[matched_id] = centroid

                    # Plate detection + OCR
                    plate_text = "UNKNOWN"
                    try:
                        # crop vehicle region for plate detection/ocr
                        vehicle_crop = frame[y1:y2, x1:x2]
                        if vehicle_crop.size > 0:
                            # if a plate-specific YOLO model is available, run it on the vehicle crop
                            if plate_model is not None:
                                try:
                                    plate_res = plate_model(vehicle_crop)
                                    # find first box from plate model
                                    found_plate = False
                                    for pr in plate_res:
                                        for pbox in pr.boxes:
                                            px1, py1, px2, py2 = map(int, pbox.xyxy[0].tolist())
                                            # clip within crop
                                            px1 = max(0, px1); py1 = max(0, py1)
                                            px2 = min(vehicle_crop.shape[1]-1, px2); py2 = min(vehicle_crop.shape[0]-1, py2)
                                            if px2 - px1 > 5 and py2 - py1 > 5:
                                                plate_crop = vehicle_crop[py1:py2, px1:px2]
                                                t = ocr_on_image(plate_crop) if ocr_reader else ""
                                                if t:
                                                    plate_text = t
                                                    found_plate = True
                                                    break
                                        if found_plate:
                                            break
                                except Exception:
                                    # fallback to OCR on whole vehicle crop
                                    plate_text = ocr_on_image(vehicle_crop) if ocr_reader else "UNKNOWN"
                            else:
                                # no plate YOLO model — try EasyOCR directly on vehicle crop
                                plate_text = ocr_on_image(vehicle_crop) if ocr_reader else "UNKNOWN"
                    except Exception:
                        plate_text = "UNKNOWN"

                    plate_text_clean = sanitize_plate(plate_text)

                    # draw on frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label_display = f"{label} {int(speed_kmph)} km/h"
                    cv2.putText(frame, label_display, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    # plate under the box
                    cv2.putText(frame, f"Plate: {plate_text_clean}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    # Save logbook (append row with plate)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    row_df = pd.DataFrame([[timestamp, label, int(speed_kmph), plate_text_clean]],
                                          columns=["timestamp", "vehicle_type", "speed_kmph", "plate_number"])
                    # write header only if file missing or empty
                    header_needed = not os.path.exists(logbook_file) or os.path.getsize(logbook_file) == 0
                    row_df.to_csv(logbook_file, mode="a", header=header_needed, index=False)

                    # Save overspeed capture (include plate)
                    if int(speed_kmph) > speed_limit:
                        safe_plate = plate_text_clean if plate_text_clean else "UNKNOWN"
                        safe_plate = re.sub(r"[^A-Z0-9_]", "_", safe_plate)
                        fname = f"{timestamp.replace(':','-')}_{label}_{int(speed_kmph)}_{safe_plate}.jpg"
                        fpath = os.path.join("overspeed_captures", fname)
                        cv2.imwrite(fpath, frame)
                # end for each box
            # end for each res

            # update tracks, show frame
            st.session_state.vehicle_tracks = new_tracks
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

            # progress for video uploads
            if mode == "Upload Video" and progress_bar:
                st.session_state.frame_num += 1
                progress_bar.progress(min(st.session_state.frame_num / total_frames, 1.0))

    # stop button
    if st.session_state.detecting and st.button("⏹️ Stop Detection"):
        st.session_state.detecting = False
        if st.session_state.cap:
            st.session_state.cap.release()
            st.session_state.cap = None
        st.success("✅ Detection stopped.")

# -------------------------
# Dashboard Tab
# -------------------------
with tab2:
    st.subheader("Traffic Monitoring Dashboard")
    try:
        st.autorefresh(interval=5000, key="refresh_dashboard")
    except Exception:
        st.info("Manual refresh required (Streamlit >=1.26 for auto-refresh).")

    if os.path.exists(logbook_file):
        df = pd.read_csv(logbook_file)
        if not df.empty:
            st.write("### Latest Entries")
            st.dataframe(df.tail(15))

            # vehicle-type filter
            if "vehicle_type" in df.columns:
                vehicle_types = df["vehicle_type"].dropna().unique().tolist()
            else:
                vehicle_types = []
            selected_types = st.multiselect("Filter by Vehicle Type", vehicle_types, default=vehicle_types)
            if len(selected_types) == 0:
                filtered_df = df.copy()
            else:
                filtered_df = df[df["vehicle_type"].isin(selected_types)]

            # overspeed metric
            if "speed_kmph" in filtered_df.columns:
                overspeed_count = filtered_df[filtered_df["speed_kmph"] > speed_limit].shape[0]
            else:
                overspeed_count = 0
            st.metric("⚠️ Overspeed Vehicles", overspeed_count)

            # per-type overspeed chart
            if "speed_kmph" in filtered_df.columns and "vehicle_type" in filtered_df.columns:
                overspeed_df = filtered_df[filtered_df["speed_kmph"] > speed_limit]
                if not overspeed_df.empty:
                    type_counts = overspeed_df["vehicle_type"].value_counts().reset_index()
                    type_counts.columns = ["vehicle_type", "overspeed_count"]
                    chart_overspeed_type = alt.Chart(type_counts).mark_bar().encode(
                        x="vehicle_type",
                        y="overspeed_count",
                        tooltip=["vehicle_type", "overspeed_count"]
                    )
                    st.write("### Overspeed Vehicles by Type")
                    st.altair_chart(chart_overspeed_type, use_container_width=True)

            # vehicle type counts chart
            if "vehicle_type" in filtered_df.columns:
                counts = filtered_df["vehicle_type"].value_counts().reset_index()
                counts.columns = ["vehicle_type", "count"]
                chart_counts = alt.Chart(counts).mark_bar().encode(
                    x="vehicle_type", y="count", tooltip=["vehicle_type", "count"]
                )
                st.altair_chart(chart_counts, use_container_width=True)

            # speed distribution
            if "speed_kmph" in filtered_df.columns:
                chart_speed = alt.Chart(filtered_df).mark_bar().encode(
                    x=alt.X("speed_kmph:Q", bin=True),
                    y="count()",
                    tooltip=["count()"]
                )
                st.write("### Speed Distribution")
                st.altair_chart(chart_speed, use_container_width=True)

            # show plate counts / sample
            if "plate_number" in filtered_df.columns:
                st.write("### Sample Plates (latest 20)")
                st.dataframe(filtered_df[["timestamp", "vehicle_type", "speed_kmph", "plate_number"]].tail(20))

            # download
            st.download_button(
                "📥 Download Filtered Logbook (CSV)",
                filtered_df.to_csv(index=False).encode("utf-8"),
                file_name="traffic_logbook_filtered.csv",
                mime="text/csv"
            )
        else:
            st.warning("⚠️ Logbook is empty. Run detection first.")
    else:
        st.error("⚠️ No logbook found. Run detection first.")

    if st.button("🗑️ Clear All Data"):
        pd.DataFrame(columns=["timestamp", "vehicle_type", "speed_kmph", "plate_number"]).to_csv(logbook_file, index=False)
        for f in os.listdir("overspeed_captures"):
            os.remove(os.path.join("overspeed_captures", f))
        st.session_state.frame_num = 0
        st.success("✅ Logbook reset and overspeed captures cleared!")

# -------------------------
# Overspeed Gallery Tab (Grid view)
# -------------------------
with tab3:
    st.subheader("📷 Overspeed Captures Gallery (Grid View)")
    view_mode = st.radio("View Mode", ["All Captures", "Overspeed Only"])

    image_files = []
    for f in os.listdir("overspeed_captures"):
        if not f.lower().endswith(".jpg"):
            continue
        # if overspeed-only, check speed safely
        if view_mode == "Overspeed Only":
            try:
                m = re.search(r"_(\d+)(?:_[A-Z0-9_]+)?\.jpg$", f, flags=re.IGNORECASE)
                speed = int(m.group(1)) if m else 0
            except Exception:
                speed = 0
            if speed <= speed_limit:
                continue
        image_files.append(f)
    image_files = sorted(image_files)

    if image_files:
        n_cols = 3
        rows = (len(image_files) + n_cols - 1) // n_cols
        for r in range(rows):
            cols = st.columns(n_cols)
            for c in range(n_cols):
                idx = r * n_cols + c
                if idx >= len(image_files):
                    continue
                img_file = image_files[idx]
                img_path = os.path.join("overspeed_captures", img_file)
                cols[c].image(img_path, caption=img_file, use_container_width=True)
                col_del, col_meta = cols[c].columns([1,1])
                with col_del:
                    if st.button(f"🗑️ Delete {img_file}", key=f"del_{img_file}"):
                        try:
                            os.remove(img_path)
                        except Exception:
                            pass
                        st.experimental_rerun()
                with col_meta:
                    # Display plate and speed extracted from filename if present
                    try:
                        m = re.search(r"_(\d+)(?:_([A-Z0-9_]+))?\.jpg$", img_file, flags=re.IGNORECASE)
                        speed = int(m.group(1)) if m and m.group(1) else 0
                        plate = m.group(2) if m and m.group(2) else ""
                    except Exception:
                        speed = 0
                        plate = ""
                    if plate:
                        cols[c].markdown(f"**Plate:** {plate}")
                    cols[c].markdown(f"**Speed:** {speed} km/h")
                    if speed > speed_limit:
                        cols[c].markdown("⚠️ Overspeed")
    else:
        st.info("No captures yet. Run detection first.")
