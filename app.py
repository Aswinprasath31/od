import streamlit as st
import pandas as pd
import os
import altair as alt

CSV_FILE = "traffic_logbook.csv"
CAPTURE_FOLDER = "overspeed_captures"

st.set_page_config(page_title="Traffic Monitoring Dashboard", layout="wide")
st.title("🚦 Traffic Monitoring Dashboard")

# Auto-refresh every 5s
st.experimental_autorefresh(interval=5000, limit=None)

# Load CSV
if os.path.exists(CSV_FILE):
    df = pd.read_csv(CSV_FILE)

    # Convert timestamp if exists
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    # Sidebar filters
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

    # Apply filters
    filtered_df = df[df["Vehicle_Type"].isin(selected_types)]
    if overspeed_filter != "All":
        filtered_df = filtered_df[filtered_df["Overspeed"] == overspeed_filter]
    filtered_df = filtered_df[(filtered_df["Speed_km/h"] >= speed_min) & (filtered_df["Speed_km/h"] <= speed_max)]

    # Summary metrics
    st.subheader("📊 Traffic Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Vehicles Logged", len(df))
    with col2:
        st.metric("Overspeeding Vehicles", (df["Overspeed"] == "YES").sum())
    with col3:
        avg_speed = round(df["Speed_km/h"].mean(), 2) if not df.empty else 0
        st.metric("Average Speed (km/h)", avg_speed)

    # Charts
    st.subheader("📈 Real-Time Charts")
    if not filtered_df.empty:
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            type_chart = alt.Chart(filtered_df).mark_bar().encode(
                x=alt.X("Vehicle_Type:N", title="Vehicle Type"),
                y=alt.Y("count()", title="Count"),
                color="Vehicle_Type:N"
            ).properties(title="Vehicle Count by Type")
            st.altair_chart(type_chart, use_container_width=True)

        with chart_col2:
            speed_chart = alt.Chart(filtered_df).mark_bar().encode(
                x=alt.X("Speed_km/h:Q", bin=alt.Bin(maxbins=20)),
                y="count()"
            ).properties(title="Speed Distribution")
            st.altair_chart(speed_chart, use_container_width=True)

        # NEW: Speed over time chart with overspeed highlighting
        st.subheader("⏱️ Real-Time Speed Tracking")

        if "Timestamp" in filtered_df.columns:
            live_df = filtered_df.tail(200).copy()  # last 200 entries

            # Vehicle ID selector
            available_ids = live_df["Vehicle_ID"].dropna().unique().tolist()
            selected_ids = st.multiselect("Select Vehicle IDs", available_ids, default=available_ids)

            # Apply ID filter
            if selected_ids:
                live_df = live_df[live_df["Vehicle_ID"].isin(selected_ids)]

            # Assign colors based on overspeed
            live_df["Speed_Status"] = live_df["Overspeed"].apply(lambda x: "Overspeed" if x == "YES" else "Normal")

            if not live_df.empty:
                line_chart = alt.Chart(live_df).mark_line(point=True).encode(
                    x=alt.X("Timestamp:T", title="Time"),
                    y=alt.Y("Speed_km/h:Q", title="Speed (km/h)"),
                    color=alt.Color("Speed_Status:N",
                                    scale=alt.Scale(domain=["Normal", "Overspeed"],
                                                    range=["green", "red"]),
                                    legend=alt.Legend(title="Status")),
                    detail="Vehicle_ID:N",
                    tooltip=["Vehicle_ID", "Vehicle_Type", "Speed_km/h", "Overspeed", "Timestamp"]
                ).properties(title="Speed Tracking per Vehicle (Last 200 Logs)")

                st.altair_chart(line_chart, use_container_width=True)
            else:
                st.info("⚠️ No data available for selected vehicle IDs.")
        else:
            st.info("⚠️ No timestamp data available for speed-over-time chart.")

    # Logbook
    st.subheader("📑 Vehicle Logbook (last 30 entries)")
    st.dataframe(filtered_df.tail(30), use_container_width=True)

    # Overspeed gallery
    st.subheader("🚨 Overspeed Evidence Gallery")
    overspeed_df = filtered_df[filtered_df["Overspeed"] == "YES"].dropna(subset=["Image_Path"])
    if not overspeed_df.empty:
        cols = st.columns(3)
        for i, row in overspeed_df.tail(9).iterrows():
            with cols[i % 3]:
                if os.path.exists(row["Image_Path"]):
                    st.image(row["Image_Path"], caption=f"ID {row['Vehicle_ID']} | {row['Speed_km/h']} km/h", use_container_width=True)
                else:
                    st.warning(f"Image missing for ID {row['Vehicle_ID']}")
    else:
        st.info("No overspeeding vehicles detected yet 🚘💨")

else:
    st.error("⚠️ No traffic logbook found. Run detection script first.")
