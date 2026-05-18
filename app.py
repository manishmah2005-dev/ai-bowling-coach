# ================================================================
# app.py — AI Bowling Coach — Streamlit Dashboard
# ================================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os
import time
from rule_engine import get_recommendations, display_recommendations
from progress_tracker import save_session, init_csv

# ----------------------------------------------------------------
# Page config
# ----------------------------------------------------------------
st.set_page_config(
    page_title="AI Bowling Coach",
    page_icon="🏏",
    layout="wide"
)

# ----------------------------------------------------------------
# MediaPipe setup
# ----------------------------------------------------------------
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

NOSE=0; LEFT_SHOULDER=11; RIGHT_SHOULDER=12
LEFT_HIP=23; RIGHT_HIP=24; LEFT_KNEE=25; RIGHT_KNEE=26
LEFT_ANKLE=27; RIGHT_ANKLE=28
LEFT_ELBOW=13; RIGHT_ELBOW=14; LEFT_WRIST=15; RIGHT_WRIST=16

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return round(360 - angle if angle > 180 else angle, 2)

def get_point(landmarks, idx, w, h):
    lm = landmarks[idx]
    return [lm.x * w, lm.y * h]

def detect_weaknesses(landmarks, w, h):
    weaknesses, tips = [], []
    nose          = get_point(landmarks, NOSE, w, h)
    left_shoulder = get_point(landmarks, LEFT_SHOULDER, w, h)
    right_shoulder= get_point(landmarks, RIGHT_SHOULDER, w, h)
    left_hip      = get_point(landmarks, LEFT_HIP, w, h)
    right_hip     = get_point(landmarks, RIGHT_HIP, w, h)
    right_knee    = get_point(landmarks, RIGHT_KNEE, w, h)
    left_ankle    = get_point(landmarks, LEFT_ANKLE, w, h)
    right_ankle   = get_point(landmarks, RIGHT_ANKLE, w, h)
    right_elbow   = get_point(landmarks, RIGHT_ELBOW, w, h)
    right_wrist   = get_point(landmarks, RIGHT_WRIST, w, h)

    shoulder_mid_x = (left_shoulder[0] + right_shoulder[0]) / 2
    head_offset = nose[0] - shoulder_mid_x
    if abs(head_offset) > 40:
        side = "off-side" if head_offset > 0 else "leg-side"
        weaknesses.append(f"Head falling to {side}")
        tips.append("Bowl: Middle stump line - LBW chance")

    front_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
    if front_knee_angle < 130:
        weaknesses.append("Lunging on front foot")
        tips.append("Bowl: Short length, fast - loses balance")

    wrist_offset = abs(right_wrist[0] - shoulder_mid_x)
    if wrist_offset > 120:
        weaknesses.append("Wide back-lift / open bat face")
        tips.append("Bowl: Inswing, off-stump - edge to slip")

    hip_mid_x = (left_hip[0] + right_hip[0]) / 2
    ankle_mid_x = (left_ankle[0] + right_ankle[0]) / 2
    if hip_mid_x - ankle_mid_x > 30:
        weaknesses.append("Weight falling back")
        tips.append("Bowl: Full length / yorker - can't drive")

    elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    if elbow_angle < 70:
        weaknesses.append("Elbow dropped (collapsed arm)")
        tips.append("Bowl: Short rising delivery - mistimed pull")

    return weaknesses, tips

def download_model():
    import urllib.request
    model_path = "pose_landmarker.task"
    if not os.path.exists(model_path):
        with st.spinner("Downloading pose model... (one time, ~30MB)"):
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
            urllib.request.urlretrieve(url, model_path)
    return model_path

# ----------------------------------------------------------------
# Analyze video file
# ----------------------------------------------------------------
def analyze_video(video_path, session_num):
    model_path = download_model()
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE
    )

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    all_weaknesses = set()
    annotated_frames = []
    frame_count = 0
    sample_every = max(1, int(fps // 4))  # sample 4 frames per second

    progress_bar = st.progress(0, text="Analyzing video...")

    with PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % sample_every != 0:
                continue

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = result.pose_landmarks[0]

                for lm in landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

                weaknesses, tips = detect_weaknesses(landmarks, w, h)
                for wk in weaknesses:
                    all_weaknesses.add(wk)

                for i, (wk, tip) in enumerate(zip(weaknesses, tips)):
                    y = 30 + i * 50
                    cv2.putText(frame, f"W: {wk}", (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.putText(frame, f"T: {tip}", (10, y+20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 0), 1)

            annotated_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            progress_bar.progress(
                min(frame_count / total_frames, 1.0),
                text=f"Analyzing... frame {frame_count}/{total_frames}"
            )

    cap.release()
    progress_bar.empty()

    weakness_list = list(all_weaknesses)
    if weakness_list:
        save_session(weakness_list, session_num)

    return weakness_list, annotated_frames

# ================================================================
# STREAMLIT UI
# ================================================================

st.title("🏏 AI Bowling Coach")
st.markdown("**Upload a batting video → Get weakness analysis + bowling strategy**")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Analyze Video", "Progress Report"])
session_num = st.sidebar.number_input("Session Number", min_value=1, value=1, step=1)

# ----------------------------------------------------------------
# PAGE 1: Analyze Video
# ----------------------------------------------------------------
if page == "Analyze Video":
    st.header("Video Analysis")

    uploaded_file = st.file_uploader(
        "Upload batting video (MP4, AVI, MOV)",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_file:
        # Save to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()

        st.video(uploaded_file)

        if st.button("Analyze", type="primary"):
            weaknesses, frames = analyze_video(tfile.name, session_num)
            os.unlink(tfile.name)

            # Show results
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Detected Weaknesses")
                if not weaknesses:
                    st.success("No weaknesses detected! Good technique.")
                else:
                    for w in weaknesses:
                        st.error(f"❌ {w}")

            with col2:
                st.subheader("Bowling Strategies")
                recs = get_recommendations(weaknesses)
                if recs:
                    for rec in recs[:3]:
                        tag = "COMBO ATTACK" if rec["priority"] == 0 else f"Strategy"
                        with st.expander(f"🎯 {tag} — {rec['variation']}"):
                            st.write(f"**Line:** {rec['line']}")
                            st.write(f"**Length:** {rec['length']}")
                            st.write(f"**Speed:** {rec['speed']}")
                            st.write(f"**Why it works:** {rec['reason']}")
                            st.write(f"**Match:** {int(rec['match_score']*100)}%")

            # Show annotated frames
            if frames:
                st.subheader("Annotated Frames")
                cols = st.columns(3)
                sample_indices = np.linspace(0, len(frames)-1, min(6, len(frames)), dtype=int)
                for i, idx in enumerate(sample_indices):
                    with cols[i % 3]:
                        st.image(frames[idx], caption=f"Frame {idx}", use_container_width=True)

            st.success(f"Session {session_num} saved to progress_log.csv!")

# ----------------------------------------------------------------
# PAGE 2: Progress Report
# ----------------------------------------------------------------
elif page == "Progress Report":
    st.header("Progress Report")

    CSV_FILE = "progress_log.csv"
    if not os.path.exists(CSV_FILE):
        st.warning("No data yet! Run a session first.")
    else:
        df = pd.read_csv(CSV_FILE)

        st.subheader("Session History")
        st.dataframe(df, use_container_width=True)

        # Weakness frequency chart
        st.subheader("Most Common Weaknesses")
        weakness_cols = ["Head_Falling","Lunging_Front_Foot","Wide_Backlift","Weight_Back","Elbow_Drop"]
        weakness_labels = ["Head Falling","Lunging","Wide Backlift","Weight Back","Elbow Drop"]
        counts = [df[col].sum() for col in weakness_cols]

        fig = px.bar(
            x=weakness_labels, y=counts,
            labels={"x": "Weakness", "y": "Times Detected"},
            color=counts,
            color_continuous_scale="Reds",
            title="Weakness Frequency Across All Sessions"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Progress over sessions
        st.subheader("Total Weaknesses Per Session")
        session_df = df.groupby("Session")["Total_Weaknesses"].mean().reset_index()
        fig2 = px.line(
            session_df, x="Session", y="Total_Weaknesses",
            markers=True,
            title="Average Weaknesses Per Session (Lower = Better)"
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Summary metrics
        st.subheader("Quick Stats")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Sessions", df["Session"].nunique())
        c2.metric("Avg Weaknesses", round(df["Total_Weaknesses"].mean(), 1))
        worst = weakness_labels[counts.index(max(counts))] if max(counts) > 0 else "None"
        c3.metric("Most Common Weakness", worst)