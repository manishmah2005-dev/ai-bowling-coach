import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarkerOptions
import csv
import os
from datetime import datetime

# Landmark indices
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def get_lm(landmarks, idx, w, h):
    lm = landmarks[idx]
    return [lm.x * w, lm.y * h]


def calculate_scores(landmarks, w, h):
    scores = {}
    details = {}

    # 1. BALANCE SCORE
    try:
        nose = get_lm(landmarks, NOSE, w, h)
        left_hip = get_lm(landmarks, LEFT_HIP, w, h)
        right_hip = get_lm(landmarks, RIGHT_HIP, w, h)
        hip_center_x = (left_hip[0] + right_hip[0]) / 2
        head_offset = abs(nose[0] - hip_center_x)
        balance_score = max(0, 100 - (head_offset / w * 200))
        scores['balance'] = round(balance_score)
        if head_offset > w * 0.08:
            side = "off-side" if nose[0] > hip_center_x else "leg-side"
            details['balance'] = f"Head falling to {side}"
        else:
            details['balance'] = "Head position good"
    except:
        scores['balance'] = 50
        details['balance'] = "Could not detect"

    # 2. FOOTWORK SCORE
    try:
        left_hip = get_lm(landmarks, LEFT_HIP, w, h)
        left_knee = get_lm(landmarks, LEFT_KNEE, w, h)
        left_ankle = get_lm(landmarks, LEFT_ANKLE, w, h)
        knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        if 130 <= knee_angle <= 160:
            footwork_score = 100
            details['footwork'] = "Footwork excellent"
        elif knee_angle < 130:
            footwork_score = max(0, int((knee_angle / 130) * 80))
            details['footwork'] = "Lunging on front foot"
        else:
            footwork_score = max(0, int(100 - ((knee_angle - 160) / 20) * 40))
            details['footwork'] = "Weight on back foot"
        scores['footwork'] = round(footwork_score)
    except:
        scores['footwork'] = 50
        details['footwork'] = "Could not detect"

    # 3. TIMING SCORE
    try:
        left_shoulder = get_lm(landmarks, LEFT_SHOULDER, w, h)
        left_elbow = get_lm(landmarks, LEFT_ELBOW, w, h)
        left_wrist = get_lm(landmarks, LEFT_WRIST, w, h)
        elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        if 90 <= elbow_angle <= 140:
            timing_score = 100
            details['timing'] = "Elbow position good"
        elif elbow_angle < 90:
            timing_score = max(0, int((elbow_angle / 90) * 70))
            details['timing'] = "Elbow too bent - early trigger"
        else:
            timing_score = max(0, int(100 - ((elbow_angle - 140) / 40) * 50))
            details['timing'] = "Elbow dropped - late swing"
        scores['timing'] = round(timing_score)
    except:
        scores['timing'] = 50
        details['timing'] = "Could not detect"

    # 4. BAT SWING SCORE
    try:
        left_shoulder = get_lm(landmarks, LEFT_SHOULDER, w, h)
        left_wrist = get_lm(landmarks, LEFT_WRIST, w, h)
        wrist_height = left_shoulder[1] - left_wrist[1]
        if wrist_height > 50:
            swing_score = 100
            details['bat_swing'] = "Back-lift good"
        elif wrist_height > 0:
            swing_score = int((wrist_height / 50) * 100)
            details['bat_swing'] = "Back-lift low"
        else:
            swing_score = max(0, int(100 + wrist_height))
            details['bat_swing'] = "Wide back-lift / open bat face"
        scores['bat_swing'] = round(swing_score)
    except:
        scores['bat_swing'] = 50
        details['bat_swing'] = "Could not detect"

    # 5. OVERALL
    scores['overall'] = int(
        scores['balance'] * 0.30 +
        scores['footwork'] * 0.30 +
        scores['timing'] * 0.25 +
        scores['bat_swing'] * 0.15
    )

    return scores, details


def get_weaknesses_from_scores(scores, details):
    weaknesses = []
    if scores.get('balance', 100) < 65:
        weaknesses.append(details.get('balance', 'Poor balance'))
    if scores.get('footwork', 100) < 65:
        weaknesses.append(details.get('footwork', 'Poor footwork'))
    if scores.get('timing', 100) < 65:
        weaknesses.append(details.get('timing', 'Poor timing'))
    if scores.get('bat_swing', 100) < 65:
        weaknesses.append(details.get('bat_swing', 'Poor bat swing'))
    return weaknesses


def analyze_video(video_path, session_num=1):
    model_path = 'pose_landmarker.task'
    if not os.path.exists(model_path):
        print("Downloading pose model...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
            model_path
        )

    base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
    options = PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], [], {}, []

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    all_scores = []
    annotated_frames = []
    last_details = {}
    frame_count = 0

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 10 != 0:
                continue

            timestamp_ms = int((frame_count / fps) * 1000)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = result.pose_landmarks[0]

                # Draw skeleton
                for lm in landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

                scores, details = calculate_scores(landmarks, w, h)
                all_scores.append(scores)
                last_details = details

                y = 30
                for key, val in scores.items():
                    if key != 'overall':
                        color = (0, 255, 0) if val >= 65 else (0, 0, 255)
                        cv2.putText(frame, f"{key.title()}: {val}/100",
                                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        y += 25

                cv2.putText(frame, f"Overall: {scores.get('overall', 0)}/100",
                            (10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                if len(annotated_frames) < 8:
                    annotated_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

    if not all_scores:
        return [], annotated_frames, {}, []

    avg_scores = {}
    for key in all_scores[0].keys():
        avg_scores[key] = round(sum(s.get(key, 0) for s in all_scores) / len(all_scores))

    weaknesses = get_weaknesses_from_scores(avg_scores, last_details)
    save_to_csv(session_num, weaknesses, avg_scores)

    return weaknesses, annotated_frames, avg_scores, []


def save_to_csv(session_num, weaknesses, scores):
    file_exists = os.path.exists('progress_log.csv')
    with open('progress_log.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                'Session', 'Date', 'Weaknesses',
                'Balance', 'Footwork', 'Timing', 'BatSwing', 'Overall'
            ])
        writer.writerow([
            session_num,
            datetime.now().strftime('%Y-%m-%d %H:%M'),
            ', '.join(weaknesses) if weaknesses else 'None detected',
            scores.get('balance', 0),
            scores.get('footwork', 0),
            scores.get('timing', 0),
            scores.get('bat_swing', 0),
            scores.get('overall', 0)
        ])