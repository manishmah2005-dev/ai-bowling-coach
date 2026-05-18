import cv2
import mediapipe as mp
import numpy as np
import time
from progress_tracker import save_session

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

NOSE           = 0
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_HIP       = 23
RIGHT_HIP      = 24
LEFT_KNEE      = 25
RIGHT_KNEE     = 26
LEFT_ANKLE     = 27
RIGHT_ANKLE    = 28
LEFT_ELBOW     = 13
RIGHT_ELBOW    = 14
LEFT_WRIST     = 15
RIGHT_WRIST    = 16

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
              np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return round(angle, 2)

def get_point(landmarks, idx, w, h):
    lm = landmarks[idx]
    return [lm.x * w, lm.y * h]

def detect_weaknesses(landmarks, w, h):
    weaknesses = []
    tips = []

    nose           = get_point(landmarks, NOSE, w, h)
    left_shoulder  = get_point(landmarks, LEFT_SHOULDER, w, h)
    right_shoulder = get_point(landmarks, RIGHT_SHOULDER, w, h)
    left_hip       = get_point(landmarks, LEFT_HIP, w, h)
    right_hip      = get_point(landmarks, RIGHT_HIP, w, h)
    right_knee     = get_point(landmarks, RIGHT_KNEE, w, h)
    left_ankle     = get_point(landmarks, LEFT_ANKLE, w, h)
    right_ankle    = get_point(landmarks, RIGHT_ANKLE, w, h)
    right_elbow    = get_point(landmarks, RIGHT_ELBOW, w, h)
    right_wrist    = get_point(landmarks, RIGHT_WRIST, w, h)

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

    avg_shoulder_x = (left_shoulder[0] + right_shoulder[0]) / 2
    wrist_offset   = abs(right_wrist[0] - avg_shoulder_x)
    if wrist_offset > 120:
        weaknesses.append("Wide back-lift / open bat face")
        tips.append("Bowl: Inswing, off-stump - edge to slip")

    hip_mid_x   = (left_hip[0] + right_hip[0]) / 2
    ankle_mid_x = (left_ankle[0] + right_ankle[0]) / 2
    weight_shift = hip_mid_x - ankle_mid_x
    if weight_shift > 30:
        weaknesses.append("Weight falling back")
        tips.append("Bowl: Full length / yorker - can't drive")

    elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    if elbow_angle < 70:
        weaknesses.append("Elbow dropped (collapsed arm)")
        tips.append("Bowl: Short rising delivery - mistimed pull")

    return weaknesses, tips

def download_model():
    import os, urllib.request
    model_path = "pose_landmarker.task"
    if not os.path.exists(model_path):
        print("Downloading pose model... (one time only, ~30MB)")
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
        urllib.request.urlretrieve(url, model_path)
        print("Model downloaded!")
    return model_path

def run_analysis(session_num=1):
    model_path = download_model()

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE
    )

    cap = cv2.VideoCapture(0)
    print(f"Session {session_num} starting... Press Q to quit and save.")

    # Track weaknesses seen during session
    session_weaknesses = set()
    save_interval = 5   # save to CSV every 5 seconds
    last_save = time.time()

    with PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = result.pose_landmarks[0]

                for lm in landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

                weaknesses, tips = detect_weaknesses(landmarks, w, h)

                # Add to session set
                for w_item in weaknesses:
                    session_weaknesses.add(w_item)

                if not weaknesses:
                    cv2.putText(frame, "Good stance - No weakness detected",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.65, (0, 255, 0), 2)
                else:
                    for i, (weakness, tip) in enumerate(zip(weaknesses, tips)):
                        y = 30 + i * 55
                        cv2.putText(frame, f"Weakness: {weakness}",
                                    (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55, (0, 0, 255), 2)
                        cv2.putText(frame, f"  Tip: {tip}",
                                    (10, y + 22), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.45, (0, 220, 0), 1)

                # Show session number and timer
                elapsed = int(time.time() - last_save)
                cv2.putText(frame, f"Session: {session_num} | Recording...",
                            (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (255, 255, 0), 1)
            else:
                cv2.putText(frame, "No person detected - stand in front of camera",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 165, 255), 2)

            # Auto-save every 5 seconds
            if time.time() - last_save > save_interval and session_weaknesses:
                save_session(list(session_weaknesses), session_num)
                last_save = time.time()

            cv2.imshow("AI Bowling Coach - Pose Analysis", frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                # Final save on quit
                if session_weaknesses:
                    save_session(list(session_weaknesses), session_num)
                    print(f"\nSession {session_num} saved!")
                print("Session ended. Run progress_tracker.py to see your progress!")
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    session = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    run_analysis(session_num=session)