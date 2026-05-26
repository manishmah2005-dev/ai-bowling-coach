import cv2
import numpy as np
from ultralytics import YOLO

# Model load karo
model = YOLO('models/best.pt')

def track_ball(frame):
    """
    Ek frame mein ball detect karo
    Returns: ball_info dict ya None
    """
    results = model(frame, verbose=False)[0]
    
    ball_info = None
    
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf > 0.3:  # 30% confidence threshold
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2  # center x
            cy = (y1 + y2) // 2  # center y
            
            ball_info = {
                'bbox': (x1, y1, x2, y2),
                'center': (cx, cy),
                'confidence': round(conf, 2),
                'line': classify_line(cx, frame.shape[1]),
                'length': classify_length(cy, frame.shape[0])
            }
            
            # Frame pe draw karo
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, f'Ball {conf:.2f}', 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 0), 2)
    
    return ball_info, frame


def classify_line(cx, frame_width):
    """Ball ki line classify karo"""
    third = frame_width // 3
    if cx < third:
        return "Leg stump"
    elif cx < 2 * third:
        return "Middle stump"
    else:
        return "Off stump"


def classify_length(cy, frame_height):
    """Ball ki length classify karo"""
    if cy < frame_height * 0.3:
        return "Short"
    elif cy < frame_height * 0.6:
        return "Good length"
    else:
        return "Full/Yorker"


def analyze_ball_trajectory(ball_positions):
    """
    Multiple frames ki ball positions se trajectory analyze karo
    Returns: delivery type
    """
    if len(ball_positions) < 3:
        return "Unknown"
    
    # Y coordinates check karo — ball upar se neeche aati hai
    y_coords = [pos[1] for pos in ball_positions]
    x_coords = [pos[0] for pos in ball_positions]
    
    # Swing detect karo — x movement
    x_movement = x_coords[-1] - x_coords[0]
    
    if abs(x_movement) > 50:
        if x_movement > 0:
            return "Outswing"
        else:
            return "Inswing"
    else:
        return "Straight"