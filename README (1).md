# 🏏 AI Bowling Coach

An AI-powered cricket training system that analyzes a batsman's technique using computer vision and recommends personalized bowling strategies to exploit weaknesses.

## 🎯 What it does

- **Pose Analysis** — Detects 33 body landmarks in real-time using MediaPipe
- **Weakness Detection** — Identifies technical flaws (head position, footwork, backlift, weight transfer, elbow)
- **Smart Recommendations** — Rule engine maps weaknesses to specific bowling strategies (line, length, speed, variation)
- **Progress Tracking** — Logs every session to CSV and visualizes improvement over time
- **Web Dashboard** — Upload any batting video and get instant AI analysis

## 🖥️ Demo

![Dashboard](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.35-green)

## 🧠 Tech Stack

| Component | Technology |
|-----------|-----------|
| Pose Estimation | MediaPipe PoseLandmarker |
| Web Dashboard | Streamlit + Plotly |
| Rule Engine | Association Rule Mining logic |
| Progress Tracking | Pandas + CSV |
| Computer Vision | OpenCV |

## 📁 Project Structure

```
ai-bowling-coach/
├── app.py                  # Streamlit dashboard (main UI)
├── pose_analysis.py        # MediaPipe pose estimation + webcam
├── rule_engine.py          # Weakness → bowling strategy mapper
├── progress_tracker.py     # CSV session logger + summary
├── pose_landmarker.task    # MediaPipe model (auto-downloaded)
├── progress_log.csv        # Session data (auto-generated)
└── requirements.txt        # Dependencies
```

## ⚙️ Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/ai-bowling-coach.git
cd ai-bowling-coach

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### Option 1 — Web Dashboard (recommended)
```bash
streamlit run app.py
```
Open `http://localhost:8501` → Upload batting video → Click Analyze

### Option 2 — Live Webcam
```bash
python pose_analysis.py 1    # 1 = session number
```

### Option 3 — View Progress
```bash
python progress_tracker.py
python rule_engine.py 1      # analyze session 1
```

## 🔍 Weaknesses Detected

| Weakness | Detection Method | Bowling Counter |
|----------|-----------------|-----------------|
| Head falling off-side | Nose vs shoulder midpoint offset | Middle stump inswing |
| Lunging on front foot | Front knee angle < 130° | Short ball / bouncer |
| Wide back-lift | Wrist distance from shoulder | Inswing to off-stump |
| Weight on back foot | Hip vs ankle midpoint shift | Full length / yorker |
| Elbow drop | Elbow joint angle < 70° | Rising short delivery |

## 🎯 Sample Output

```
Weaknesses Detected (3):
  - Weight falling back
  - Lunging on front foot
  - Elbow dropped (collapsed arm)

[COMBO ATTACK]
  Line      : Yorker Line
  Length    : Yorker
  Speed     : Fast (145+ kmph)
  Variation : Fast Yorker
  Why       : Weight back + elbow drop = completely jammed
```

## 🔮 Future Scope

- [ ] YOLOv8 ball tracking integration
- [ ] Hardware bowling machine control via serial/PWM
- [ ] Mobile app (Flutter)
- [ ] Multi-player comparison
- [ ] Patent filing (Provisional Patent Application)

## 👨‍💻 Author

Built as a B.Tech final year project demonstrating applied AI in sports technology.

---
⭐ Star this repo if you found it useful!
