# 🏏 Cricket Motion Analyzer

Computer-vision pipeline for **batting, bowling, and fielding biomechanics**.  
Takes any cricket video, performs 3-D pose estimation, and produces a
fully-annotated MP4 with actionable coaching overlays plus a CSV of
frame-level metrics.

---

## Key Features

| Module | Output |
|--------|--------|
| **Frame Extractor** (`video_extractor.py`) | Samples video at user-defined FPS |
| **Pose Estimator** (`pose_estimator.py`)  | MediaPipe Pose – 33 landmarks / frame |
| **Biomechanics Engine** (`feature_extractor.py`) | Elbow, knee, trunk, bat-tilt, hip-shoulder X-factor, shot-type heuristic |
| **Cinematic Renderer** (`renderer.py`) | Skeleton overlay, swing trail, colour-coded metric badges, velocity bar, slow-mo burst, auto-zoom, optional ghost template |
| **3-D Viewer** (`visualizer.py`) | Interactive Plotly skeleton for spot-checking any frame |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Pose detection | **MediaPipe** 0.9.x |
| Computer Vision | **OpenCV** 4.8 |
| Data | **NumPy / Pandas** |
| Visualization | OpenCV overlay, **Plotly 3-D** |
| Runtime | Python ≥ 3.9 (pure CPU; GPU optional) |

---

