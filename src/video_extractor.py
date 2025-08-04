### video_extractor.py
"""Video frame extraction utility.

Extracts frames from an input video at a user‑defined sampling rate (fps) and
stores them under an output directory. Designed for production robustness:
  • Validates paths & FPS.
  • Uses logging for traceability.
  • Returns a manifest (list of saved frame paths).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import cv2

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s — %(message)s", datefmt="%H:%M:%S"))
    LOGGER.addHandler(_h)
    LOGGER.setLevel(logging.INFO)

def extract_frames(video_path: str | Path, output_dir: str | Path, fps: int = 5, overwrite: bool = False) -> List[Path]:
    video_path = Path(video_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(video_path)

    if overwrite:
        for p in output_dir.glob("*.jpg"):
            p.unlink(missing_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if native_fps == 0:
        raise RuntimeError("Source FPS reported as 0")

    stride = int(max(native_fps // fps, 1))
    LOGGER.info("Extracting frames %s → %s | stride=%d", video_path.name, output_dir.name, stride)

    saved: List[Path] = []
    idx = saved_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            fpath = output_dir / f"frame_{saved_idx:05d}.jpg"
            cv2.imwrite(str(fpath), frame)
            saved.append(fpath)
            saved_idx += 1
        idx += 1
    cap.release()
    LOGGER.info("Saved %d frames", len(saved))
    return saved

