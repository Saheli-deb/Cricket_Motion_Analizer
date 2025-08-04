
"""Pose estimation via MediaPipe Pose.

Runs on a list of frame image paths and stores `frame_xxxxx.json` with landmark
(x, y, z, visibility) for 33 body points.
Returns a manifest of saved JSON paths.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Sequence

import cv2
import mediapipe as mp

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s — %(message)s", datefmt="%H:%M:%S"))
    LOGGER.addHandler(_h)
    LOGGER.setLevel(logging.INFO)

mp_pose = mp.solutions.pose

def extract_pose_from_frames(frame_paths: Sequence[str | Path], output_dir: str | Path, log_every: int = 25) -> List[Path]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    saved: List[Path] = []
    with mp_pose.Pose(static_image_mode=True) as pose:
        for i, fpath in enumerate(frame_paths):
            img = cv2.imread(str(fpath))
            if img is None:
                LOGGER.warning("[Pose] unreadable %s", fpath)
                continue
            res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not res.pose_landmarks:
                continue
            lm_json = {idx: {"x": lm.x, "y": lm.y, "z": lm.z, "vis": lm.visibility} for idx, lm in enumerate(res.pose_landmarks.landmark)}
            jpath = output_dir / f"{Path(fpath).stem}.json"
            with open(jpath, "w") as jf:
                json.dump(lm_json, jf)
            saved.append(jpath)
            if i % log_every == 0:
                LOGGER.info("[Pose] %d/%d", i + 1, len(frame_paths))
    LOGGER.info("[Pose] saved %d JSONs", len(saved))
    return saved
