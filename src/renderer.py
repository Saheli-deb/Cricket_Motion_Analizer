
from __future__ import annotations

"""Cinematic‑grade skeleton renderer.

New 💎 upgrades:
  • Joint trails (fade‑out arc of the bat/hand) ⇒ shows swing path.
  • Color‑coded error indicators (green OK, red needs correction).
  • Stylish lower‑third scoreboard with live metrics (frame, elbow‑angle).
  • Adjustable joint radius / line thickness via constants.
"""
import json
import math
from collections import deque
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
JOINT_R = 6
LIMB_THICK = 3
TRAIL_LEN = 15            # frames kept in trail buffer
TRAIL_COLOR = (255, 215, 0)  # gold

# Pose connections (MediaPipe)
_CONNECTIONS = [
    (11,13), (13,15), (12,14), (14,16),
    (11,12), (11,23), (12,24),
    (23,25), (25,27), (24,26), (26,28),
]
LEFT = {11,13,15,23,25,27}
RIGHT = {12,14,16,24,26,28}

COLOR_LEFT = (  0,  80, 255)   # orange / BGR
COLOR_RIGHT = (255,  80,   0)  # cyan‑ish
COLOR_TORSO = ( 60, 255,  60)  # light green

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _color_for_joint(idx: int):
    if idx in LEFT: return COLOR_LEFT
    if idx in RIGHT: return COLOR_RIGHT
    return COLOR_TORSO


def _angle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return abs(ang if ang < 180 else 360-ang)


def _draw_text(img, txt, pos, color):
    (w, h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 0.8, 1)
    x, y = pos
    cv2.rectangle(img, (x-8, y-h-10), (x+w+8, y+8), (0,0,0,0), -1)
    cv2.addWeighted(img, 1.0, img, 0, 0, img)  # keep alpha
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2, cv2.LINE_AA)

# ---------------------------------------------------------------------------
# Main renderer with trail + lower third
# ---------------------------------------------------------------------------

def render_skeleton_video(frame_paths: Sequence[Path], json_paths: Sequence[Path], output_mp4: Path, fps: int = 30):
    output_mp4 = Path(output_mp4).expanduser().resolve()
    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    sample = cv2.imread(str(frame_paths[0]))
    h, w = sample.shape[:2]
    vw = cv2.VideoWriter(str(output_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # ring buffer for trail of bat tip (landmark 16)
    trail = deque(maxlen=TRAIL_LEN)

    for idx, (fpath, jpath) in enumerate(zip(frame_paths, json_paths)):
        frame = cv2.imread(str(fpath))
        lm = json.load(open(jpath))
        pts = {int(i): (int(lm[i]["x"]*w), int(lm[i]["y"]*h)) for i in lm if lm[i]["vis"]>0.3}

        # add bat‑tip point (16) to trail
        if 16 in pts:
            trail.append(pts[16])

        # draw fading trail
        for t, p in enumerate(reversed(trail)):
            alpha = 1.0 - t/len(trail)
            cv2.circle(frame, p, int(JOINT_R*alpha), TRAIL_COLOR, -1, cv2.LINE_AA)

        # joints & limbs
        for pid, (xj,yj) in pts.items():
            cv2.circle(frame, (xj,yj), JOINT_R, _color_for_joint(pid), -1, cv2.LINE_AA)
        for a,b in _CONNECTIONS:
            if a in pts and b in pts:
                cv2.line(frame, pts[a], pts[b], (180,180,180), LIMB_THICK, cv2.LINE_AA)

        # live metric: right‑elbow angle
        if {12,14,16}.issubset(pts):
            ang = _angle(pts[12], pts[14], pts[16])
            color = (0,255,0) if ang<170 else (0,0,255)
            _draw_text(frame, f"Elbow {ang:.0f}°", (30,40), color)

        # lower‑third bar
        cv2.rectangle(frame, (0, h-60), (w, h), (0,0,0), -1)
        _draw_text(frame, f"Frame {idx+1}/{len(frame_paths)}", (20, h-20), (255,255,255))

        vw.write(frame)

    vw.release()
    print(f"[Render] cinematic video saved → {output_mp4}")
