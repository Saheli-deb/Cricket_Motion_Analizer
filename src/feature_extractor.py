
"""Biomechanical feature extraction.

Given landmark JSON files, compute joint angles (e.g., elbow, knee), shoulder‑hip
separation, and simple velocity metrics. Saves a CSV summarising features per
frame and returns the DataFrame for further analysis.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import List, Sequence

import pandas as pd

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s — %(message)s", datefmt="%H:%M:%S"))
    LOGGER.addHandler(_h)
    LOGGER.setLevel(logging.INFO)

# --------------------- utility maths ---------------------

def _angle(a, b, c):
    """Return angle ABC in degrees using 2‑D projection (x,y)."""
    ang = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    )
    return abs(ang if ang < 180 else 360 - ang)

# ---------------------------------------------------------

def extract_biomechanics(json_paths: Sequence[str | Path], csv_out: str | Path) -> pd.DataFrame:
    rows: List[dict] = []

    for jpath in json_paths:
        with open(jpath) as jf:
            lm = json.load(jf)

        def pt(i):
            return (lm[str(i)]["x"], lm[str(i)]["y"], lm[str(i)]["z"])

        try:
            # Right elbow = shoulder(12), elbow(14), wrist(16)
            r_elbow = _angle(pt(12), pt(14), pt(16))
            # Right knee = hip(24), knee(26), ankle(28)
            r_knee = _angle(pt(24), pt(26), pt(28))
            # Trunk separation angle (shoulder‑hip line vs. horizontal)
            shoulder = pt(12)
            hip = pt(24)
            trunk = math.degrees(math.atan2(shoulder[1] - hip[1], shoulder[0] - hip[0]))
        except KeyError:
            LOGGER.debug("Missing keypoints in %s", jpath.name)
            continue

        rows.append({
            "frame": Path(jpath).stem,
            "r_elbow_deg": r_elbow,
            "r_knee_deg": r_knee,
            "trunk_deg": trunk,
        })

    df = pd.DataFrame(rows)
    csv_out = Path(csv_out).expanduser().resolve()
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_out, index=False)
    LOGGER.info("[Feature] saved %d rows → %s", len(df), csv_out.name)
    return df

