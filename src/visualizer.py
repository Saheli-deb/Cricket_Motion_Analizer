
"""3‑D Pose visualizer using Plotly.

Functions:
  • `plot_pose_3d(json_path)` → interactive Plotly figure
  • `compare_poses(actual_json, ideal_json)` → overlay comparison figure

Assumes MediaPipe landmark indexing (0‑32). Z is used directly; Y is inverted
for natural viewer orientation (upwards).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import plotly.graph_objects as go

# MediaPipe Pose connection pairs (subset for clarity)
_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,7),  # Nose‑>Eyes‑>Ears
    (0,4), (4,5), (5,6), (6,8),
    (9,10),  # Shoulders
    (11,13), (13,15),  # Left arm
    (12,14), (14,16),  # Right arm
    (11,12), (11,23), (12,24),  # Torso
    (23,25), (25,27),  # Left leg
    (24,26), (26,28)   # Right leg
]

def _load_landmarks(json_path: str | Path):
    with open(json_path) as jf:
        data = json.load(jf)
    xs, ys, zs = [], [], []
    for i in range(33):
        pt = data.get(str(i))
        if pt:
            xs.append(pt["x"])
            ys.append(-pt["y"])  # flip Y for visual
            zs.append(pt["z"])
        else:
            xs.append(None)
            ys.append(None)
            zs.append(None)
    return xs, ys, zs

def _skeleton_trace(xs, ys, zs, name, color):
    lines_x, lines_y, lines_z = [], [], []
    for a, b in _CONNECTIONS:
        if xs[a] is None or xs[b] is None:
            continue
        lines_x += [xs[a], xs[b], None]
        lines_y += [ys[a], ys[b], None]
        lines_z += [zs[a], zs[b], None]
    return go.Scatter3d(x=lines_x, y=lines_y, z=lines_z, mode="lines", line=dict(width=4, color=color), name=name)

def plot_pose_3d(json_path: str | Path, title: Optional[str] = None):
    xs, ys, zs = _load_landmarks(json_path)
    fig = go.Figure()
    fig.add_trace(_skeleton_trace(xs, ys, zs, "Actual", "blue"))
    fig.update_layout(title=title or Path(json_path).stem,
                      scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
                      margin=dict(l=0, r=0, b=0, t=30))
    fig.show()


def compare_poses(actual_json: str | Path, ideal_json: str | Path):
    xs_a, ys_a, zs_a = _load_landmarks(actual_json)
    xs_i, ys_i, zs_i = _load_landmarks(ideal_json)
    fig = go.Figure()
    fig.add_trace(_skeleton_trace(xs_a, ys_a, zs_a, "Actual", "blue"))
    fig.add_trace(_skeleton_trace(xs_i, ys_i, zs_i, "Ideal", "green"))
    fig.update_layout(title="Actual vs Ideal Pose",
                      scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
                      margin=dict(l=0, r=0, b=0, t=30))
    fig.show()
