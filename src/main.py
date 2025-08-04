
from __future__ import annotations

"""
Cricket Motion Analyzer – main entry point.

Run from project root, for example:
    python -m src.main --video data/raw_videos/Video-1.mp4 --fps 5
"""

import argparse
import logging
from pathlib import Path

from src.video_extractor import extract_frames
from src.pose_estimator import extract_pose_from_frames
from src.feature_extractor import extract_biomechanics
from src.renderer import render_skeleton_video
from src.visualizer import plot_pose_3d

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s — %(message)s",
    datefmt="%H:%M:%S",
)


def run_pipeline(video_path: Path, fps: int = 5) -> None:
    """Run full analysis on a single cricket video."""
    root = Path(__file__).resolve().parents[1]
    vid_name = video_path.stem

    # ── output folders & filenames ──────────────────────────────────────────────
    frames_dir = root / "data" / "extracted_frames" / vid_name
    kp_dir = root / "data" / "keypoints" / vid_name
    csv_out = root / "data" / "analysis" / f"{vid_name}_features.csv"
    mp4_out = root / "data" / "analysis" / f"{vid_name}_visualized.mp4"

    # 1️⃣  Frame extraction
    frames = extract_frames(video_path, frames_dir, fps=fps, overwrite=True)
    if not frames:
        logging.error("No frames extracted — aborting.")
        return

    # 2️⃣  Pose estimation (MediaPipe)
    jsons = extract_pose_from_frames(frames, kp_dir)
    if not jsons:
        logging.error("No pose landmarks detected — aborting.")
        return

    # 3️⃣  Biomechanical feature extraction
    extract_biomechanics(jsons, csv_out)

    # 4️⃣  Render cinematic overlay video
    render_skeleton_video(frames, jsons, mp4_out, fps=fps)
    logging.info("✅ Visualization saved → %s", mp4_out.relative_to(root))

    # 5️⃣  Optional sanity-check plot (first frame, interactive)
    plot_pose_3d(jsons[0], title=f"{vid_name} — first pose")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cricket Motion Analysis Pipeline")
    parser.add_argument(
        "--video", required=True, help="Path to cricket video file (.mp4, .mov, etc.)"
    )
    parser.add_argument(
        "--fps", type=int, default=5, help="Target frames-per-second for sampling"
    )
    args = parser.parse_args()

    run_pipeline(Path(args.video).expanduser().resolve(), fps=args.fps)


if __name__ == "__main__":
    main()
