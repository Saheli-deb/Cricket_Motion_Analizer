from __future__ import annotations

"""Streamlit dashboard – professional UI for Cricket Motion Analyzer.

Features
========
• Upload any .mp4 clip (≤ 500 MB by default).
• Choose sampling FPS.
• One‑click **Analyze!** button runs the full pipeline
  (frame extraction → pose estimation → biomechanics → cinematic render).
• Live status + progress bar.
• Embedded video player for the annotated MP4.
• Download links for CSV metrics & MP4.
• Interactive 3‑D Plotly pose of first frame.
"""

import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st

from src.main import run_pipeline
from src.visualizer import plot_pose_3d

st.set_page_config(page_title="Cricket Motion Analyzer", page_icon="🏏", layout="wide")

# ───────────────────────────── Sidebar ─────────────────────────────
with st.sidebar:
    st.header("📂 Upload Video")
    video_file = st.file_uploader("MP4 clip", type=["mp4", "mov", "mkv"], accept_multiple_files=False)
    fps = st.slider("Sampling FPS", min_value=2, max_value=15, value=5)
    run_btn = st.button("Analyze! 🏃‍♂️")
    st.markdown("---")
    st.markdown("**Made with ❤️ using MediaPipe & Streamlit**")

st.title("🏏 Cricket Motion Analyzer")

placeholder = st.empty()

if run_btn and video_file:
    with st.spinner("⏳ Processing… this may take a moment depending on video length …"):
        # Save upload to a temp location
        with tempfile.TemporaryDirectory() as tmpdir:
            vid_path = Path(tmpdir) / video_file.name
            vid_path.write_bytes(video_file.read())

            # Run pipeline (no UI) – results saved under project data/
            run_pipeline(vid_path, fps=fps)

            # Derive output paths
            root = Path(__file__).resolve().parents[1]
            vid_name = vid_path.stem
            csv_path = root / "data" / "analysis" / f"{vid_name}_features.csv"
            mp4_path = root / "data" / "analysis" / f"{vid_name}_visualized.mp4"
            kp_dir = root / "data" / "keypoints" / vid_name
            first_json: Optional[Path] = None
            if kp_dir.exists():
                try:
                    first_json = sorted(kp_dir.glob("*.json"))[0]
                except IndexError:
                    first_json = None

        st.success("✅ Analysis complete!")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("🎦 Annotated Video")
            st.video(str(mp4_path))
            st.download_button("Download MP4", mp4_path.read_bytes(), file_name=mp4_path.name)

        with col2:
            st.subheader("📊 Metrics CSV")
            st.download_button("Download CSV", csv_path.read_bytes(), file_name=csv_path.name)
            if first_json:
                st.subheader("🔍 3‑D Pose (first frame)")
                fig = plot_pose_3d(first_json, title="First Frame 3‑D")
                st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Upload a cricket clip in the sidebar and hit **Analyze!** to begin.")
