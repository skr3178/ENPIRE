"""Episode recording: save rollout videos and trajectories for audit.

Mirrors the ENPIRE task requirement: "For each policy evaluation, save a video with
a unique name and a `_success` or `_failure` suffix." Videos are written with imageio
(ffmpeg-free via the imageio-ffmpeg fallback, or pillow for GIF). Trajectories are
saved as compressed .npz so a run can be replayed / inspected later.
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np


def video_name(prefix: str, seed: int, success: bool) -> str:
    return f"{prefix}_seed{seed:03d}_{'success' if success else 'failure'}.mp4"


def save_video(frames: List[np.ndarray], path: str, fps: int = 10) -> Optional[str]:
    """Write RGB frames to an mp4. Returns the path, or None if no frames / no writer."""
    if not frames:
        return None
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    import imageio.v2 as imageio

    try:
        with imageio.get_writer(path, fps=fps, macro_block_size=1) as w:
            for f in frames:
                w.append_data(np.asarray(f, dtype=np.uint8))
        return path
    except Exception:
        # No mp4 encoder available: remove the truncated stub and fall back to GIF.
        if os.path.exists(path):
            os.remove(path)
        gif = os.path.splitext(path)[0] + ".gif"
        imageio.mimsave(gif, [np.asarray(f, dtype=np.uint8) for f in frames], fps=fps)
        return gif


def save_trajectory(path: str, observations: np.ndarray, actions: np.ndarray,
                    coverages: np.ndarray) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, observations=observations, actions=actions, coverages=coverages)
    return path
