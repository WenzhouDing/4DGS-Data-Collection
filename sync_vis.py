#!/usr/bin/env python3
"""
Sync fine-tune visualizer
=========================
Interactively inspect synced multi-camera video and apply per-camera ±2
frame offset adjustments. Useful when audio-clap sync hits the right second
but a camera's internal clock means the visually-correct frame is 1-2
frames off the auto-detected sync.

Layout: an N-camera grid of frames (same wider-than-tall heuristic as the
sync pipeline), one frame slider scrubbing across the synced timeline, and
one ±2 offset slider per camera.  Saving writes only a small JSON; this
tool never re-trims the videos.

Outputs:
  output/<session>/sync_adjustments.json  — { ref_cam, frame_offsets, ... }

Downstream tools that respect this file (TBD) can apply the offsets when
indexing into the synced videos.

Usage:
    uv run python sync_vis.py [--base DIR] [--session SESSION]
                              [--cams N] [--ref-cam N]
"""

import argparse
import json
import math
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


def parse_args():
    p = argparse.ArgumentParser(description="Sync fine-tune visualizer")
    p.add_argument("--base", default=".", help="Project root")
    p.add_argument("--session", default="session_01", help="Synced session")
    p.add_argument("--cams", type=int, default=12, help="Number of cameras")
    p.add_argument("--ref-cam", type=int, default=1,
                   help="Reference camera (its offset stays at 0)")
    p.add_argument("--start-frame", type=int, default=None,
                   help="Initial frame to display (default: 10%% of total)")
    return p.parse_args()


def grid_layout(n):
    """Wider-than-tall grid heuristic (matches sync_pipeline preview)."""
    if n <= 3:
        return n, 1
    cols = math.ceil(n / 2)
    rows = math.ceil(n / cols)
    return cols, rows


def read_frame(cap, frame_idx, total_frames):
    """Seek and decode the frame at frame_idx (clamped). Returns RGB or None."""
    fi = max(0, min(total_frames - 1, int(frame_idx)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, bgr = cap.read()
    if not ret:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main():
    args = parse_args()
    BASE = os.path.abspath(args.base)
    SESSION = args.session
    NUM_CAMS = args.cams
    REF_CAM = args.ref_cam

    if not (1 <= REF_CAM <= NUM_CAMS):
        print(f"ERROR: --ref-cam {REF_CAM} must be in 1..{NUM_CAMS}")
        raise SystemExit(1)

    SYNCED_DIR = os.path.join(BASE, "output", SESSION, "synced_raw")
    SAVE_PATH = os.path.join(BASE, "output", SESSION, "sync_adjustments.json")

    # Open captures
    caps = {}
    for cam in range(1, NUM_CAMS + 1):
        vid = os.path.join(SYNCED_DIR, f"cam{cam}_synced.mp4")
        if not os.path.exists(vid):
            print(f"  WARNING: {vid} not found, skipping cam {cam}")
            continue
        cap = cv2.VideoCapture(vid)
        if not cap.isOpened():
            print(f"  WARNING: cannot open {vid}")
            cap.release()
            continue
        caps[cam] = cap

    if not caps:
        print(f"ERROR: no synced videos found in {SYNCED_DIR}")
        raise SystemExit(1)
    if REF_CAM not in caps:
        print(f"ERROR: synced video for reference cam {REF_CAM} not found")
        raise SystemExit(1)

    total_frames = int(caps[REF_CAM].get(cv2.CAP_PROP_FRAME_COUNT))
    fps = caps[REF_CAM].get(cv2.CAP_PROP_FPS)
    print(f"Loaded {len(caps)} cameras, {total_frames} frames @ {fps:.2f} fps")
    print(f"Reference: cam{REF_CAM}")

    # Load existing adjustments if present (resume editing)
    offsets = {cam: 0 for cam in caps}
    if os.path.exists(SAVE_PATH):
        try:
            with open(SAVE_PATH) as f:
                saved = json.load(f)
            for cam in caps:
                v = saved.get("frame_offsets", {}).get(f"cam{cam}")
                if v is not None:
                    offsets[cam] = int(v)
            print(f"Resumed from {os.path.basename(SAVE_PATH)}: "
                  f"{ {f'cam{c}': offsets[c] for c in caps} }")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"WARNING: could not load {SAVE_PATH}: {e}")

    # Layout
    grid_cols, grid_rows = grid_layout(len(caps))
    fig = plt.figure(figsize=(max(10, grid_cols * 2.5), grid_rows * 2 + 3.5))
    fig.suptitle(
        f"Sync visualizer — {SESSION}   "
        f"(cam{REF_CAM} = reference, ±2 frames per cam)",
        fontsize=11,
    )

    # Video axes
    video_axes = {}
    img_artists = {}
    cams_sorted = sorted(caps.keys())
    for i, cam in enumerate(cams_sorted):
        r = i // grid_cols
        c = i % grid_cols
        ax_left = 0.005 + c * (1.0 / grid_cols)
        ax_w = (1.0 / grid_cols) * 0.97
        # Top region: y in [0.30, 0.95]
        top, bot = 0.95, 0.30
        ax_h = (top - bot) / grid_rows * 0.92
        ax_bottom = top - (r + 1) * ((top - bot) / grid_rows) + 0.01
        ax = fig.add_axes([ax_left, ax_bottom, ax_w, ax_h])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"cam{cam}{' (ref)' if cam == REF_CAM else ''}",
                     color="red" if cam == REF_CAM else "black", fontsize=9)
        artist = ax.imshow(np.zeros((100, 100, 3), dtype=np.uint8))
        video_axes[cam] = ax
        img_artists[cam] = artist

    # Frame slider
    init_frame = (args.start_frame if args.start_frame is not None
                  else max(0, total_frames // 10))
    ax_frame = fig.add_axes([0.10, 0.22, 0.80, 0.03])
    frame_slider = Slider(ax_frame, "Frame", 0, max(0, total_frames - 1),
                          valinit=init_frame, valstep=1)

    # Per-camera offset sliders, in up to two rows
    offset_sliders = {}
    n_cams = len(cams_sorted)
    slider_cols = min(6, n_cams)
    for i, cam in enumerate(cams_sorted):
        r = i // slider_cols
        c = i % slider_cols
        ax_left = 0.07 + c * (0.86 / slider_cols)
        ax_w = (0.86 / slider_cols) * 0.85
        ax_bottom = 0.13 - r * 0.045
        ax_s = fig.add_axes([ax_left, ax_bottom, ax_w, 0.025])
        label = f"c{cam}{'*' if cam == REF_CAM else ''}"
        s = Slider(ax_s, label, -2, 2, valinit=offsets[cam], valstep=1)
        offset_sliders[cam] = s

    # Save button + status
    ax_save = fig.add_axes([0.45, 0.02, 0.10, 0.045])
    btn_save = Button(ax_save, "Save")
    ax_status = fig.add_axes([0.05, 0.02, 0.38, 0.045])
    ax_status.axis("off")
    status_text = ax_status.text(0, 0.5, "", fontsize=10)

    # Render helpers — only re-decode what changed for snappy interactivity
    def render_cam(cam):
        f = int(frame_slider.val)
        off = int(offset_sliders[cam].val)
        target = f + off
        img = read_frame(caps[cam], target, total_frames)
        if img is not None:
            img_artists[cam].set_data(img)
        video_axes[cam].set_title(
            f"cam{cam}{' (ref)' if cam == REF_CAM else ''}  "
            f"f={target} (off={off:+d})",
            color="red" if cam == REF_CAM else "black", fontsize=9,
        )

    def render_all():
        for cam in cams_sorted:
            render_cam(cam)
        status_text.set_text("")
        fig.canvas.draw_idle()

    def on_frame_change(_):
        render_all()

    def on_offset_change(cam):
        def cb(_):
            render_cam(cam)
            status_text.set_text("")
            fig.canvas.draw_idle()
        return cb

    frame_slider.on_changed(on_frame_change)
    for cam, s in offset_sliders.items():
        s.on_changed(on_offset_change(cam))

    def on_save(_):
        out = {
            "session": SESSION,
            "ref_cam": REF_CAM,
            "fps": fps,
            "total_frames": total_frames,
            "frame_offsets": {f"cam{c}": int(offset_sliders[c].val)
                              for c in cams_sorted},
        }
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        with open(SAVE_PATH, "w") as fh:
            json.dump(out, fh, indent=2)
        status_text.set_text(f"Saved -> {os.path.basename(SAVE_PATH)}")
        fig.canvas.draw_idle()
        print(f"Saved {SAVE_PATH}: {out['frame_offsets']}")

    btn_save.on_clicked(on_save)

    render_all()
    plt.show()

    for cap in caps.values():
        cap.release()


if __name__ == "__main__":
    main()
