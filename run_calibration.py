#!/usr/bin/env python3
"""
GoPro Hero 10 Intrinsic Calibration from Checkerboard
=====================================================
Extracts frames from synced multi-camera video, auto-detects the checkerboard
size, runs cv2.calibrateCamera() per camera, and outputs:
  - Minimal intrinsics JSON (K, distortion, image size, RMS)
  - Combined calibration JSON + checkerboard config
  - Validation images: corner overlays + per-frame error bar chart

Usage:
    python run_calibration.py [--base DIR] [--session SESSION] [--cams N]
                              [--square-size M] [--fps-extract FPS]

Defaults:
    --base          .               (project root containing output/ folder)
    --session       session_04      (session folder name under output/)
    --cams          5
    --square-size   0.03            (square side length in metres)
    --fps-extract   0.5             (frames per second to extract for calibration)
"""

import argparse
import cv2
import numpy as np
import json
import os
import glob
import subprocess


# ─── CLI ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Checkerboard intrinsic calibration")
    p.add_argument("--base", default=".", help="Project root (contains output/ folder)")
    p.add_argument("--session", default="session_04", help="Session to calibrate from")
    p.add_argument("--cams", type=int, default=5, help="Number of cameras")
    p.add_argument("--square-size", type=float, default=0.03,
                   help="Checkerboard square side in metres (default 0.03 = 3 cm)")
    p.add_argument("--fps-extract", type=float, default=0.5,
                   help="Extraction rate in fps (default 0.5 = one frame every 2 s)")
    return p.parse_args()


# ─── BOARD CANDIDATES ────────────────────────────────────────────
BOARD_CANDIDATES = [
    (9, 6),   # 10x7 squares
    (8, 6),   # 9x7 squares
    (7, 5),   # 8x6 squares
    (8, 5),   # 9x6 squares
    (7, 4),   # 8x5 squares
    (6, 4),   # 7x5 squares
    (10, 7),  # 11x8 squares
]

# How many sample frames to draw corner overlays on per camera
VALIDATION_SAMPLE_COUNT = 4


# ─── VALIDATION VISUALISATION ────────────────────────────────────
def draw_corner_overlay(img, corners_detected, corners_reprojected, board_size):
    """Draw detected (green) and reprojected (red) corners on an image copy."""
    vis = img.copy()
    # Detected corners in green
    cv2.drawChessboardCorners(vis, board_size, corners_detected, True)
    # Reprojected corners in red circles
    for pt in corners_reprojected.reshape(-1, 2):
        cv2.circle(vis, (int(pt[0]), int(pt[1])), 8, (0, 0, 255), 2)
    return vis


def make_error_bar_chart(per_frame_errors, cam_label, output_path):
    """Save a per-frame reprojection error bar chart as PNG using matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, len(per_frame_errors) * 0.18), 4))
    x = np.arange(len(per_frame_errors))
    colors = ["#e74c3c" if e > 0.1 else "#f39c12" if e > 0.06 else "#2ecc71"
              for e in per_frame_errors]
    ax.bar(x, per_frame_errors, color=colors, width=0.8)
    ax.axhline(np.mean(per_frame_errors), color="#3498db", linestyle="--",
               linewidth=1.5, label=f"mean = {np.mean(per_frame_errors):.4f} px")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Reprojection error (px)")
    ax.set_title(f"{cam_label} — Per-Frame Reprojection Error")
    ax.legend()
    ax.set_xlim(-0.5, len(per_frame_errors) - 0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def make_summary_chart(cam_rms, output_path):
    """Bar chart comparing RMS across all cameras."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(cam_rms.keys())
    values = list(cam_rms.values())

    fig, ax = plt.subplots(figsize=(max(4, len(labels) * 1.2), 4))
    bars = ax.bar(labels, values, color="#3498db", width=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("RMS Reprojection Error (px)")
    ax.set_title("Calibration RMS — All Cameras")
    ax.set_ylim(0, max(values) * 1.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ─── MAIN ─────────────────────────────────────────────────────────
def main():
    args = parse_args()

    BASE = os.path.abspath(args.base)
    SESSION = args.session
    NUM_CAMS = args.cams
    SQUARE_SIZE_M = args.square_size
    FPS_EXTRACT = args.fps_extract

    SYNCED_DIR = os.path.join(BASE, "output", SESSION, "synced_raw")
    OUTPUT = os.path.join(BASE, "output", "calibration")
    VALIDATION = os.path.join(OUTPUT, "validation")
    WORK = os.path.join(BASE, ".calib_work")
    os.makedirs(OUTPUT, exist_ok=True)
    os.makedirs(VALIDATION, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)

    # ── Extract frames ────────────────────────────────────────────
    print("=" * 70)
    print(f"Extracting frames from {SESSION} at {FPS_EXTRACT} fps")
    print("=" * 70)

    for cam in range(1, NUM_CAMS + 1):
        vid = os.path.join(SYNCED_DIR, f"cam{cam}_synced.mp4")
        if not os.path.exists(vid):
            print(f"  WARNING: {vid} not found, skipping cam {cam}")
            continue
        out_dir = os.path.join(WORK, f"cam{cam}_hires")
        os.makedirs(out_dir, exist_ok=True)
        existing = glob.glob(os.path.join(out_dir, "*.jpg"))
        if existing:
            print(f"  Cam {cam}: {len(existing)} frames already extracted, skipping")
            continue
        subprocess.run([
            "ffmpeg", "-v", "quiet", "-i", vid,
            "-vf", f"fps={FPS_EXTRACT}",
            "-q:v", "2",
            os.path.join(out_dir, "frame_%04d.jpg"),
        ], check=True)
        extracted = glob.glob(os.path.join(out_dir, "*.jpg"))
        print(f"  Cam {cam}: extracted {len(extracted)} frames")

    # ── Auto-detect board size ────────────────────────────────────
    print("\n" + "=" * 70)
    print("Auto-detecting checkerboard size...")
    print("=" * 70)

    detected_board = None
    test_frames = sorted(glob.glob(os.path.join(WORK, "cam1_hires", "*.jpg")))

    for board_size in BOARD_CANDIDATES:
        hits = 0
        for fpath in test_frames[:40]:
            img = cv2.imread(fpath)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, _ = cv2.findChessboardCorners(
                gray, board_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH +
                cv2.CALIB_CB_NORMALIZE_IMAGE +
                cv2.CALIB_CB_FAST_CHECK,
            )
            if ret:
                hits += 1
        print(f"  Board {board_size[0]}x{board_size[1]}: "
              f"detected in {hits}/{min(len(test_frames), 40)} frames")
        if hits >= 5 and (detected_board is None or hits > detected_board[1]):
            detected_board = (board_size, hits)

    if detected_board is None:
        print("ERROR: Could not detect any checkerboard pattern!")
        raise SystemExit(1)

    BOARD_SIZE = detected_board[0]
    print(f"\n  -> Using board size: {BOARD_SIZE[0]}x{BOARD_SIZE[1]} inner corners "
          f"({detected_board[1]} detections)")

    # ── Object points ─────────────────────────────────────────────
    objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_M

    # ── Calibrate each camera ─────────────────────────────────────
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    all_results = {}
    cam_rms_map = {}

    for cam in range(1, NUM_CAMS + 1):
        print(f"\n{'=' * 70}")
        print(f"Calibrating Camera {cam}")
        print("=" * 70)

        frames = sorted(glob.glob(os.path.join(WORK, f"cam{cam}_hires", "*.jpg")))
        obj_points = []
        img_points = []
        used_frame_paths = []
        img_shape = None

        for fpath in frames:
            img = cv2.imread(fpath)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img_shape is None:
                img_shape = gray.shape[::-1]  # (w, h)

            ret, corners = cv2.findChessboardCorners(
                gray, BOARD_SIZE,
                cv2.CALIB_CB_ADAPTIVE_THRESH +
                cv2.CALIB_CB_NORMALIZE_IMAGE +
                cv2.CALIB_CB_FAST_CHECK,
            )
            if ret:
                corners_refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria
                )
                obj_points.append(objp)
                img_points.append(corners_refined)
                used_frame_paths.append(fpath)

        print(f"  Detected checkerboard in {len(used_frame_paths)}/{len(frames)} frames")
        print(f"  Image size: {img_shape}")

        if len(obj_points) < 5:
            print(f"  WARNING: not enough detections ({len(obj_points)}), need >= 5")
            all_results[f"cam{cam}"] = {"error": "insufficient detections"}
            continue

        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, img_shape, None, None
        )

        # Per-frame reprojection error
        reproj_errors = []
        reproj_points = []
        for i in range(len(obj_points)):
            proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
            err = cv2.norm(img_points[i], proj, cv2.NORM_L2) / len(proj)
            reproj_errors.append(err)
            reproj_points.append(proj)

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        print(f"\n  Results:")
        print(f"    RMS reprojection error: {ret:.4f} px")
        print(f"    Mean per-frame error:   {np.mean(reproj_errors):.4f} px")
        print(f"    Max per-frame error:    {np.max(reproj_errors):.4f} px")
        print(f"    Focal length: fx={fx:.2f}, fy={fy:.2f} px")
        print(f"    Principal point: cx={cx:.2f}, cy={cy:.2f} px")
        print(f"    Distortion: {dist.flatten()}")

        cam_rms_map[f"Cam {cam}"] = ret

        # ── Minimal intrinsics JSON (only what downstream needs) ──
        result = {
            "image_size": [img_shape[0], img_shape[1]],
            "K": K.tolist(),
            "dist": dist.flatten().tolist(),
            "rms_error_px": round(ret, 6),
        }
        all_results[f"cam{cam}"] = result

        cam_path = os.path.join(OUTPUT, f"cam{cam}_intrinsics.json")
        with open(cam_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"    -> Saved {os.path.basename(cam_path)}")

        # ── Validation: corner overlay images ─────────────────────
        print(f"\n  Generating validation images...")
        cam_val_dir = os.path.join(VALIDATION, f"cam{cam}")
        os.makedirs(cam_val_dir, exist_ok=True)

        # Pick evenly spaced sample frames
        n = len(used_frame_paths)
        if n <= VALIDATION_SAMPLE_COUNT:
            sample_indices = list(range(n))
        else:
            sample_indices = [int(i * (n - 1) / (VALIDATION_SAMPLE_COUNT - 1))
                              for i in range(VALIDATION_SAMPLE_COUNT)]

        for idx in sample_indices:
            fpath = used_frame_paths[idx]
            fname = os.path.basename(fpath)
            img = cv2.imread(fpath)
            vis = draw_corner_overlay(img, img_points[idx], reproj_points[idx], BOARD_SIZE)
            # Scale down for reasonable file size
            h_vis, w_vis = vis.shape[:2]
            scale = 1200 / w_vis
            vis_small = cv2.resize(vis, (int(w_vis * scale), int(h_vis * scale)))
            out_path = os.path.join(cam_val_dir, f"corners_{fname}")
            cv2.imwrite(out_path, vis_small, [cv2.IMWRITE_JPEG_QUALITY, 85])
            print(f"    -> {os.path.relpath(out_path, OUTPUT)}")

        # ── Validation: per-frame error bar chart ─────────────────
        chart_path = os.path.join(cam_val_dir, "reproj_error_per_frame.png")
        make_error_bar_chart(reproj_errors, f"Cam {cam}", chart_path)
        print(f"    -> {os.path.relpath(chart_path, OUTPUT)}")

    # ── Summary chart across all cameras ──────────────────────────
    if cam_rms_map:
        summary_path = os.path.join(VALIDATION, "rms_all_cameras.png")
        make_summary_chart(cam_rms_map, summary_path)
        print(f"\n  -> validation/rms_all_cameras.png")

    # ── Combined calibration file ─────────────────────────────────
    print(f"\n{'=' * 70}")
    print("Saving combined calibration")
    print("=" * 70)

    combined = {
        "checkerboard": {
            "inner_corners": list(BOARD_SIZE),
            "square_size_m": SQUARE_SIZE_M,
            "board_squares": [BOARD_SIZE[0] + 1, BOARD_SIZE[1] + 1],
        },
        "source_session": SESSION,
        "cameras": all_results,
    }

    combined_path = os.path.join(OUTPUT, "calibration_all_cameras.json")
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"  -> {os.path.basename(combined_path)}")

    # ── Checkerboard config file ──────────────────────────────────
    config = {
        "inner_corners": list(BOARD_SIZE),
        "square_size_m": SQUARE_SIZE_M,
        "board_squares": [BOARD_SIZE[0] + 1, BOARD_SIZE[1] + 1],
    }
    config_path = os.path.join(OUTPUT, "checkerboard_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  -> {os.path.basename(config_path)}")

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("CALIBRATION SUMMARY")
    print("=" * 70)
    print(f"  Board: {BOARD_SIZE[0]+1}x{BOARD_SIZE[1]+1} squares, "
          f"{BOARD_SIZE[0]}x{BOARD_SIZE[1]} inner corners, "
          f"{SQUARE_SIZE_M*100:.0f}cm/square")
    for cam in range(1, NUM_CAMS + 1):
        r = all_results.get(f"cam{cam}", {})
        if "error" in r:
            print(f"  Cam {cam}: FAILED - {r['error']}")
        else:
            K_arr = np.array(r["K"])
            print(f"  Cam {cam}: RMS={r['rms_error_px']:.4f}px, "
                  f"fx={K_arr[0,0]:.1f}, fy={K_arr[1,1]:.1f}, "
                  f"cx={K_arr[0,2]:.1f}, cy={K_arr[1,2]:.1f}")
    print(f"\nOutput: {OUTPUT}/")
    print(f"Validation: {VALIDATION}/")
    print("Done!")


if __name__ == "__main__":
    main()
