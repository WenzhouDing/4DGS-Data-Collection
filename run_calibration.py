#!/usr/bin/env python3
"""
GoPro Hero 10 Intrinsic + Extrinsic Calibration
================================================
Reads synced multi-camera video directly in lockstep (all cameras
advance together on the same frame numbers).  Detects checkerboard
corners using findChessboardCornersSB (sector-based) on downscaled
frames, with full-resolution cornerSubPix refinement.  Detection is
parallelised across cameras via ThreadPoolExecutor within each frame.
Early stopping fires after --max-frames frames where ALL cameras
detected the board, guaranteeing shared frames for stereo calibration.

Only frames with detected corners are saved to disk (for validation).

Outputs:
  output/calibration/
    cam{N}_intrinsics.json       — K, dist, image_size, rms
    cam{N}_extrinsics.json       — R, T, E, F, stereo_rms (vs cam1)
    calibration_all_cameras.json — combined intrinsics + extrinsics
    checkerboard_config.json
    frame_extraction_log.json    — source frame numbers per camera
    validation/
      cam{N}/corners_*.jpg       — corner overlay samples
      cam{N}/reproj_error.png    — per-frame intrinsic error
      stereo/pair_1_{N}_epipolar_*.jpg
      stereo/stereo_rms.png
      rms_all_cameras.png

Usage:
    python run_calibration.py --board 9x12 --square-size 0.03 \\
        [--base DIR] [--session SESSION] [--cams N] [--every N]
"""

import argparse
import cv2
import numpy as np
import json
import os
import sys
import glob
import tempfile
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


# ─── CLI ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Checkerboard intrinsic + extrinsic calibration")
    p.add_argument("--base", default=".",
                   help="Project root (contains output/ folder)")
    p.add_argument("--session", default="session_01",
                   help="Session to calibrate from")
    p.add_argument("--cams", type=int, default=5,
                   help="Number of cameras")
    p.add_argument("--board", required=True,
                   help="Board size as COLSxROWS in squares, e.g. '9x12'")
    p.add_argument("--square-size", type=float, required=True,
                   help="Square side in metres (e.g. 0.03 for 30 mm)")
    p.add_argument("--every", type=int, default=30,
                   help="Process every Nth frame (default 30 = 4fps at 120fps). "
                        "Use 1 for all frames at the cost of speed.")
    p.add_argument("--max-frames", type=int, default=60,
                   help="Stop after this many frames with ALL cameras detecting "
                        "the board (default 60). 0 = no limit.")
    return p.parse_args()


VALIDATION_SAMPLE_COUNT = 4
TOTAL_STEPS = 3


# ─── PROGRESS HELPERS ────────────────────────────────────────────
def step_header(step_num, title):
    print(f"\n[{step_num}/{TOTAL_STEPS}] {title}")
    print("=" * 70)


def progress(msg, end="\n"):
    sys.stdout.write(msg + end)
    sys.stdout.flush()


def fmt_time(seconds):
    """Format seconds as mm:ss or hh:mm:ss."""
    s = int(seconds)
    if s < 3600:
        return f"{s//60}:{s%60:02d}"
    return f"{s//3600}:{(s%3600)//60:02d}:{s%60:02d}"


# ─── VISUALISATION HELPERS ────────────────────────────────────────
def draw_corner_overlay(img, corners_detected, corners_reprojected, board_size):
    vis = img.copy()
    cv2.drawChessboardCorners(vis, board_size, corners_detected, True)
    for pt in corners_reprojected.reshape(-1, 2):
        cv2.circle(vis, (int(pt[0]), int(pt[1])), 8, (0, 0, 255), 2)
    return vis


def draw_epipolar_lines(img1, img2, pts1, pts2, F, num_lines=12):
    h, w = img1.shape[:2]
    vis1 = img1.copy()
    vis2 = img2.copy()

    n = len(pts1)
    step = max(1, n // num_lines)
    indices = list(range(0, n, step))[:num_lines]

    colors = [tuple(int(c) for c in np.random.RandomState(i).randint(50, 255, 3))
              for i in range(len(indices))]

    for ci, idx in enumerate(indices):
        color = colors[ci]
        p1 = pts1[idx].reshape(1, 1, 2).astype(np.float64)
        p2 = pts2[idx].reshape(1, 1, 2).astype(np.float64)

        line2 = cv2.computeCorrespondEpilines(p1, 1, F).reshape(-1, 3)
        a, b, c = line2[0]
        x0, x1 = 0, w
        y0 = int(-c / b) if abs(b) > 1e-6 else 0
        y1 = int(-(c + a * w) / b) if abs(b) > 1e-6 else h
        cv2.line(vis2, (x0, y0), (x1, y1), color, 2)
        cv2.circle(vis2, (int(pts2[idx][0]), int(pts2[idx][1])), 10, color, -1)

        line1 = cv2.computeCorrespondEpilines(p2, 2, F).reshape(-1, 3)
        a, b, c = line1[0]
        y0 = int(-c / b) if abs(b) > 1e-6 else 0
        y1 = int(-(c + a * w) / b) if abs(b) > 1e-6 else h
        cv2.line(vis1, (x0, y0), (x1, y1), color, 2)
        cv2.circle(vis1, (int(pts1[idx][0]), int(pts1[idx][1])), 10, color, -1)

    return vis1, vis2


def make_bar_chart(values, labels, title, ylabel, output_path,
                   threshold_colors=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if threshold_colors:
        colors = [threshold_colors(v) for v in values]
    else:
        colors = "#3498db"

    fig, ax = plt.subplots(figsize=(max(4, len(labels) * 1.2), 4))
    bars = ax.bar(labels, values, color=colors, width=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(values) * 0.02,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(values) * 1.3 if values else 1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def make_error_bar_chart(per_frame_errors, cam_label, output_path):
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


def scale_image(img, target_width=1200):
    h, w = img.shape[:2]
    scale = target_width / w
    return cv2.resize(img, (int(w * scale), int(h * scale)))


# ─── CORNER DETECTION ────────────────────────────────────────────
DETECT_WIDTH = 960  # downscale to this width for findChessboardCornersSB


def _detect_frame(gray, board_size):
    """Detect checkerboard in one frame. Returns refined corners or None.

    Uses findChessboardCornersSB on a downscaled image for speed,
    then cornerSubPix at full resolution for accuracy.
    """
    h, w = gray.shape
    if w > DETECT_WIDTH:
        scale = DETECT_WIDTH / w
        small = cv2.resize(gray, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
    else:
        small = gray
        scale = 1.0

    ret, corners = cv2.findChessboardCornersSB(
        small, board_size, cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ret:
        return None

    if scale != 1.0:
        corners /= scale
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30, 0.001)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)


# ─── MAIN ─────────────────────────────────────────────────────────
def main():
    args = parse_args()

    BASE = os.path.abspath(args.base)
    SESSION = args.session
    NUM_CAMS = args.cams
    SQUARE_SIZE_M = args.square_size
    EVERY_N = args.every
    MAX_FRAMES = args.max_frames

    # Parse board size: "9x12" squares -> (8, 11) inner corners
    parts = args.board.lower().split("x")
    if len(parts) != 2:
        print(f"ERROR: --board must be COLSxROWS, e.g. '9x12', got '{args.board}'")
        raise SystemExit(1)
    try:
        board_cols, board_rows = int(parts[0]), int(parts[1])
    except ValueError:
        print(f"ERROR: --board values must be integers, got '{args.board}'")
        raise SystemExit(1)
    BOARD_SIZE = (board_cols - 1, board_rows - 1)

    SYNCED_DIR = os.path.join(BASE, "output", SESSION, "synced_raw")
    OUTPUT = os.path.join(BASE, "output", "calibration")
    VALIDATION = os.path.join(OUTPUT, "validation")
    STEREO_VAL = os.path.join(VALIDATION, "stereo")
    os.makedirs(OUTPUT, exist_ok=True)
    os.makedirs(VALIDATION, exist_ok=True)
    os.makedirs(STEREO_VAL, exist_ok=True)

    # Print config
    print("=" * 70)
    print("MULTI-CAMERA CALIBRATION")
    print("=" * 70)
    print(f"  Session:       {SESSION}")
    print(f"  Cameras:       {NUM_CAMS}")
    print(f"  Board:         {board_cols}x{board_rows} squares "
          f"-> {BOARD_SIZE[0]}x{BOARD_SIZE[1]} inner corners")
    print(f"  Square size:   {SQUARE_SIZE_M*1000:.1f} mm")
    print(f"  Frame skip:    every {EVERY_N} frame(s)"
          f"{' (all frames)' if EVERY_N == 1 else ''}")
    print(f"  Max frames:    {MAX_FRAMES} shared (all cams)"
          f"{' (no limit)' if MAX_FRAMES == 0 else ''}")
    print(f"  Detect width:  {DETECT_WIDTH}px (downscaled for speed)")
    print(f"  Synced dir:    {SYNCED_DIR}")
    print(f"  Output:        {OUTPUT}")

    # Object points
    objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_M

    # Temp dir for saving only detected frames (for validation images)
    tmpdir = tempfile.mkdtemp(prefix="gopro_calib_")

    try:
        # ══════════════════════════════════════════════════════════
        # STEP 1: READ VIDEO + DETECT CORNERS (merged for efficiency)
        # ══════════════════════════════════════════════════════════
        # Reads directly from video — no ffmpeg extraction step.
        # Only frames with detected corners are saved to disk.
        step_header(1, f"Detecting corners ({BOARD_SIZE[0]}x{BOARD_SIZE[1]} inner)")

        # Open all video captures
        caps = {}
        cam_tmp_dirs = {}
        for cam in range(1, NUM_CAMS + 1):
            vid = os.path.join(SYNCED_DIR, f"cam{cam}_synced.mp4")
            if not os.path.exists(vid):
                print(f"  WARNING: {vid} not found, skipping")
                continue
            cap = cv2.VideoCapture(vid)
            if cap.get(cv2.CAP_PROP_FPS) <= 0:
                print(f"  WARNING: cannot read {vid}, skipping")
                cap.release()
                continue
            caps[cam] = cap
            cam_tmp_dirs[cam] = os.path.join(tmpdir, f"cam{cam}")
            os.makedirs(cam_tmp_dirs[cam], exist_ok=True)

        if not caps:
            print("\nERROR: No valid video files found!")
            raise SystemExit(1)

        source_fps = list(caps.values())[0].get(cv2.CAP_PROP_FPS)
        total_frames = int(list(caps.values())[0].get(cv2.CAP_PROP_FRAME_COUNT))

        cam_corners = {cam: {} for cam in range(1, NUM_CAMS + 1)}
        cam_detected_frames = {cam: [] for cam in caps}
        img_shape = None
        shared_count = 0
        scanned = 0
        t0 = time.time()
        last_progress = t0

        # Process all cameras in lockstep — same frame number, parallel detection.
        # If any camera can't grab a frame, lockstep is broken — stop rather
        # than silently skip, since mismatched frame indices would corrupt
        # stereo correspondences.
        with ThreadPoolExecutor(max_workers=len(caps)) as pool:
            frame_num = 0
            failed_cam = None
            while frame_num < total_frames:
                # Advance all captures together
                all_ok = True
                for cam in caps:
                    if not caps[cam].grab():
                        failed_cam = cam
                        all_ok = False
                        break
                if not all_ok:
                    print(f"\n  Cam {failed_cam}: grab() failed at frame "
                          f"{frame_num}/{total_frames} — stopping detection "
                          f"(this is normal at end-of-stream).")
                    break

                if frame_num % EVERY_N == 0:
                    scanned += 1

                    # Retrieve frames from all cameras
                    imgs = {}
                    grays = {}
                    for cam in caps:
                        ret, img = caps[cam].retrieve()
                        if ret:
                            imgs[cam] = img
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            grays[cam] = gray
                            if img_shape is None:
                                img_shape = gray.shape[::-1]

                    # Detect corners across all cameras in parallel
                    futures = {
                        pool.submit(_detect_frame, grays[cam], BOARD_SIZE): cam
                        for cam in grays
                    }
                    results = {}
                    for fut in as_completed(futures):
                        results[futures[fut]] = fut.result()

                    # Record detections
                    fname = f"frame_{frame_num:06d}.jpg"
                    frame_detected = []
                    for cam, corners in results.items():
                        if corners is not None:
                            fpath = os.path.join(cam_tmp_dirs[cam], fname)
                            cv2.imwrite(fpath, imgs[cam],
                                        [cv2.IMWRITE_JPEG_QUALITY, 95])
                            cam_corners[cam][fname] = (corners, fpath)
                            cam_detected_frames[cam].append(frame_num)
                            frame_detected.append(cam)

                    if len(frame_detected) == len(caps):
                        shared_count += 1

                    # Progress (every 0.5s)
                    now = time.time()
                    if now - last_progress >= 0.5:
                        last_progress = now
                        el = now - t0
                        fps_p = scanned / el if el > 0 else 0
                        pct = 100 * frame_num / total_frames
                        eta = ((total_frames - frame_num)
                               / (frame_num / el)
                               if frame_num > 0 and el > 0 else 0)
                        det_counts = [len(cam_corners[c])
                                      for c in sorted(caps.keys())]
                        progress(
                            f"\r  {pct:.0f}%  "
                            f"shared: {shared_count}"
                            f"{'/' + str(MAX_FRAMES) if MAX_FRAMES > 0 else ''}  "
                            f"per-cam: {det_counts}  "
                            f"[{fps_p:.0f} fps, ETA {fmt_time(eta)}]"
                            f"          ",
                            end="")

                    # Early stopping — enough shared detections
                    if MAX_FRAMES > 0 and shared_count >= MAX_FRAMES:
                        break

                frame_num += 1

        for cam in caps:
            caps[cam].release()
        elapsed = time.time() - t0

        progress(
            f"\r  Done — {shared_count} shared detections, "
            f"{scanned} frames scanned  [{fmt_time(elapsed)}]"
            f"                              \n", end="")

        if img_shape is None:
            print("\nERROR: No frames could be read from any camera!")
            raise SystemExit(1)

        # Build scan info for extraction log
        cam_scan_info = {}
        for cam in range(1, NUM_CAMS + 1):
            cam_scan_info[cam] = {
                "source_fps": source_fps if cam in caps else 0,
                "total_frames": total_frames if cam in caps else 0,
                "every_n": EVERY_N,
                "scanned": scanned if cam in caps else 0,
                "detected": len(cam_corners[cam]),
                "frame_numbers": cam_detected_frames.get(cam, []),
                "elapsed_sec": round(elapsed, 1),
            }

        # Detection summary
        print()
        total_det = sum(len(cam_corners[c]) for c in range(1, NUM_CAMS + 1))
        print(f"  Total: {total_det} detections across {NUM_CAMS} cameras "
              f"({scanned} frames scanned, {shared_count} shared)")

        # ══════════════════════════════════════════════════════════
        # STEP 2: INTRINSIC CALIBRATION
        # ══════════════════════════════════════════════════════════
        step_header(2, "Intrinsic calibration")

        intrinsics = {}
        cam_rms_map = {}

        for cam in range(1, NUM_CAMS + 1):
            corners_dict = cam_corners[cam]
            if len(corners_dict) < 5:
                print(f"  Cam {cam}/{NUM_CAMS}: SKIP — "
                      f"only {len(corners_dict)} detections (need >= 5)")
                intrinsics[cam] = None
                continue

            progress(
                f"  Cam {cam}/{NUM_CAMS}: calibrating "
                f"({len(corners_dict)} frames)...", end="")

            frame_names = sorted(corners_dict.keys())
            obj_pts = [objp] * len(frame_names)
            img_pts = [corners_dict[fn][0] for fn in frame_names]

            try:
                ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                    obj_pts, img_pts, img_shape, None, None
                )
            except cv2.error as e:
                print(f"\r  Cam {cam}/{NUM_CAMS}: FAILED — calibrateCamera "
                      f"raised: {e}                                   ")
                intrinsics[cam] = None
                continue

            reproj_errors = []
            reproj_pts = []
            for i in range(len(obj_pts)):
                proj, _ = cv2.projectPoints(
                    obj_pts[i], rvecs[i], tvecs[i], K, dist)
                err = cv2.norm(img_pts[i], proj, cv2.NORM_L2) / len(proj)
                reproj_errors.append(err)
                reproj_pts.append(proj)

            intrinsics[cam] = {
                "K": K, "dist": dist, "rms": ret,
                "rvecs": rvecs, "tvecs": tvecs,
                "frame_names": frame_names,
                "img_pts": img_pts, "reproj_pts": reproj_pts,
                "reproj_errors": reproj_errors,
            }
            cam_rms_map[f"Cam {cam}"] = ret

            progress(
                f"\r  Cam {cam}/{NUM_CAMS}: RMS={ret:.4f}px  "
                f"fx={K[0,0]:.1f} fy={K[1,1]:.1f} "
                f"cx={K[0,2]:.1f} cy={K[1,2]:.1f}  "
                f"({len(corners_dict)} frames)          ")

            # Save JSON
            result_json = {
                "image_size": [img_shape[0], img_shape[1]],
                "K": K.tolist(),
                "dist": dist.flatten().tolist(),
                "rms_error_px": round(ret, 6),
            }
            cam_path = os.path.join(OUTPUT, f"cam{cam}_intrinsics.json")
            with open(cam_path, "w") as f:
                json.dump(result_json, f, indent=2)

            # Validation: corner overlays
            cam_val_dir = os.path.join(VALIDATION, f"cam{cam}")
            os.makedirs(cam_val_dir, exist_ok=True)
            n = len(frame_names)
            if n <= VALIDATION_SAMPLE_COUNT:
                sample_idx = list(range(n))
            else:
                sample_idx = [int(i * (n - 1) / (VALIDATION_SAMPLE_COUNT - 1))
                              for i in range(VALIDATION_SAMPLE_COUNT)]

            for idx in sample_idx:
                fname = frame_names[idx]
                fpath = corners_dict[fname][1]
                img = cv2.imread(fpath)
                vis = draw_corner_overlay(
                    img, img_pts[idx], reproj_pts[idx], BOARD_SIZE)
                vis = scale_image(vis)
                out_path = os.path.join(cam_val_dir, f"corners_{fname}")
                cv2.imwrite(out_path, vis, [cv2.IMWRITE_JPEG_QUALITY, 85])

            chart_path = os.path.join(cam_val_dir, "reproj_error_per_frame.png")
            make_error_bar_chart(reproj_errors, f"Cam {cam}", chart_path)

        if cam_rms_map:
            make_bar_chart(
                list(cam_rms_map.values()), list(cam_rms_map.keys()),
                "Intrinsic Calibration RMS — All Cameras",
                "RMS Reprojection Error (px)",
                os.path.join(VALIDATION, "rms_all_cameras.png"),
            )

        # ══════════════════════════════════════════════════════════
        # STEP 3: EXTRINSIC (STEREO) CALIBRATION
        # ══════════════════════════════════════════════════════════
        step_header(3, "Extrinsic calibration (each cam vs Cam 1)")

        ref_cam = 1
        if intrinsics[ref_cam] is None:
            print("  ERROR: Cam 1 intrinsic calibration failed!")
            raise SystemExit(1)

        K1 = intrinsics[ref_cam]["K"]
        dist1 = intrinsics[ref_cam]["dist"]
        extrinsics = {}
        stereo_rms_map = {}

        for cam in range(2, NUM_CAMS + 1):
            if intrinsics[cam] is None:
                print(f"  Pair 1-{cam}: SKIP — cam {cam} intrinsics failed")
                continue

            K2 = intrinsics[cam]["K"]
            dist2 = intrinsics[cam]["dist"]

            # Match by frame name — encodes source frame number,
            # so same name = same source frame = same timestamp.
            ref_names = set(cam_corners[ref_cam].keys())
            other_names = set(cam_corners[cam].keys())
            shared = sorted(ref_names & other_names)

            if len(shared) < 8:
                print(f"  Pair 1-{cam}: SKIP — "
                      f"{len(shared)} shared frames (need >= 8)")
                continue

            progress(
                f"  Pair 1-{cam}: calibrating ({len(shared)} shared)...",
                end="")

            obj_pts_shared = [objp] * len(shared)
            pts1 = [cam_corners[ref_cam][fn][0] for fn in shared]
            pts2 = [cam_corners[cam][fn][0] for fn in shared]

            flags = cv2.CALIB_FIX_INTRINSIC
            stereo_criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                100, 1e-6,
            )

            try:
                ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
                    obj_pts_shared, pts1, pts2,
                    K1, dist1, K2, dist2,
                    img_shape,
                    criteria=stereo_criteria,
                    flags=flags,
                )
            except cv2.error as e:
                print(f"\r  Pair 1-{cam}: FAILED — stereoCalibrate "
                      f"raised: {e}                                   ")
                continue

            # Invert to get camN's pose in cam1's frame
            # stereoCalibrate gives P_camN = R * P_cam1 + T
            # We want camN in cam1: R_inv = R^T, T_inv = -R^T * T
            R_inv = R.T
            T_inv = -R.T @ T

            baseline_m = np.linalg.norm(T_inv)
            sy = np.sqrt(R_inv[0, 0]**2 + R_inv[1, 0]**2)
            if sy > 1e-6:
                ex = np.arctan2(R_inv[2, 1], R_inv[2, 2])
                ey = np.arctan2(-R_inv[2, 0], sy)
                ez = np.arctan2(R_inv[1, 0], R_inv[0, 0])
            else:
                ex = np.arctan2(-R_inv[1, 2], R_inv[1, 1])
                ey = np.arctan2(-R_inv[2, 0], sy)
                ez = 0

            stereo_rms_map[f"1-{cam}"] = ret

            progress(
                f"\r  Pair 1-{cam}: RMS={ret:.4f}px  "
                f"baseline={baseline_m*100:.2f}cm  "
                f"T=[{T_inv[0,0]:.4f}, {T_inv[1,0]:.4f}, {T_inv[2,0]:.4f}]m  "
                f"({len(shared)} shared)          ")

            ext_data = {
                "reference": "cam1",
                "target": f"cam{cam}",
                "R": R_inv.tolist(),
                "T": T_inv.flatten().tolist(),
                "E": E.tolist(),
                "F": F.tolist(),
                "stereo_rms_px": round(ret, 6),
                "baseline_m": round(baseline_m, 6),
                "euler_deg": {
                    "rx": round(np.degrees(ex), 4),
                    "ry": round(np.degrees(ey), 4),
                    "rz": round(np.degrees(ez), 4),
                },
                "shared_frames": len(shared),
            }
            extrinsics[cam] = ext_data

            ext_path = os.path.join(OUTPUT, f"cam{cam}_extrinsics.json")
            with open(ext_path, "w") as f:
                json.dump(ext_data, f, indent=2)

            # Epipolar validation — undistort images and points so F is valid
            sample_shared = [shared[0], shared[len(shared) // 2]]
            for fn in sample_shared:
                fpath1 = cam_corners[ref_cam][fn][1]
                fpath2 = cam_corners[cam][fn][1]
                img1 = cv2.undistort(cv2.imread(fpath1), K1, dist1)
                img2 = cv2.undistort(cv2.imread(fpath2), K2, dist2)

                c1 = cv2.undistortPoints(
                    cam_corners[ref_cam][fn][0].reshape(-1, 1, 2),
                    K1, dist1, P=K1).reshape(-1, 2)
                c2 = cv2.undistortPoints(
                    cam_corners[cam][fn][0].reshape(-1, 1, 2),
                    K2, dist2, P=K2).reshape(-1, 2)

                vis1, vis2 = draw_epipolar_lines(img1, img2, c1, c2, F)
                vis1 = scale_image(vis1)
                vis2 = scale_image(vis2)

                base = fn.replace(".jpg", "")
                p1 = os.path.join(STEREO_VAL,
                                  f"pair_1_{cam}_{base}_cam1.jpg")
                p2 = os.path.join(STEREO_VAL,
                                  f"pair_1_{cam}_{base}_cam{cam}.jpg")
                cv2.imwrite(p1, vis1, [cv2.IMWRITE_JPEG_QUALITY, 85])
                cv2.imwrite(p2, vis2, [cv2.IMWRITE_JPEG_QUALITY, 85])

        if stereo_rms_map:
            make_bar_chart(
                list(stereo_rms_map.values()),
                [f"Cam 1-{k.split('-')[1]}" for k in stereo_rms_map.keys()],
                "Stereo Calibration RMS — All Pairs",
                "Stereo RMS (px)",
                os.path.join(STEREO_VAL, "stereo_rms.png"),
            )

        # ══════════════════════════════════════════════════════════
        # SAVE COMBINED FILES
        # ══════════════════════════════════════════════════════════
        print(f"\n{'=' * 70}")
        print("Saving outputs")
        print("=" * 70)

        cam_data = {}
        for cam in range(1, NUM_CAMS + 1):
            entry = {}
            if intrinsics[cam] is not None:
                entry["image_size"] = [img_shape[0], img_shape[1]]
                entry["K"] = intrinsics[cam]["K"].tolist()
                entry["dist"] = intrinsics[cam]["dist"].flatten().tolist()
                entry["rms_error_px"] = round(intrinsics[cam]["rms"], 6)
            else:
                entry["error"] = "intrinsic calibration failed"

            if cam in extrinsics:
                entry["extrinsics"] = {
                    "reference": "cam1",
                    "R": extrinsics[cam]["R"],
                    "T": extrinsics[cam]["T"],
                    "stereo_rms_px": extrinsics[cam]["stereo_rms_px"],
                    "baseline_m": extrinsics[cam]["baseline_m"],
                }
            cam_data[f"cam{cam}"] = entry

        combined = {
            "checkerboard": {
                "inner_corners": list(BOARD_SIZE),
                "square_size_m": SQUARE_SIZE_M,
                "board_squares": [board_cols, board_rows],
            },
            "source_session": SESSION,
            "cameras": cam_data,
        }
        combined_path = os.path.join(OUTPUT, "calibration_all_cameras.json")
        with open(combined_path, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"  -> {os.path.basename(combined_path)}")

        config = {
            "inner_corners": list(BOARD_SIZE),
            "square_size_m": SQUARE_SIZE_M,
            "board_squares": [board_cols, board_rows],
        }
        config_path = os.path.join(OUTPUT, "checkerboard_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"  -> {os.path.basename(config_path)}")

        # Frame extraction log — which source frames had detections
        extraction_log = {}
        for cam, info in cam_scan_info.items():
            extraction_log[f"cam{cam}"] = {
                "source_fps": round(info["source_fps"], 4),
                "total_source_frames": info["total_frames"],
                "every_n": info["every_n"],
                "frames_scanned": info["scanned"],
                "frames_detected": info["detected"],
                "detected_frame_numbers": info["frame_numbers"],
                "elapsed_sec": info["elapsed_sec"],
            }
        log_path = os.path.join(OUTPUT, "frame_extraction_log.json")
        with open(log_path, "w") as f:
            json.dump(extraction_log, f, indent=2)
        print(f"  -> {os.path.basename(log_path)}")

        # ── Summary ───────────────────────────────────────────────
        print(f"\n{'=' * 70}")
        print("CALIBRATION COMPLETE")
        print("=" * 70)
        print(f"  Board: {board_cols}x{board_rows} squares, "
              f"{BOARD_SIZE[0]}x{BOARD_SIZE[1]} inner corners, "
              f"{SQUARE_SIZE_M*1000:.0f}mm/square")

        print(f"\n  Intrinsics:")
        for cam in range(1, NUM_CAMS + 1):
            if intrinsics[cam] is None:
                print(f"    Cam {cam}: FAILED")
            else:
                K = intrinsics[cam]["K"]
                n = len(intrinsics[cam]["frame_names"])
                print(f"    Cam {cam}: RMS={intrinsics[cam]['rms']:.4f}px  "
                      f"fx={K[0,0]:.1f} fy={K[1,1]:.1f}  ({n} frames)")

        print(f"\n  Extrinsics (vs Cam 1):")
        for cam in range(2, NUM_CAMS + 1):
            if cam in extrinsics:
                e = extrinsics[cam]
                print(f"    Pair 1-{cam}: RMS={e['stereo_rms_px']:.4f}px  "
                      f"baseline={e['baseline_m']*100:.2f}cm  "
                      f"({e['shared_frames']} shared)")
            else:
                print(f"    Pair 1-{cam}: FAILED")

        print(f"\n  Output: {OUTPUT}/")
        print("  Done!")

    finally:
        print(f"\n  Cleaning up temp dir...")
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
