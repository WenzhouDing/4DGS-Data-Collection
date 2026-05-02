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
    cam{N}_extrinsics.json       — R, T, E, F, stereo_rms (vs --ref-cam)
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
    p.add_argument("--cams", type=int, default=12,
                   help="Number of cameras")
    p.add_argument("--ref-cam", type=int, default=1,
                   help="Reference camera (its frame is the world origin; "
                        "all extrinsics are expressed in it)")
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

# Wide-baseline bridging tunables. With many cameras (12+ in 2x6 grid), some
# pairs (e.g. opposite ends of the same row) rarely see the board together
# and produce a poor or failed direct stereoCalibrate. We compute every
# viable pair and chain through well-calibrated intermediates when needed.
MIN_SHARED_FOR_PAIR = 8       # min shared frames to attempt stereoCalibrate
GOOD_DIRECT_RMS_PX = 1.5      # use direct pair if rms below this; else bridge
MAX_BRIDGE_HOPS = 6           # cap bridging chain length


# ─── TRANSFORM MATH ──────────────────────────────────────────────
def compose_transforms(R1, T1, R2, T2):
    """Compose two transforms (A→B then B→C) into A→C.

    Convention: P_out = R @ P_in + T (the cv2.stereoCalibrate output convention).
    """
    return R2 @ R1, R2 @ T1 + T2


def invert_transform(R, T):
    """Inverse of (R, T): if P_out = R @ P_in + T, then P_in = R^T @ P_out - R^T @ T."""
    return R.T, -R.T @ T


def fundamental_from_KRT(K1, K2, R, T):
    """Compute F (p2.T @ F @ p1 = 0 in pixel coords) from intrinsics + extrinsics.

    R, T must be the cam1→cam2 transform.
    """
    Tx = np.array([[0, -T[2, 0], T[1, 0]],
                   [T[2, 0], 0, -T[0, 0]],
                   [-T[1, 0], T[0, 0], 0]])
    E = Tx @ R
    return np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)


def find_bridging_path(ref, target, edges, num_cams, max_hops=MAX_BRIDGE_HOPS):
    """BFS shortest hop-count path from ref to target through good edges.

    `edges` is a set of (i, j) tuples with i < j. Returns the cam list along
    the path (e.g. [3, 4, 5, 6]) or None if no path within max_hops.
    """
    if ref == target:
        return [ref]
    visited = {ref}
    queue = [(ref, [ref])]
    while queue:
        node, path = queue.pop(0)
        if len(path) - 1 >= max_hops:
            continue
        for other in range(1, num_cams + 1):
            if other in visited:
                continue
            if (min(node, other), max(node, other)) not in edges:
                continue
            new_path = path + [other]
            if other == target:
                return new_path
            visited.add(other)
            queue.append((other, new_path))
    return None


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
    REF_CAM = args.ref_cam
    SQUARE_SIZE_M = args.square_size
    EVERY_N = args.every
    MAX_FRAMES = args.max_frames

    if not (1 <= REF_CAM <= NUM_CAMS):
        print(f"ERROR: --ref-cam {REF_CAM} must be in 1..{NUM_CAMS}")
        raise SystemExit(1)

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
    print(f"  Reference cam: {REF_CAM} (extrinsics expressed in cam{REF_CAM} frame)")
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
        # Strategy for wide-baseline rigs (e.g. 12-cam 2x6 grids): compute
        # *every* viable pair and build a graph of well-calibrated edges.
        # For each non-ref cam, prefer the direct pair if its rms meets the
        # quality bar; otherwise BFS the shortest hop-count path through
        # good intermediate pairs and chain the transforms.  This recovers
        # extrinsics for opposite-end cams that don't directly share enough
        # frames (or where stereoCalibrate is poorly conditioned), at the
        # cost of one extra stereoCalibrate per pair (~1 min for 12 cams).
        step_header(3, f"Extrinsic calibration (all pairs, then chain to Cam {REF_CAM})")

        ref_cam = REF_CAM
        if intrinsics[ref_cam] is None:
            print(f"  ERROR: Cam {ref_cam} intrinsic calibration failed!")
            raise SystemExit(1)

        stereo_criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6,
        )

        # ── Step 3a: stereoCalibrate every viable pair (i<j) ─────
        all_pairs = {}        # (i, j) where i<j -> {R, T, F, rms, shared}
        n_attempted = n_succeeded = n_skipped_few = 0
        cam_ids = [c for c in range(1, NUM_CAMS + 1) if intrinsics[c] is not None]
        for ii, i in enumerate(cam_ids):
            for j in cam_ids[ii + 1:]:
                shared = sorted(set(cam_corners[i]) & set(cam_corners[j]))
                if len(shared) < MIN_SHARED_FOR_PAIR:
                    n_skipped_few += 1
                    continue
                n_attempted += 1
                obj_pts = [objp] * len(shared)
                pts_i = [cam_corners[i][fn][0] for fn in shared]
                pts_j = [cam_corners[j][fn][0] for fn in shared]
                try:
                    ret, _, _, _, _, R_ij, T_ij, _, F_ij = cv2.stereoCalibrate(
                        obj_pts, pts_i, pts_j,
                        intrinsics[i]["K"], intrinsics[i]["dist"],
                        intrinsics[j]["K"], intrinsics[j]["dist"],
                        img_shape,
                        criteria=stereo_criteria,
                        flags=cv2.CALIB_FIX_INTRINSIC,
                    )
                    all_pairs[(i, j)] = {
                        "R": R_ij, "T": T_ij, "F": F_ij,
                        "rms": ret, "shared": shared,
                    }
                    n_succeeded += 1
                    progress(f"\r  Pair {i}-{j}: rms={ret:.3f}px "
                             f"({len(shared)} shared)" + " " * 20)
                except cv2.error as e:
                    progress(f"\r  Pair {i}-{j}: FAILED ({e})" + " " * 20 + "\n", end="")
        print()
        print(f"  All-pairs: {n_succeeded}/{n_attempted} converged "
              f"({n_skipped_few} skipped for <{MIN_SHARED_FOR_PAIR} shared)")

        # ── Step 3b: graph of "good" edges for bridging ──────────
        good_edges = {(i, j) for (i, j), p in all_pairs.items()
                      if p["rms"] <= GOOD_DIRECT_RMS_PX}
        print(f"  Graph: {len(good_edges)} edges meet quality bar "
              f"(rms ≤ {GOOD_DIRECT_RMS_PX:.1f}px) for bridging")

        def pair_lookup(i, j):
            """Get (R_i_to_j, T_i_to_j, info) for any direction; None if missing."""
            key = (min(i, j), max(i, j))
            if key not in all_pairs:
                return None
            p = all_pairs[key]
            if i < j:
                return p["R"], p["T"], p
            R_inv, T_inv = invert_transform(p["R"], p["T"])
            return R_inv, T_inv, p

        # ── Step 3c: pick best path per cam, compose transforms ──
        extrinsics = {}
        stereo_rms_map = {}
        method_summary = {"direct": 0, "direct_low_quality": 0,
                          "bridged": 0, "no_path": 0}

        for cam in range(1, NUM_CAMS + 1):
            if cam == ref_cam or intrinsics[cam] is None:
                continue

            direct = pair_lookup(ref_cam, cam)
            use_direct = (direct is not None
                          and direct[2]["rms"] <= GOOD_DIRECT_RMS_PX)

            R_chain = T_chain = None
            method = path = path_rms = None
            F_for_viz = None  # only set for direct pairs we can validate visually

            if use_direct:
                R_chain, T_chain, info = direct
                F_for_viz = info["F"]
                path = [ref_cam, cam]
                path_rms = [info["rms"]]
                method = "direct"
                method_summary["direct"] += 1
            else:
                bridged = find_bridging_path(ref_cam, cam, good_edges, NUM_CAMS)
                if bridged is None:
                    if direct is not None:
                        # No bridge available; use the (low-quality) direct anyway
                        R_chain, T_chain, info = direct
                        F_for_viz = info["F"]
                        path = [ref_cam, cam]
                        path_rms = [info["rms"]]
                        method = (f"direct (low quality: rms={info['rms']:.2f}px "
                                  f"> {GOOD_DIRECT_RMS_PX}; no bridge available)")
                        method_summary["direct_low_quality"] += 1
                    else:
                        print(f"  Cam {cam}: NO PATH from cam{ref_cam} "
                              f"(no direct pair; no bridge through good edges)")
                        method_summary["no_path"] += 1
                        continue
                else:
                    R_chain, T_chain = np.eye(3), np.zeros((3, 1))
                    path = bridged
                    path_rms = []
                    for k in range(len(bridged) - 1):
                        R_k, T_k, info_k = pair_lookup(bridged[k], bridged[k + 1])
                        R_chain, T_chain = compose_transforms(
                            R_chain, T_chain, R_k, T_k)
                        path_rms.append(info_k["rms"])
                    # Derive F for the bridged ref->cam transform so epipolar
                    # validation still works downstream.
                    F_for_viz = fundamental_from_KRT(
                        intrinsics[ref_cam]["K"], intrinsics[cam]["K"],
                        R_chain, T_chain)
                    method = f"bridged via {'->'.join(map(str, bridged))}"
                    method_summary["bridged"] += 1

            # Invert to "camN's pose in ref's frame"
            R_inv, T_inv = invert_transform(R_chain, T_chain)
            baseline_m = float(np.linalg.norm(T_inv))

            # Euler decomp (R = Rx · Ry · Rz; cv2 column-vec convention).
            # Branch handles gimbal lock when ey ≈ ±π/2.
            sy = np.sqrt(R_inv[0, 0] ** 2 + R_inv[1, 0] ** 2)
            if sy > 1e-6:
                ex = np.arctan2(R_inv[2, 1], R_inv[2, 2])
                ey = np.arctan2(-R_inv[2, 0], sy)
                ez = np.arctan2(R_inv[1, 0], R_inv[0, 0])
            else:
                ex = np.arctan2(-R_inv[1, 2], R_inv[1, 1])
                ey = np.arctan2(-R_inv[2, 0], sy)
                ez = 0

            # Aggregate quality: max link RMS along the path (worst link
            # dominates). For direct pairs this just equals the direct rms.
            agg_rms = max(path_rms) if path_rms else float("nan")
            stereo_rms_map[f"{ref_cam}-{cam}"] = agg_rms

            progress(
                f"  Pair {ref_cam}-{cam}: rms={agg_rms:.3f}px  "
                f"baseline={baseline_m*100:.2f}cm  "
                f"T=[{T_inv[0,0]:.4f}, {T_inv[1,0]:.4f}, {T_inv[2,0]:.4f}]m  "
                f"[{method}]                 \n", end="")

            ext_data = {
                "reference": f"cam{ref_cam}",
                "target": f"cam{cam}",
                "R": R_inv.tolist(),
                "T": T_inv.flatten().tolist(),
                "F": F_for_viz.tolist() if F_for_viz is not None else None,
                "stereo_rms_px": round(agg_rms, 6),
                "baseline_m": round(baseline_m, 6),
                "euler_deg": {
                    "rx": round(np.degrees(ex), 4),
                    "ry": round(np.degrees(ey), 4),
                    "rz": round(np.degrees(ez), 4),
                },
                "method": method,
                "path": path,
                "path_rms": [round(r, 6) for r in path_rms],
            }
            extrinsics[cam] = ext_data

            ext_path = os.path.join(OUTPUT, f"cam{cam}_extrinsics.json")
            with open(ext_path, "w") as f:
                json.dump(ext_data, f, indent=2)

            # Epipolar validation images — only meaningful for direct pairs
            # (bridged pairs have no shared frames between ref and cam).
            if use_direct:
                shared = direct[2]["shared"]
                K1 = intrinsics[ref_cam]["K"]
                dist1 = intrinsics[ref_cam]["dist"]
                K2 = intrinsics[cam]["K"]
                dist2 = intrinsics[cam]["dist"]
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
                    vis1, vis2 = draw_epipolar_lines(img1, img2, c1, c2, F_for_viz)
                    vis1 = scale_image(vis1)
                    vis2 = scale_image(vis2)
                    base = fn.replace(".jpg", "")
                    p1 = os.path.join(STEREO_VAL,
                                      f"pair_{ref_cam}_{cam}_{base}_cam{ref_cam}.jpg")
                    p2 = os.path.join(STEREO_VAL,
                                      f"pair_{ref_cam}_{cam}_{base}_cam{cam}.jpg")
                    cv2.imwrite(p1, vis1, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    cv2.imwrite(p2, vis2, [cv2.IMWRITE_JPEG_QUALITY, 85])

        print(f"\n  Methods used: direct={method_summary['direct']}, "
              f"direct_low_quality={method_summary['direct_low_quality']}, "
              f"bridged={method_summary['bridged']}, "
              f"no_path={method_summary['no_path']}")

        if stereo_rms_map:
            make_bar_chart(
                list(stereo_rms_map.values()),
                [f"Cam {k}" for k in stereo_rms_map.keys()],
                "Stereo Calibration RMS — All Pairs (worst link if bridged)",
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
                e = extrinsics[cam]
                entry["extrinsics"] = {
                    "reference": f"cam{REF_CAM}",
                    "R": e["R"],
                    "T": e["T"],
                    "stereo_rms_px": e["stereo_rms_px"],
                    "baseline_m": e["baseline_m"],
                    "method": e["method"],
                    "path": e["path"],
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

        print(f"\n  Extrinsics (vs Cam {REF_CAM}):")
        for cam in range(1, NUM_CAMS + 1):
            if cam == REF_CAM:
                continue
            if cam in extrinsics:
                e = extrinsics[cam]
                print(f"    Pair {REF_CAM}-{cam}: RMS={e['stereo_rms_px']:.4f}px  "
                      f"baseline={e['baseline_m']*100:.2f}cm  "
                      f"[{e['method']}]")
            else:
                print(f"    Pair {REF_CAM}-{cam}: FAILED (no path)")

        print(f"\n  Output: {OUTPUT}/")
        print("  Done!")

    finally:
        print(f"\n  Cleaning up temp dir...")
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
