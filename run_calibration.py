#!/usr/bin/env python3
"""
GoPro Hero 10 Intrinsic Calibration from Checkerboard
=====================================================
Extracts frames from synced multi-camera video, auto-detects the checkerboard
size, runs cv2.calibrateCamera() per camera, and outputs calibration JSONs.

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
    WORK = os.path.join(BASE, ".calib_work")
    os.makedirs(OUTPUT, exist_ok=True)
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

    for cam in range(1, NUM_CAMS + 1):
        print(f"\n{'=' * 70}")
        print(f"Calibrating Camera {cam}")
        print("=" * 70)

        frames = sorted(glob.glob(os.path.join(WORK, f"cam{cam}_hires", "*.jpg")))
        obj_points = []
        img_points = []
        used_frames = []
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
                used_frames.append(os.path.basename(fpath))

        print(f"  Detected checkerboard in {len(used_frames)}/{len(frames)} frames")
        print(f"  Image size: {img_shape}")

        if len(obj_points) < 5:
            print(f"  WARNING: not enough detections ({len(obj_points)}), need >= 5")
            all_results[f"cam{cam}"] = {
                "error": "insufficient detections",
                "detections": len(obj_points),
            }
            continue

        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, img_shape, None, None
        )

        # Per-frame reprojection error
        reproj_errors = []
        for i in range(len(obj_points)):
            proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
            err = cv2.norm(img_points[i], proj, cv2.NORM_L2) / len(proj)
            reproj_errors.append(err)

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        print(f"\n  Results:")
        print(f"    RMS reprojection error: {ret:.4f} px")
        print(f"    Mean per-frame error:   {np.mean(reproj_errors):.4f} px")
        print(f"    Max per-frame error:    {np.max(reproj_errors):.4f} px")
        print(f"    Focal length: fx={fx:.2f}, fy={fy:.2f} px")
        print(f"    Principal point: cx={cx:.2f}, cy={cy:.2f} px")
        print(f"    Distortion: {dist.flatten()}")

        w, h = img_shape
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))

        result = {
            "camera": f"cam{cam}",
            "image_size": {"width": w, "height": h},
            "num_frames_used": len(used_frames),
            "rms_reprojection_error_px": round(ret, 6),
            "mean_reprojection_error_px": round(float(np.mean(reproj_errors)), 6),
            "max_reprojection_error_px": round(float(np.max(reproj_errors)), 6),
            "camera_matrix_K": K.tolist(),
            "distortion_coefficients": dist.flatten().tolist(),
            "new_camera_matrix": new_K.tolist(),
            "undistort_roi": list(roi),
            "focal_length_px": {"fx": round(fx, 4), "fy": round(fy, 4)},
            "principal_point_px": {"cx": round(cx, 4), "cy": round(cy, 4)},
            "used_frames": used_frames,
        }
        all_results[f"cam{cam}"] = result

        cam_path = os.path.join(OUTPUT, f"cam{cam}_intrinsics.json")
        with open(cam_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"    -> Saved {os.path.basename(cam_path)}")

    # ── Combined calibration file ─────────────────────────────────
    print(f"\n{'=' * 70}")
    print("Saving combined calibration")
    print("=" * 70)

    combined = {
        "checkerboard": {
            "inner_corners": list(BOARD_SIZE),
            "square_size_m": SQUARE_SIZE_M,
            "square_size_cm": SQUARE_SIZE_M * 100,
            "board_squares": [BOARD_SIZE[0] + 1, BOARD_SIZE[1] + 1],
            "description": (
                f"{BOARD_SIZE[0]+1}x{BOARD_SIZE[1]+1} squares, "
                f"{BOARD_SIZE[0]}x{BOARD_SIZE[1]} inner corners, "
                f"{SQUARE_SIZE_M*100:.0f}cm per square"
            ),
        },
        "source": f"{SESSION} synced footage",
        "camera_model": "GoPro Hero 10",
        "lens_mode": "Wide",
        "resolution": "3840x2160 (4K)",
        "fps": "59.94",
        "cameras": all_results,
    }

    combined_path = os.path.join(OUTPUT, "calibration_all_cameras.json")
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"  -> {os.path.basename(combined_path)}")

    # ── Checkerboard config file ──────────────────────────────────
    config = {
        "checkerboard_config": {
            "inner_corners_cols": BOARD_SIZE[0],
            "inner_corners_rows": BOARD_SIZE[1],
            "square_size_meters": SQUARE_SIZE_M,
            "square_size_cm": SQUARE_SIZE_M * 100,
            "total_squares_cols": BOARD_SIZE[0] + 1,
            "total_squares_rows": BOARD_SIZE[1] + 1,
            "board_width_m": (BOARD_SIZE[0] + 1) * SQUARE_SIZE_M,
            "board_height_m": (BOARD_SIZE[1] + 1) * SQUARE_SIZE_M,
        },
        "opencv_usage": {
            "findChessboardCorners_size": f"({BOARD_SIZE[0]}, {BOARD_SIZE[1]})",
            "object_point_scale": SQUARE_SIZE_M,
            "note": "Inner corners = squares - 1 in each dimension",
        },
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
            print(f"  Cam {cam}: RMS={r['rms_reprojection_error_px']:.4f}px, "
                  f"fx={r['focal_length_px']['fx']:.1f}, fy={r['focal_length_px']['fy']:.1f}, "
                  f"cx={r['principal_point_px']['cx']:.1f}, cy={r['principal_point_px']['cy']:.1f}, "
                  f"{r['num_frames_used']} frames")
    print(f"\nOutput: {OUTPUT}/")
    print("Done!")


if __name__ == "__main__":
    main()
