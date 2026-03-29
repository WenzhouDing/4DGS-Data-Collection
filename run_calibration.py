#!/usr/bin/env python3
"""
GoPro Hero 10 Intrinsic + Extrinsic Calibration
================================================
Extracts frames from synced multi-camera video into a temp directory,
auto-detects checkerboard, runs per-camera intrinsic calibration, then
runs pairwise stereo calibration (each camera vs cam1) for extrinsics.

Outputs:
  output/calibration/
    cam{N}_intrinsics.json       — K, dist, image_size, rms
    cam{N}_extrinsics.json       — R, T, E, F, stereo_rms (relative to cam1)
    calibration_all_cameras.json — combined intrinsics + extrinsics
    checkerboard_config.json
    validation/
      cam{N}/corners_*.jpg       — corner overlay samples
      cam{N}/reproj_error.png    — per-frame intrinsic error
      stereo/pair_1_{N}_epipolar_*.jpg — epipolar line validation
      stereo/stereo_rms.png      — cross-pair RMS chart
      rms_all_cameras.png        — intrinsic RMS chart

Usage:
    python run_calibration.py [--base DIR] [--session SESSION] [--cams N]
                              [--square-size M] [--fps-extract FPS]
"""

import argparse
import cv2
import numpy as np
import json
import os
import glob
import subprocess
import tempfile
import shutil


# ─── CLI ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Checkerboard intrinsic + extrinsic calibration")
    p.add_argument("--base", default=".",
                   help="Project root (contains output/ folder)")
    p.add_argument("--session", default="session_04",
                   help="Session to calibrate from")
    p.add_argument("--cams", type=int, default=5,
                   help="Number of cameras")
    p.add_argument("--square-size", type=float, default=0.03,
                   help="Checkerboard square side in metres (default 0.03 = 3 cm)")
    p.add_argument("--fps-extract", type=float, default=0.5,
                   help="Extraction rate in fps (default 0.5 = one frame every 2 s)")
    return p.parse_args()


# ─── BOARD CANDIDATES ────────────────────────────────────────────
BOARD_CANDIDATES = [
    (9, 6), (8, 6), (7, 5), (8, 5), (7, 4), (6, 4), (10, 7),
]

VALIDATION_SAMPLE_COUNT = 4


# ─── VISUALISATION HELPERS ────────────────────────────────────────
def draw_corner_overlay(img, corners_detected, corners_reprojected, board_size):
    """Detected corners (green) + reprojected (red circles)."""
    vis = img.copy()
    cv2.drawChessboardCorners(vis, board_size, corners_detected, True)
    for pt in corners_reprojected.reshape(-1, 2):
        cv2.circle(vis, (int(pt[0]), int(pt[1])), 8, (0, 0, 255), 2)
    return vis


def draw_epipolar_lines(img1, img2, pts1, pts2, F, num_lines=12):
    """Draw epipolar lines on img2 for points in img1, and vice versa."""
    h, w = img1.shape[:2]
    vis1 = img1.copy()
    vis2 = img2.copy()

    # Subsample points
    n = len(pts1)
    step = max(1, n // num_lines)
    indices = list(range(0, n, step))[:num_lines]

    colors = [tuple(int(c) for c in np.random.RandomState(i).randint(50, 255, 3))
              for i in range(len(indices))]

    for ci, idx in enumerate(indices):
        color = colors[ci]
        p1 = pts1[idx].reshape(1, 1, 2).astype(np.float64)
        p2 = pts2[idx].reshape(1, 1, 2).astype(np.float64)

        # Epipolar line in img2 from point in img1
        line2 = cv2.computeCorrespondEpilines(p1, 1, F).reshape(-1, 3)
        a, b, c = line2[0]
        x0, x1 = 0, w
        y0 = int(-c / b) if abs(b) > 1e-6 else 0
        y1 = int(-(c + a * w) / b) if abs(b) > 1e-6 else h
        cv2.line(vis2, (x0, y0), (x1, y1), color, 2)
        cv2.circle(vis2, (int(pts2[idx][0]), int(pts2[idx][1])), 10, color, -1)

        # Epipolar line in img1 from point in img2
        line1 = cv2.computeCorrespondEpilines(p2, 2, F).reshape(-1, 3)
        a, b, c = line1[0]
        y0 = int(-c / b) if abs(b) > 1e-6 else 0
        y1 = int(-(c + a * w) / b) if abs(b) > 1e-6 else h
        cv2.line(vis1, (x0, y0), (x1, y1), color, 2)
        cv2.circle(vis1, (int(pts1[idx][0]), int(pts1[idx][1])), 10, color, -1)

    return vis1, vis2


def make_bar_chart(values, labels, title, ylabel, output_path,
                   threshold_colors=None):
    """Generic bar chart."""
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
    """Per-frame reprojection error bar chart."""
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
    """Resize keeping aspect ratio."""
    h, w = img.shape[:2]
    scale = target_width / w
    return cv2.resize(img, (int(w * scale), int(h * scale)))


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
    STEREO_VAL = os.path.join(VALIDATION, "stereo")
    os.makedirs(OUTPUT, exist_ok=True)
    os.makedirs(VALIDATION, exist_ok=True)
    os.makedirs(STEREO_VAL, exist_ok=True)

    # ── Extract frames to temp dir ────────────────────────────────
    tmpdir = tempfile.mkdtemp(prefix="gopro_calib_")
    print("=" * 70)
    print(f"Extracting frames from {SESSION} at {FPS_EXTRACT} fps")
    print(f"Temp directory: {tmpdir}")
    print("=" * 70)

    try:
        for cam in range(1, NUM_CAMS + 1):
            vid = os.path.join(SYNCED_DIR, f"cam{cam}_synced.mp4")
            if not os.path.exists(vid):
                print(f"  WARNING: {vid} not found, skipping cam {cam}")
                continue
            out_dir = os.path.join(tmpdir, f"cam{cam}")
            os.makedirs(out_dir, exist_ok=True)
            subprocess.run([
                "ffmpeg", "-v", "quiet", "-i", vid,
                "-vf", f"fps={FPS_EXTRACT}",
                "-q:v", "2",
                os.path.join(out_dir, "frame_%04d.jpg"),
            ], check=True)
            extracted = glob.glob(os.path.join(out_dir, "*.jpg"))
            print(f"  Cam {cam}: extracted {len(extracted)} frames")

        # ── Auto-detect board size ────────────────────────────────
        print("\n" + "=" * 70)
        print("Auto-detecting checkerboard size...")
        print("=" * 70)

        detected_board = None
        test_frames = sorted(glob.glob(os.path.join(tmpdir, "cam1", "*.jpg")))

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

        # ── Object points ─────────────────────────────────────────
        objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2)
        objp *= SQUARE_SIZE_M

        # ── Detect corners in ALL cameras, keyed by frame name ────
        # This shared structure is used by both intrinsic and extrinsic phases.
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # cam_corners[cam][frame_name] = (corners_refined, frame_path)
        cam_corners = {}
        img_shape = None

        for cam in range(1, NUM_CAMS + 1):
            cam_corners[cam] = {}
            frames = sorted(glob.glob(os.path.join(tmpdir, f"cam{cam}", "*.jpg")))
            detected = 0
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
                    fname = os.path.basename(fpath)
                    cam_corners[cam][fname] = (corners_refined, fpath)
                    detected += 1

            total = len(frames)
            print(f"  Cam {cam}: detected board in {detected}/{total} frames")

        # ══════════════════════════════════════════════════════════
        # PHASE 1: INTRINSIC CALIBRATION
        # ══════════════════════════════════════════════════════════
        print(f"\n{'=' * 70}")
        print("PHASE 1: INTRINSIC CALIBRATION")
        print("=" * 70)

        intrinsics = {}   # cam -> {K, dist, ...}
        cam_rms_map = {}

        for cam in range(1, NUM_CAMS + 1):
            print(f"\n--- Cam {cam} ---")
            corners_dict = cam_corners[cam]
            if len(corners_dict) < 5:
                print(f"  WARNING: only {len(corners_dict)} detections, need >= 5")
                intrinsics[cam] = None
                continue

            frame_names = sorted(corners_dict.keys())
            obj_pts = [objp] * len(frame_names)
            img_pts = [corners_dict[fn][0] for fn in frame_names]

            ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                obj_pts, img_pts, img_shape, None, None
            )

            # Per-frame reprojection error
            reproj_errors = []
            reproj_pts = []
            for i in range(len(obj_pts)):
                proj, _ = cv2.projectPoints(
                    obj_pts[i], rvecs[i], tvecs[i], K, dist)
                err = cv2.norm(img_pts[i], proj, cv2.NORM_L2) / len(proj)
                reproj_errors.append(err)
                reproj_pts.append(proj)

            print(f"  RMS: {ret:.4f} px | fx={K[0,0]:.1f} fy={K[1,1]:.1f} "
                  f"cx={K[0,2]:.1f} cy={K[1,2]:.1f}")

            intrinsics[cam] = {
                "K": K, "dist": dist, "rms": ret,
                "rvecs": rvecs, "tvecs": tvecs,
                "frame_names": frame_names,
                "img_pts": img_pts, "reproj_pts": reproj_pts,
                "reproj_errors": reproj_errors,
            }
            cam_rms_map[f"Cam {cam}"] = ret

            # Save minimal JSON
            result_json = {
                "image_size": [img_shape[0], img_shape[1]],
                "K": K.tolist(),
                "dist": dist.flatten().tolist(),
                "rms_error_px": round(ret, 6),
            }
            cam_path = os.path.join(OUTPUT, f"cam{cam}_intrinsics.json")
            with open(cam_path, "w") as f:
                json.dump(result_json, f, indent=2)
            print(f"  -> {os.path.basename(cam_path)}")

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

            # Validation: error bar chart
            chart_path = os.path.join(cam_val_dir, "reproj_error_per_frame.png")
            make_error_bar_chart(reproj_errors, f"Cam {cam}", chart_path)

        # Intrinsic summary chart
        if cam_rms_map:
            make_bar_chart(
                list(cam_rms_map.values()), list(cam_rms_map.keys()),
                "Intrinsic Calibration RMS — All Cameras",
                "RMS Reprojection Error (px)",
                os.path.join(VALIDATION, "rms_all_cameras.png"),
            )

        # ══════════════════════════════════════════════════════════
        # PHASE 2: EXTRINSIC (STEREO) CALIBRATION
        # ══════════════════════════════════════════════════════════
        print(f"\n{'=' * 70}")
        print("PHASE 2: EXTRINSIC CALIBRATION (each cam vs Cam 1)")
        print("=" * 70)

        ref_cam = 1
        if intrinsics[ref_cam] is None:
            print("ERROR: Reference camera (cam1) intrinsic calibration failed!")
            raise SystemExit(1)

        K1 = intrinsics[ref_cam]["K"]
        dist1 = intrinsics[ref_cam]["dist"]
        extrinsics = {}
        stereo_rms_map = {}

        for cam in range(2, NUM_CAMS + 1):
            print(f"\n--- Pair: Cam 1 <-> Cam {cam} ---")

            if intrinsics[cam] is None:
                print(f"  SKIP: cam {cam} intrinsics failed")
                continue

            K2 = intrinsics[cam]["K"]
            dist2 = intrinsics[cam]["dist"]

            # Find frames where BOTH cameras detected the board
            ref_names = set(cam_corners[ref_cam].keys())
            other_names = set(cam_corners[cam].keys())
            shared = sorted(ref_names & other_names)

            print(f"  Shared frames: {len(shared)} "
                  f"(cam1: {len(ref_names)}, cam{cam}: {len(other_names)})")

            if len(shared) < 8:
                print(f"  WARNING: need >= 8 shared frames, got {len(shared)}")
                continue

            obj_pts_shared = [objp] * len(shared)
            pts1 = [cam_corners[ref_cam][fn][0] for fn in shared]
            pts2 = [cam_corners[cam][fn][0] for fn in shared]

            # stereoCalibrate with fixed intrinsics
            flags = cv2.CALIB_FIX_INTRINSIC
            stereo_criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                100, 1e-6,
            )

            ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
                obj_pts_shared, pts1, pts2,
                K1, dist1, K2, dist2,
                img_shape,
                criteria=stereo_criteria,
                flags=flags,
            )

            # Decompose T to get baseline distance
            baseline_m = np.linalg.norm(T)
            # Rotation as Rodrigues vector + Euler angles
            rvec, _ = cv2.Rodrigues(R)
            # Euler angles (rough — assumes ZYX)
            sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
            if sy > 1e-6:
                ex = np.arctan2(R[2, 1], R[2, 2])
                ey = np.arctan2(-R[2, 0], sy)
                ez = np.arctan2(R[1, 0], R[0, 0])
            else:
                ex = np.arctan2(-R[1, 2], R[1, 1])
                ey = np.arctan2(-R[2, 0], sy)
                ez = 0

            print(f"  Stereo RMS: {ret:.4f} px")
            print(f"  Baseline: {baseline_m*100:.2f} cm")
            print(f"  Translation: [{T[0,0]:.4f}, {T[1,0]:.4f}, {T[2,0]:.4f}] m")
            print(f"  Euler (deg): rx={np.degrees(ex):.2f} ry={np.degrees(ey):.2f} "
                  f"rz={np.degrees(ez):.2f}")

            stereo_rms_map[f"1-{cam}"] = ret

            ext_data = {
                "reference": "cam1",
                "target": f"cam{cam}",
                "R": R.tolist(),
                "T": T.flatten().tolist(),
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
            print(f"  -> {os.path.basename(ext_path)}")

            # ── Validation: epipolar lines on sample frames ───────
            print(f"  Generating epipolar validation...")
            # Pick 2 sample shared frames
            sample_shared = [shared[0], shared[len(shared) // 2]]
            for fn in sample_shared:
                fpath1 = cam_corners[ref_cam][fn][1]
                fpath2 = cam_corners[cam][fn][1]
                img1 = cv2.imread(fpath1)
                img2 = cv2.imread(fpath2)

                # Use detected corner points for epipolar visualisation
                c1 = cam_corners[ref_cam][fn][0].reshape(-1, 2)
                c2 = cam_corners[cam][fn][0].reshape(-1, 2)

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
                print(f"    -> stereo/pair_1_{cam}_{base}_*.jpg")

        # Stereo RMS summary chart
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
        print("Saving combined calibration")
        print("=" * 70)

        # Build minimal combined JSON
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
                "board_squares": [BOARD_SIZE[0] + 1, BOARD_SIZE[1] + 1],
            },
            "source_session": SESSION,
            "cameras": cam_data,
        }

        combined_path = os.path.join(OUTPUT, "calibration_all_cameras.json")
        with open(combined_path, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"  -> {os.path.basename(combined_path)}")

        # Checkerboard config
        config = {
            "inner_corners": list(BOARD_SIZE),
            "square_size_m": SQUARE_SIZE_M,
            "board_squares": [BOARD_SIZE[0] + 1, BOARD_SIZE[1] + 1],
        }
        config_path = os.path.join(OUTPUT, "checkerboard_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"  -> {os.path.basename(config_path)}")

        # ── Summary ───────────────────────────────────────────────
        print(f"\n{'=' * 70}")
        print("CALIBRATION SUMMARY")
        print("=" * 70)
        print(f"  Board: {BOARD_SIZE[0]+1}x{BOARD_SIZE[1]+1} squares, "
              f"{BOARD_SIZE[0]}x{BOARD_SIZE[1]} inner corners, "
              f"{SQUARE_SIZE_M*100:.0f}cm/square")
        print(f"\n  Intrinsics:")
        for cam in range(1, NUM_CAMS + 1):
            if intrinsics[cam] is None:
                print(f"    Cam {cam}: FAILED")
            else:
                K = intrinsics[cam]["K"]
                print(f"    Cam {cam}: RMS={intrinsics[cam]['rms']:.4f}px "
                      f"fx={K[0,0]:.1f} fy={K[1,1]:.1f}")
        print(f"\n  Extrinsics (vs Cam 1):")
        for cam in range(2, NUM_CAMS + 1):
            if cam in extrinsics:
                e = extrinsics[cam]
                print(f"    Cam 1 -> Cam {cam}: stereo_rms={e['stereo_rms_px']:.4f}px "
                      f"baseline={e['baseline_m']*100:.2f}cm")
            else:
                print(f"    Cam 1 -> Cam {cam}: FAILED")

        print(f"\nOutput: {OUTPUT}/")
        print("Done!")

    finally:
        # Clean up temp frames
        print(f"\nCleaning up temp dir: {tmpdir}")
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
