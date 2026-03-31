#!/usr/bin/env python3
"""
Epipolar Geometry Validation
=============================
Reads calibration output (intrinsics + extrinsics), samples frames from
synced video, detects checkerboard corners, and measures epipolar error
(point-to-line distance on undistorted images).

Outputs per-pair statistics and validation images to
  output/calibration/validation/epipolar_eval/

Usage:
    python run_eval_epipolar.py [--base DIR] [--session SESSION] [--cams N]
                                [--board COLSxROWS] [--num-frames N]
"""

import argparse
import cv2
import numpy as np
import json
import os
import sys


def parse_args():
    p = argparse.ArgumentParser(description="Epipolar geometry validation")
    p.add_argument("--base", default=".", help="Project root")
    p.add_argument("--session", default="session_01", help="Synced session")
    p.add_argument("--cams", type=int, default=5, help="Number of cameras")
    p.add_argument("--board", default="9x12",
                   help="Board size as COLSxROWS in squares")
    p.add_argument("--num-frames", type=int, default=10,
                   help="Number of frames to sample for evaluation")
    return p.parse_args()


def load_calib(calib_dir, num_cams):
    """Load intrinsics and extrinsics from calibration JSONs."""
    cams = {}
    for cam in range(1, num_cams + 1):
        intr_path = os.path.join(calib_dir, f"cam{cam}_intrinsics.json")
        if not os.path.exists(intr_path):
            continue
        with open(intr_path) as f:
            intr = json.load(f)
        K = np.array(intr["K"])
        dist = np.array(intr["dist"])
        cams[cam] = {"K": K, "dist": dist}

        ext_path = os.path.join(calib_dir, f"cam{cam}_extrinsics.json")
        if os.path.exists(ext_path):
            with open(ext_path) as f:
                ext = json.load(f)
            # Saved R, T are camN's pose in cam1's frame.
            # Invert to get cam1→camN transform for computing F.
            R_saved = np.array(ext["R"])
            T_saved = np.array(ext["T"]).reshape(3, 1)
            R_orig = R_saved.T
            T_orig = -R_saved.T @ T_saved
            cams[cam]["R_orig"] = R_orig
            cams[cam]["T_orig"] = T_orig
            cams[cam]["F"] = np.array(ext["F"])
    return cams


def compute_F(K1, K2, R, T):
    """Compute fundamental matrix from intrinsics and extrinsics.
    R, T: cam1→cam2 transform (OpenCV convention).
    """
    Tx = np.array([
        [0, -T[2, 0], T[1, 0]],
        [T[2, 0], 0, -T[0, 0]],
        [-T[1, 0], T[0, 0], 0],
    ])
    E = Tx @ R
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    return F


def epipolar_distance(pts, lines):
    """Distance from each point to its corresponding epipolar line.
    pts:   (N, 2)
    lines: (N, 3) — ax + by + c = 0
    """
    a, b, c = lines[:, 0], lines[:, 1], lines[:, 2]
    x, y = pts[:, 0], pts[:, 1]
    return np.abs(a * x + b * y + c) / np.sqrt(a**2 + b**2)


def detect_board(gray, board_size):
    """Detect checkerboard. Returns refined corners or None."""
    w = gray.shape[1]
    if w > 960:
        scale = 960.0 / w
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


def draw_epipolar_eval(img1, img2, pts1, pts2, F, dists1, dists2,
                       num_lines=15):
    """Draw epipolar lines with error annotations on undistorted images."""
    h, w = img1.shape[:2]
    vis1 = img1.copy()
    vis2 = img2.copy()

    n = len(pts1)
    step = max(1, n // num_lines)
    indices = list(range(0, n, step))[:num_lines]

    for ci, idx in enumerate(indices):
        # Color: green if < 1px, yellow if < 2px, red otherwise
        d = max(dists1[idx], dists2[idx])
        if d < 1.0:
            color = (0, 200, 0)
        elif d < 2.0:
            color = (0, 200, 200)
        else:
            color = (0, 0, 220)

        p1 = pts1[idx].reshape(1, 1, 2).astype(np.float64)
        p2 = pts2[idx].reshape(1, 1, 2).astype(np.float64)

        # Epipolar line in img2 from point in img1
        line2 = cv2.computeCorrespondEpilines(p1, 1, F).reshape(-1, 3)
        a, b, c = line2[0]
        x0, x1_pt = 0, w
        y0 = int(-c / b) if abs(b) > 1e-6 else 0
        y1 = int(-(c + a * w) / b) if abs(b) > 1e-6 else h
        cv2.line(vis2, (x0, y0), (x1_pt, y1), color, 2)
        cv2.circle(vis2, (int(pts2[idx][0]), int(pts2[idx][1])), 8,
                   color, -1)

        # Epipolar line in img1 from point in img2
        line1 = cv2.computeCorrespondEpilines(p2, 2, F).reshape(-1, 3)
        a, b, c = line1[0]
        y0 = int(-c / b) if abs(b) > 1e-6 else 0
        y1 = int(-(c + a * w) / b) if abs(b) > 1e-6 else h
        cv2.line(vis1, (x0, y0), (x1_pt, y1), color, 2)
        cv2.circle(vis1, (int(pts1[idx][0]), int(pts1[idx][1])), 8,
                   color, -1)

    return vis1, vis2


def main():
    args = parse_args()

    BASE = os.path.abspath(args.base)
    SESSION = args.session
    NUM_CAMS = args.cams
    NUM_FRAMES = args.num_frames

    parts = args.board.lower().split("x")
    board_cols, board_rows = int(parts[0]), int(parts[1])
    BOARD_SIZE = (board_cols - 1, board_rows - 1)

    CALIB_DIR = os.path.join(BASE, "output", "calibration")
    SYNCED_DIR = os.path.join(BASE, "output", SESSION, "synced_raw")
    OUT_DIR = os.path.join(CALIB_DIR, "validation", "epipolar_eval")
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("EPIPOLAR GEOMETRY VALIDATION")
    print("=" * 70)
    print(f"  Session:     {SESSION}")
    print(f"  Cameras:     {NUM_CAMS}")
    print(f"  Board:       {board_cols}x{board_rows} -> "
          f"{BOARD_SIZE[0]}x{BOARD_SIZE[1]} inner corners")
    print(f"  Frames:      {NUM_FRAMES}")

    # Load calibration
    cams = load_calib(CALIB_DIR, NUM_CAMS)
    print(f"  Loaded:      {len(cams)} cameras")

    ref_cam = 1
    K1 = cams[ref_cam]["K"]
    dist1 = cams[ref_cam]["dist"]

    # Open synced videos
    caps = {}
    for cam in range(1, NUM_CAMS + 1):
        vid = os.path.join(SYNCED_DIR, f"cam{cam}_synced.mp4")
        if os.path.exists(vid):
            caps[cam] = cv2.VideoCapture(vid)

    total_frames = int(list(caps.values())[0].get(cv2.CAP_PROP_FRAME_COUNT))

    # Sample frame indices evenly across the video
    sample_indices = np.linspace(
        total_frames * 0.1, total_frames * 0.9, NUM_FRAMES, dtype=int)

    print(f"  Total video: {total_frames} frames")
    print(f"  Sampling:    {len(sample_indices)} frames")
    print()

    # Evaluate each pair
    pair_results = {}
    for cam in range(2, NUM_CAMS + 1):
        if cam not in cams or "R_orig" not in cams[cam]:
            continue

        K2 = cams[cam]["K"]
        dist2 = cams[cam]["dist"]

        # Recompute F from our calibration for consistency
        F = compute_F(K1, K2, cams[cam]["R_orig"], cams[cam]["T_orig"])

        REJECT_THRESH = 10.0  # reject frame if any point > this (px)

        all_dists = []
        frames_evaluated = 0
        frames_rejected = 0
        vis_saved = 0

        for fi, target_frame in enumerate(sample_indices):
            # Seek both captures to the same frame
            caps[ref_cam].set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            caps[cam].set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret1, img1 = caps[ref_cam].read()
            ret2, img2 = caps[cam].read()
            if not ret1 or not ret2:
                continue

            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            corners1 = detect_board(gray1, BOARD_SIZE)
            corners2 = detect_board(gray2, BOARD_SIZE)
            if corners1 is None or corners2 is None:
                continue

            # Undistort corner points (keep in pixel coords)
            pts1 = cv2.undistortPoints(
                corners1.reshape(-1, 1, 2), K1, dist1, P=K1
            ).reshape(-1, 2)
            pts2 = cv2.undistortPoints(
                corners2.reshape(-1, 1, 2), K2, dist2, P=K2
            ).reshape(-1, 2)

            # Epipolar lines and distances
            lines2 = cv2.computeCorrespondEpilines(
                pts1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
            lines1 = cv2.computeCorrespondEpilines(
                pts2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)

            d2 = epipolar_distance(pts2, lines2)
            d1 = epipolar_distance(pts1, lines1)
            frame_max = max(d1.max(), d2.max())

            # Reject frames with bad detections
            if frame_max > REJECT_THRESH:
                frames_rejected += 1
                sys.stdout.write(
                    f"\r  Pair 1-{cam}: frame {target_frame} REJECTED "
                    f"(max err {frame_max:.1f}px)          \n")
                sys.stdout.flush()
                continue

            frame_dists = np.concatenate([d1, d2])
            all_dists.append(frame_dists)
            frames_evaluated += 1

            # Save first 3 validation images from good frames
            if vis_saved < 3:
                img1_u = cv2.undistort(img1, K1, dist1)
                img2_u = cv2.undistort(img2, K2, dist2)
                v1, v2 = draw_epipolar_eval(
                    img1_u, img2_u, pts1, pts2, F, d1, d2)

                scale = 1200 / v1.shape[1]
                v1 = cv2.resize(v1, None, fx=scale, fy=scale)
                v2 = cv2.resize(v2, None, fx=scale, fy=scale)

                tag = f"pair_1_{cam}_f{target_frame}"
                cv2.imwrite(os.path.join(OUT_DIR, f"{tag}_cam1.jpg"),
                            v1, [cv2.IMWRITE_JPEG_QUALITY, 90])
                cv2.imwrite(os.path.join(OUT_DIR, f"{tag}_cam{cam}.jpg"),
                            v2, [cv2.IMWRITE_JPEG_QUALITY, 90])
                vis_saved += 1

            sys.stdout.write(
                f"\r  Pair 1-{cam}: {frames_evaluated} good, "
                f"{frames_rejected} rejected / "
                f"{fi+1} checked")
            sys.stdout.flush()

        if all_dists:
            all_dists = np.concatenate(all_dists)
            stats = {
                "mean_px": round(float(np.mean(all_dists)), 4),
                "median_px": round(float(np.median(all_dists)), 4),
                "p95_px": round(float(np.percentile(all_dists, 95)), 4),
                "max_px": round(float(np.max(all_dists)), 4),
                "frames_good": frames_evaluated,
                "frames_rejected": frames_rejected,
                "total_point_pairs": len(all_dists) // 2,
            }
            pair_results[f"1-{cam}"] = stats
            print(f"\r  Pair 1-{cam}: mean={stats['mean_px']:.3f}px  "
                  f"median={stats['median_px']:.3f}px  "
                  f"p95={stats['p95_px']:.3f}px  "
                  f"max={stats['max_px']:.3f}px  "
                  f"({frames_evaluated} good, {frames_rejected} rejected)"
                  f"          ")
        else:
            print(f"\r  Pair 1-{cam}: no valid frames found "
                  f"({frames_rejected} rejected)")

    # Release captures
    for c in caps:
        caps[c].release()

    # Save summary JSON
    summary = {
        "session": SESSION,
        "num_sample_frames": NUM_FRAMES,
        "board": f"{board_cols}x{board_rows}",
        "pairs": pair_results,
    }
    summary_path = os.path.join(OUT_DIR, "epipolar_eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'Pair':<10} {'Mean':>8} {'Median':>8} {'P95':>8} {'Max':>8}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for pair, s in pair_results.items():
        print(f"  {pair:<10} {s['mean_px']:>7.3f}px {s['median_px']:>7.3f}px "
              f"{s['p95_px']:>7.3f}px {s['max_px']:>7.3f}px")

    all_means = [s["mean_px"] for s in pair_results.values()]
    if all_means:
        overall = np.mean(all_means)
        verdict = "GOOD" if overall < 1.0 else "OK" if overall < 2.0 else "POOR"
        print(f"\n  Overall mean: {overall:.3f}px — {verdict}")
        print(f"  (< 1px = good, < 2px = acceptable, > 2px = poor)")

    print(f"\n  Output: {OUT_DIR}/")
    print("  Done!")


if __name__ == "__main__":
    main()
