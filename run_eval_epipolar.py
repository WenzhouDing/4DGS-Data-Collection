#!/usr/bin/env python3
"""
Epipolar Geometry Validation
=============================
Reads calibration output (intrinsics + extrinsics), samples frames from
synced video, detects checkerboard corners, and measures epipolar error
(point-to-line distance on undistorted images).

Outputs per-pair statistics and validation images to
  output/<episode>/calibration/validation/epipolar_eval/

Usage:
    python run_eval_epipolar.py [--base DIR] [--episode EPISODE] [--cams N]
                                [--board COLSxROWS] [--num-frames N]
"""

import argparse
import cv2
import numpy as np
import json
import os
import sys
from collections import Counter

from calibration_geometry import (
    distortion_diagnostics,
    fundamental_from_KRT,
    undistort_points_checked,
    world_to_camera,
)
from calibration_frame_offsets import load_frame_offsets, normalize_frame_offsets


GOOD_MEAN_PX = 1.0
ACCEPTABLE_MEAN_PX = 2.0
HIGH_ERROR_PX = 10.0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Epipolar geometry validation")
    p.add_argument("--base", default=".", help="Project root")
    p.add_argument("--episode", default="episode_0001", help="Synced episode")
    p.add_argument("--calibration-dir",
                   help="Read calibration from this directory instead of episode/calibration")
    p.add_argument("--cams", type=int, default=12, help="Number of cameras")
    p.add_argument("--ref-cam", type=int, default=None,
                   help="Reference camera (default: infer from saved calibration)")
    p.add_argument("--board", default="9x12",
                   help="Board size as COLSxROWS in squares")
    p.add_argument("--num-frames", type=int, default=10,
                   help="Number of frames to sample for evaluation")
    p.add_argument("--frame-indices", type=int, nargs="+",
                   help="Explicit nonnegative, unique, ascending logical frame indices; "
                        "overrides --num-frames")
    p.add_argument("--frame-offsets",
                   help="JSON camN->integer offsets; source frame = logical frame + offset. "
                        "Otherwise use calibration offsets only for its source episode")
    p.add_argument("--out", help="Directory for the evaluation report and images")
    p.add_argument("--allow-invalid-intrinsics", action="store_true",
                   help="Measure valid points for diagnosis despite invalid lens models; "
                        "the overall result remains FAILED")
    return p.parse_args(argv)


def infer_reference(calib_dir, num_cams, requested=None):
    """Require one saved reference; never silently reinterpret camera poses."""
    references = set()
    for cam in range(1, num_cams + 1):
        path = os.path.join(calib_dir, f"cam{cam}_extrinsics.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            reference = json.load(f).get("reference")
        if not isinstance(reference, str) or not reference.startswith("cam"):
            raise ValueError(f"{path}: missing or invalid reference camera")
        try:
            references.add(int(reference[3:]))
        except ValueError as exc:
            raise ValueError(f"{path}: invalid reference {reference!r}") from exc
    if len(references) > 1:
        raise ValueError(f"Conflicting calibration references: {sorted(references)}")
    saved = next(iter(references), None)
    if saved is None and requested is None:
        raise ValueError("Cannot infer reference: no extrinsics files; supply --ref-cam")
    if saved is not None and requested is not None and requested != saved:
        raise ValueError(f"--ref-cam {requested} does not match saved reference cam{saved}")
    reference = saved if saved is not None else requested
    if not 1 <= reference <= num_cams:
        raise ValueError(f"Reference cam{reference} must be in 1..{num_cams}")
    return reference


def load_calib(calib_dir, num_cams, reference=None):
    """Load OpenCV world-to-camera poses, accepting explicitly identified legacy poses.

    Missing convention metadata is legacy camera-to-world. Saved F is deliberately
    ignored: derive it from the normalized pose and the two current intrinsics.
    Camera-local errors remain available for complete per-pair reporting.
    """
    cams = {}
    for cam in range(1, num_cams + 1):
        entry = {"errors": []}
        cams[cam] = entry
        intr_path = os.path.join(calib_dir, f"cam{cam}_intrinsics.json")
        if not os.path.exists(intr_path):
            entry["errors"].append(f"Missing intrinsics: {intr_path}")
            continue
        try:
            with open(intr_path) as f:
                intr = json.load(f)
            entry["K"] = np.asarray(intr["K"], dtype=float)
            entry["dist"] = np.asarray(intr["dist"], dtype=float)
            entry["image_size"] = tuple(intr["image_size"])
            entry["lens_diagnostics"] = distortion_diagnostics(
                entry["K"], entry["dist"], entry["image_size"])
        except (ValueError, KeyError, TypeError, OSError, cv2.error) as exc:
            entry["errors"].append(f"Invalid intrinsics for cam{cam}: {exc}")
            entry["lens_diagnostics"] = {"valid": False, "reasons": [str(exc)]}
            continue

        ext_path = os.path.join(calib_dir, f"cam{cam}_extrinsics.json")
        if os.path.exists(ext_path):
            try:
                with open(ext_path) as f:
                    ext = json.load(f)
                expected = f"cam{reference}" if reference is not None else None
                if ext.get("target", f"cam{cam}") != f"cam{cam}":
                    raise ValueError("Extrinsics target does not match its camera filename")
                entry["R"], entry["t"] = world_to_camera(ext, expected_reference=expected)
                if cam == reference and (
                        not np.allclose(entry["R"], np.eye(3), atol=1e-10, rtol=0)
                        or not np.allclose(entry["t"], 0, atol=1e-10, rtol=0)):
                    raise ValueError("Reference camera must have identity rotation and zero translation")
            except (ValueError, KeyError, TypeError, OSError) as exc:
                entry["errors"].append(f"Invalid extrinsics for cam{cam}: {exc}")
    return cams


def resolve_frame_offsets(path, calib_dir, episode, num_cams):
    if path is not None:
        return load_frame_offsets(path, num_cams), {"kind": "file", "path": os.path.abspath(path)}
    combined_path = os.path.join(calib_dir, "calibration_all_cameras.json")
    if os.path.exists(combined_path):
        with open(combined_path) as stream:
            combined = json.load(stream)
        if not isinstance(combined, dict):
            raise ValueError("Combined calibration must be a JSON object")
        if combined.get("source_episode") == episode and "source_frame_offsets" in combined:
            return normalize_frame_offsets(combined["source_frame_offsets"], num_cams), {
                "kind": "calibration", "path": combined_path, "source_episode": episode,
            }
    return normalize_frame_offsets({}, num_cams), {"kind": "zero_default"}


def compute_F(K1, K2, R, T):
    """Compute fundamental matrix from intrinsics and extrinsics.
    R, T: ref→target transform (OpenCV convention).
    """
    return fundamental_from_KRT(K1, K2, R, T)


def epipolar_distance(pts, lines):
    """Distance from each point to its corresponding epipolar line.
    pts:   (N, 2)
    lines: (N, 3) — ax + by + c = 0
    """
    a, b, c = lines[:, 0], lines[:, 1], lines[:, 2]
    x, y = pts[:, 0], pts[:, 1]
    denominator = np.hypot(a, b)
    distances = np.full(len(pts), np.nan)
    np.divide(np.abs(a * x + b * y + c), denominator, out=distances,
              where=denominator > 1e-12)
    return distances


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


def draw_epipolar_line(image, line, color):
    """Clip a line to the image, including exactly vertical epipolar lines."""
    a, b, c = line
    height, width = image.shape[:2]
    intersections = []
    if abs(b) > 1e-12:
        for x in (0, width - 1):
            y = -(c + a*x) / b
            if 0 <= y <= height - 1:
                intersections.append((x, int(round(y))))
    if abs(a) > 1e-12:
        for y in (0, height - 1):
            x = -(c + b*y) / a
            if 0 <= x <= width - 1:
                intersections.append((int(round(x)), y))
    if len(intersections) >= 2:
        cv2.line(image, intersections[0], intersections[-1], color, 2)


def draw_epipolar_eval(img1, img2, pts1, pts2, F, dists1, dists2,
                       num_lines=15):
    """Draw epipolar lines with error annotations on undistorted images."""
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
        draw_epipolar_line(vis2, line2[0], color)
        cv2.circle(vis2, (int(pts2[idx][0]), int(pts2[idx][1])), 8,
                   color, -1)

        # Epipolar line in img1 from point in img2
        line1 = cv2.computeCorrespondEpilines(p2, 2, F).reshape(-1, 3)
        draw_epipolar_line(vis1, line1[0], color)
        cv2.circle(vis1, (int(pts1[idx][0]), int(pts1[idx][1])), 8,
                   color, -1)

    return vis1, vis2


def new_pair_record(num_samples):
    return {
        "status": "INCOMPLETE", "reasons": [],
        "frames_requested": num_samples, "frames_sampled": 0,
        "frames_read_failed": 0, "frames_without_detections": 0,
        "frames_evaluated": 0, "frames_high_error": 0,
        "frames_invalid_points": 0, "invalid_point_pairs": 0,
        "total_point_pairs": 0,
        "mean_px": None, "median_px": None, "p95_px": None, "max_px": None,
    }


def json_safe(value):
    """Reports use JSON null, never nonstandard NaN/Infinity literals."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def finish_report(summary, out_dir):
    records = list(summary["pairs"].values())
    statuses = Counter(r["status"] for r in records)
    summary["pair_status_counts"] = dict(statuses)
    summary["expected_pair_count"] = len(records)
    measured = [r for r in records if r["total_point_pairs"] > 0]
    summary["evaluated_pair_count"] = len(measured)
    if summary["errors"] or summary["invalid_cameras"] or statuses["FAILED"]:
        verdict = "FAILED"
    elif not records or statuses["INCOMPLETE"]:
        verdict = "INCOMPLETE"
    elif statuses["POOR"]:
        verdict = "POOR"
    elif statuses["OK"]:
        verdict = "OK"
    else:
        verdict = "GOOD"
    summary["verdict"] = verdict
    summary["overall_mean_px"] = (
        sum(r["mean_px"] * r["total_point_pairs"] for r in measured)
        / sum(r["total_point_pairs"] for r in measured) if measured else None)
    summary["worst_pair_mean_px"] = max(
        (r["mean_px"] for r in measured), default=None)
    summary["exit_code"] = 0 if verdict in {"GOOD", "OK"} else 1 if verdict == "POOR" else 2
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "epipolar_eval_summary.json")
    with open(report_path, "w") as f:
        json.dump(json_safe(summary), f, indent=2, allow_nan=False)
    print("\nSUMMARY")
    for pair, record in summary["pairs"].items():
        metric = (f"mean={record['mean_px']:.3f}px, max={record['max_px']:.3f}px"
                  if record["mean_px"] is not None else "no measurements")
        print(f"  Pair {pair}: {record['status']} — {metric}; "
              f"{record['frames_evaluated']} evaluated, "
              f"{record['frames_high_error']} high-error frames")
        for reason in record["reasons"]:
            print(f"    {reason}")
    for error in summary["errors"]:
        print(f"  ERROR: {error}")
    print(f"  Coverage: {len(measured)}/{len(records)} expected pairs measured")
    print(f"  Verdict: {verdict}")
    print("  Thresholds per pair: GOOD mean < 1px, OK mean < 2px; "
          "any residual > 10px makes the pair POOR.")
    print("  All finite residuals are included; invalid or untested pairs prevent GOOD/OK.")
    print(f"  Report: {report_path}")
    return summary["exit_code"]


def evaluate_pair(ref_cam, cam, cams, caps, sample_indices, board_size, out_dir, record,
                  frame_offsets=None):
    """Measure every finite correspondence, retaining large residuals."""
    first, second = cams[ref_cam], cams[cam]
    frame_offsets = frame_offsets or {}
    K1, dist1 = first["K"], first["dist"]
    K2, dist2 = second["K"], second["dist"]
    invalid_lens = not (first["lens_diagnostics"]["valid"]
                        and second["lens_diagnostics"]["valid"])
    F = compute_F(K1, K2, second["R"], second["t"])
    if not np.all(np.isfinite(F)) or np.linalg.norm(F) < 1e-15:
        record["status"] = "FAILED"
        record["reasons"].append("Degenerate or nonfinite fundamental matrix")
        return
    all_dists = []
    vis_saved = 0
    failed_geometry = False
    for target_frame in sample_indices:
        record["frames_sampled"] += 1
        # Seeking compressed videos can still introduce frame-index uncertainty;
        # these checks measure supplied correspondences, not synchronization.
        caps[ref_cam].set(cv2.CAP_PROP_POS_FRAMES,
                          int(target_frame) + frame_offsets.get(ref_cam, 0))
        caps[cam].set(cv2.CAP_PROP_POS_FRAMES,
                      int(target_frame) + frame_offsets.get(cam, 0))
        ret1, img1 = caps[ref_cam].read()
        ret2, img2 = caps[cam].read()
        if not ret1 or not ret2:
            record["frames_read_failed"] += 1
            continue
        if (tuple(img1.shape[1::-1]) != first["image_size"]
                or tuple(img2.shape[1::-1]) != second["image_size"]):
            record["reasons"].append("Video dimensions do not match calibrated image_size")
            failed_geometry = True
            break
        corners1 = detect_board(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), board_size)
        corners2 = detect_board(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), board_size)
        if corners1 is None or corners2 is None:
            record["frames_without_detections"] += 1
            continue
        if len(corners1) != len(corners2) or not len(corners1):
            record["reasons"].append("Detected corner counts do not match")
            failed_geometry = True
            continue
        pts1, valid1 = undistort_points_checked(corners1, K1, dist1)
        pts2, valid2 = undistort_points_checked(corners2, K2, dist2)
        pts1, pts2 = np.asarray(pts1).reshape(-1, 2), np.asarray(pts2).reshape(-1, 2)
        valid = (np.asarray(valid1).reshape(-1) & np.asarray(valid2).reshape(-1)
                 & np.isfinite(pts1).all(axis=1) & np.isfinite(pts2).all(axis=1))
        invalid_count = int(len(valid) - np.count_nonzero(valid))
        pts1, pts2 = pts1[valid], pts2[valid]
        if len(pts1):
            lines2 = cv2.computeCorrespondEpilines(
                pts1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
            lines1 = cv2.computeCorrespondEpilines(
                pts2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
            d2, d1 = epipolar_distance(pts2, lines2), epipolar_distance(pts1, lines1)
            finite = np.isfinite(d1) & np.isfinite(d2)
            invalid_count += int(len(finite) - np.count_nonzero(finite))
            pts1, pts2, d1, d2 = pts1[finite], pts2[finite], d1[finite], d2[finite]
        if invalid_count:
            record["frames_invalid_points"] += 1
            record["invalid_point_pairs"] += invalid_count
        if not len(pts1):
            continue
        frame_dists = np.concatenate([d1, d2])
        all_dists.append(frame_dists)
        record["frames_evaluated"] += 1
        if np.max(frame_dists) > HIGH_ERROR_PX:
            record["frames_high_error"] += 1
            print(f"  Pair {ref_cam}-{cam}: frame {target_frame} high error "
                  f"{np.max(frame_dists):.1f}px (included in statistics)")
        # Keep overlays from measured frames, including high-error examples.
        # An invalid global lens map cannot produce a trustworthy image overlay.
        if vis_saved < 3 and not invalid_lens:
            img1_u, img2_u = cv2.undistort(img1, K1, dist1), cv2.undistort(img2, K2, dist2)
            v1, v2 = draw_epipolar_eval(img1_u, img2_u, pts1, pts2, F, d1, d2)
            for camera, visual in ((ref_cam, v1), (cam, v2)):
                scale = 1200 / visual.shape[1]
                visual = cv2.resize(visual, None, fx=scale, fy=scale)
                tag = f"pair_{ref_cam}_{cam}_f{target_frame}_cam{camera}.jpg"
                cv2.imwrite(os.path.join(out_dir, tag), visual,
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
            vis_saved += 1
    if all_dists:
        distances = np.concatenate(all_dists)
        record.update({
            "mean_px": float(np.mean(distances)),
            "median_px": float(np.median(distances)),
            "p95_px": float(np.percentile(distances, 95)),
            "max_px": float(np.max(distances)),
            "total_point_pairs": len(distances) // 2,
        })
    if invalid_lens:
        record["reasons"].append("Invalid lens model; diagnostic measurements cannot validate calibration")
    if record["invalid_point_pairs"]:
        record["reasons"].append("Some points failed undistortion round-trip or epipolar-line checks")
    if invalid_lens or failed_geometry or record["invalid_point_pairs"]:
        record["status"] = "FAILED"
    elif not all_dists:
        record["status"] = "INCOMPLETE"
        record["reasons"].append("No measurable checkerboard correspondences")
    elif record["frames_read_failed"]:
        record["status"] = "INCOMPLETE"
        record["reasons"].append("Some sampled frames could not be read")
    elif record["mean_px"] >= ACCEPTABLE_MEAN_PX or record["frames_high_error"]:
        record["status"] = "POOR"
    elif record["mean_px"] >= GOOD_MEAN_PX:
        record["status"] = "OK"
    else:
        record["status"] = "GOOD"


def main(argv=None):
    args = parse_args(argv)
    calib_dir = (os.path.abspath(args.calibration_dir) if args.calibration_dir else
                 os.path.join(os.path.abspath(args.base), "output", args.episode, "calibration"))
    synced_dir = os.path.join(os.path.abspath(args.base), "output", args.episode, "synced_raw")
    out_dir = os.path.abspath(args.out) if args.out else os.path.join(calib_dir, "validation", "epipolar_eval")
    requested_count = len(args.frame_indices) if args.frame_indices is not None else args.num_frames
    summary = {
        "schema_version": 2, "episode": args.episode, "num_cameras": args.cams,
        "calibration_dir": calib_dir,
        "num_sample_frames": requested_count, "requested_sample_count": requested_count,
        "requested_frame_indices": args.frame_indices,
        "sample_frame_indices": list(args.frame_indices) if args.frame_indices is not None else [],
        "sampling_method": "explicit" if args.frame_indices is not None else "evenly_spaced",
        "board": args.board,
        "reference": None, "requested_ref_cam": args.ref_cam,
        "extrinsics_convention": "world_to_camera",
        "diagnostic_mode": args.allow_invalid_intrinsics,
        "thresholds": {"good_mean_px_below": GOOD_MEAN_PX,
                       "acceptable_mean_px_below": ACCEPTABLE_MEAN_PX,
                       "high_error_px_above": HIGH_ERROR_PX},
        "statistics_policy": "All finite residuals; no high-error rejection. Worst pair determines quality.",
        "lens_diagnostics": {}, "invalid_cameras": [], "errors": [], "pairs": {},
    }
    try:
        if args.cams < 2 or requested_count < 1:
            raise ValueError("--cams must be at least 2 and the requested frame count must be positive")
        if args.frame_indices is not None and (
                any(index < 0 for index in args.frame_indices)
                or any(first >= second for first, second in
                       zip(args.frame_indices, args.frame_indices[1:]))):
            raise ValueError("--frame-indices must be nonnegative, unique, and ascending")
        parts = args.board.lower().split("x")
        if len(parts) != 2:
            raise ValueError("--board must be COLSxROWS in squares")
        cols, rows = (int(part) for part in parts)
        if min(cols, rows) < 3:
            raise ValueError("--board needs at least 3 squares per dimension")
        ref_cam = infer_reference(calib_dir, args.cams, args.ref_cam)
        summary["reference"] = f"cam{ref_cam}"
        summary["pairs"] = {f"{ref_cam}-{cam}": new_pair_record(requested_count)
                            for cam in range(1, args.cams + 1) if cam != ref_cam}
        cams = load_calib(calib_dir, args.cams, ref_cam)
        frame_offsets, offset_source = resolve_frame_offsets(
            args.frame_offsets, calib_dir, args.episode, args.cams)
        summary["frame_offsets"] = {f"cam{cam}": offset for cam, offset in frame_offsets.items()}
        summary["frame_offsets_source"] = offset_source
    except (ValueError, KeyError, TypeError, OSError) as exc:
        summary["errors"].append(str(exc))
        return finish_report(summary, out_dir)
    for cam, entry in cams.items():
        if "lens_diagnostics" in entry:
            diagnostics = entry["lens_diagnostics"]
            summary["lens_diagnostics"][f"cam{cam}"] = diagnostics
            if not diagnostics["valid"]:
                summary["invalid_cameras"].append(f"cam{cam}")
    print(f"EPIPOLAR GEOMETRY VALIDATION — {args.episode}, reference cam{ref_cam}")
    if summary["invalid_cameras"] and not args.allow_invalid_intrinsics:
        summary["errors"].append(
            "Invalid intrinsics: " + ", ".join(summary["invalid_cameras"])
            + ". Use --allow-invalid-intrinsics only for diagnostic measurements.")
        for cam in range(1, args.cams + 1):
            if cam == ref_cam:
                continue
            record = summary["pairs"][f"{ref_cam}-{cam}"]
            invalid = {f"cam{ref_cam}", f"cam{cam}"} & set(summary["invalid_cameras"])
            record["status"] = "FAILED" if invalid else "INCOMPLETE"
            record["reasons"].append("Evaluation stopped because lens validation failed")
        return finish_report(summary, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    caps, video_errors, frame_counts = {}, {}, {}
    try:
        for cam in range(1, args.cams + 1):
            video = os.path.join(synced_dir, f"cam{cam}_synced.mp4")
            if not os.path.exists(video):
                video_errors[cam] = f"Missing video: {video}"
                continue
            cap = cv2.VideoCapture(video)
            if not cap.isOpened():
                cap.release()
                video_errors[cam] = f"Cannot open video: {video}"
                continue
            count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if not np.isfinite(count) or count < 1:
                cap.release()
                video_errors[cam] = f"Video has no readable frame count: {video}"
                continue
            frame_counts[cam] = int(count)
            caps[cam] = cap
        summary["source_frame_counts"] = {f"cam{cam}": count for cam, count in frame_counts.items()}
        if args.frame_indices is not None:
            sample_indices = list(args.frame_indices)
            summary["sample_frame_indices"] = sample_indices
            summary["source_frame_indices"] = {
                f"cam{cam}": [index + frame_offsets[cam] for index in sample_indices]
                for cam in range(1, args.cams + 1)}
            invalid_ranges = {
                cam: [index + frame_offsets[cam] for index in sample_indices
                      if index + frame_offsets[cam] < 0 or index + frame_offsets[cam] >= count]
                for cam, count in frame_counts.items()
                if sample_indices[0] + frame_offsets[cam] < 0
                or sample_indices[-1] + frame_offsets[cam] >= count}
            if invalid_ranges:
                for cam, indices in invalid_ranges.items():
                    summary["errors"].append(
                        f"Requested source frame indices {indices} outside cam{cam} video range "
                        f"0..{frame_counts[cam] - 1} after offset {frame_offsets[cam]:+d}")
                for cam in range(1, args.cams + 1):
                    if cam == ref_cam:
                        continue
                    record = summary["pairs"][f"{ref_cam}-{cam}"]
                    record["status"] = ("FAILED" if ref_cam in invalid_ranges or cam in invalid_ranges
                                        else "INCOMPLETE")
                    record["reasons"].append("Evaluation stopped: requested frame indices are out of range")
                return finish_report(summary, out_dir)
        else:
            low = max([0] + [-frame_offsets[cam] for cam in frame_counts])
            high = min((count - 1 - frame_offsets[cam] for cam, count in frame_counts.items()),
                       default=-1)
            total_frames = high - low + 1
            summary["common_logical_frame_range"] = [low, high] if total_frames > 0 else None
            sample_indices = np.unique(np.linspace(
                low + total_frames * 0.1, min(low + total_frames * 0.9, high),
                args.num_frames, dtype=int)) if total_frames > 0 else []
        summary["sample_frame_indices"] = [int(i) for i in sample_indices]
        summary["source_frame_indices"] = {
            f"cam{cam}": [int(index) + frame_offsets[cam] for index in sample_indices]
            for cam in range(1, args.cams + 1)}
        for cam in range(1, args.cams + 1):
            if cam == ref_cam:
                continue
            record = summary["pairs"][f"{ref_cam}-{cam}"]
            record["source_frame_indices"] = {
                f"cam{camera}": summary["source_frame_indices"][f"cam{camera}"]
                for camera in (ref_cam, cam)}
            for camera in (ref_cam, cam):
                record["reasons"].extend(cams[camera]["errors"])
                if camera in video_errors:
                    record["reasons"].append(video_errors[camera])
            if "R" not in cams[cam] and not cams[cam]["errors"]:
                record["reasons"].append(f"Missing extrinsics for cam{cam}")
            if not len(sample_indices):
                record["reasons"].append("No common readable logical frames after applying offsets")
            if record["reasons"]:
                if any(reason.startswith("Invalid ") for reason in record["reasons"]):
                    record["status"] = "FAILED"
                continue
            try:
                evaluate_pair(ref_cam, cam, cams, caps, sample_indices,
                              (cols - 1, rows - 1), out_dir, record, frame_offsets)
            except (ValueError, KeyError, TypeError, cv2.error, np.linalg.LinAlgError) as exc:
                record["status"] = "FAILED"
                record["reasons"].append(f"Geometry evaluation failed: {exc}")
    finally:
        for cap in caps.values():
            cap.release()
    return finish_report(summary, out_dir)


if __name__ == "__main__":
    sys.exit(main())
