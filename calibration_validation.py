#!/usr/bin/env python3
"""Validate a calibration before reuse, optionally copying it with a backup.

Checks saved final-test evidence and current geometry; does not remeasure video
or treat convention migration as a distortion refit.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np

from calibration_geometry import (
    EXTRINSICS_CONVENTION, distortion_diagnostics, fundamental_from_KRT,
    world_to_camera,
)
from calibration_frame_offsets import normalize_frame_offsets
from calibration_fit import rational_positive_poles
from run_calibration import publish_calibration


def _read(path):
    with path.open() as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _equal_numeric(a, b, label):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    _require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
             and np.allclose(a, b, rtol=1e-9, atol=1e-10),
             f"Individual/combined calibration mismatch: {label}")


def _finite_nonnegative(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value) and value >= 0


def validate_calibration(calibration_dir, num_cams, reference):
    """Fail closed for unvalidated, incomplete, or incompatible saved calibration."""
    directory = Path(calibration_dir).resolve()
    _require(num_cams >= 2 and 1 <= reference <= num_cams, "Invalid camera count/reference")
    expected = {f"cam{i}" for i in range(1, num_cams+1)}
    ref_name = f"cam{reference}"
    combined = _read(directory / "calibration_all_cameras.json")
    _require(combined.get("schema_version") == 2 and
             combined.get("extrinsics_convention") == EXTRINSICS_CONVENTION,
             "Calibration needs schema 2/world_to_camera metadata; migration alone does not validate lens models")
    _require(combined.get("reference_camera") == ref_name, "Calibration reference does not match --ref-cam")
    offsets = normalize_frame_offsets(combined.get("source_frame_offsets", {}), num_cams)
    cameras = combined.get("cameras")
    _require(isinstance(cameras, dict) and set(cameras) == expected,
             "Combined calibration does not contain exactly the requested cameras")
    poses, intrinsics = {}, {}
    for name in sorted(expected):
        intr = _read(directory / f"{name}_intrinsics.json")
        entry = cameras[name]
        for key in ("K", "dist", "image_size"):
            _equal_numeric(intr[key], entry[key], f"{name}.{key}")
        diagnostics = distortion_diagnostics(intr["K"], intr["dist"], intr["image_size"])
        _require(diagnostics["valid"], f"{name} lens model is invalid: {'; '.join(diagnostics['reasons'])}")
        _require(not rational_positive_poles(intr["dist"]),
                 f"{name} rational distortion has a positive-radius pole; refit with a nonsingular model")
        intrinsics[name] = intr
        ext = _read(directory / f"{name}_extrinsics.json")
        nested = entry["extrinsics"]
        for candidate in (ext, nested):
            _require(candidate.get("convention") == EXTRINSICS_CONVENTION,
                     f"{name} needs explicit world_to_camera convention")
            _require(candidate.get("target") == name, f"Wrong extrinsics target for {name}")
        R, t = world_to_camera(ext, expected_reference=ref_name)
        Rc, tc = world_to_camera(nested, expected_reference=ref_name)
        _equal_numeric(R, Rc, f"{name}.R")
        _equal_numeric(t, tc, f"{name}.T")
        poses[name] = (R, t, ext, nested)
    Rref, tref, _, _ = poses[ref_name]
    _require(np.allclose(Rref, np.eye(3), atol=1e-10, rtol=0)
             and np.allclose(tref, 0, atol=1e-10, rtol=0),
             "Reference transform must be identity")
    for name, (R, t, ext, nested) in poses.items():
        if name == ref_name:
            continue
        F = fundamental_from_KRT(intrinsics[ref_name]["K"], intrinsics[name]["K"], R, t)
        _require(np.linalg.norm(F) > 1e-15, f"Degenerate baseline for {name}")
        for candidate in (ext, nested):
            saved = np.asarray(candidate.get("F"), dtype=float)
            _require(saved.shape == (3, 3) and np.isfinite(saved).all()
                     and np.linalg.norm(saved) > 1e-15, f"Missing/invalid F for {name}")
            scale = np.sum(saved*F) / np.sum(F*F)
            _require(np.linalg.norm(saved-scale*F) / np.linalg.norm(saved) < 1e-8,
                     f"Fundamental matrix has wrong geometry/direction for {name}")

    split = _read(directory / "observation_split.json")
    _require(normalize_frame_offsets(split.get("source_frame_offsets", {}), num_cams) == offsets,
             "Observation split and calibration have different source-frame offsets")
    sets = {}
    for key in ("training", "model_selection", "test"):
        values = split.get(key)
        _require(isinstance(values, list) and all(isinstance(v, str) and
                 re.fullmatch(r"frame_[0-9]{6,}\.jpg", v) for v in values),
                 f"Invalid or missing {key} source-frame split")
        sets[key] = set(values)
        _require(len(sets[key]) == len(values), f"Duplicate frames in {key} split")
    _require(len(sets["training"]) >= 5 and len(sets["model_selection"]) >= 3
             and len(sets["test"]) >= 3, "Insufficient saved fit/selection/test frame coverage")
    _require(not (sets["training"] & sets["model_selection"] or
                  sets["training"] & sets["test"] or sets["model_selection"] & sets["test"]),
             "Training, model-selection, and final-test frames overlap")
    test_indices = {int(name[6:-4]) for name in sets["test"]}
    report = _read(directory / "heldout_epipolar.json")
    _require(report.get("passed") is True and report.get("reference") == ref_name,
             "Saved independent final-test validation did not pass")
    pairs = report.get("pairs")
    expected_pairs = {f"{reference}-{i}" for i in range(1, num_cams+1) if i != reference}
    _require(isinstance(pairs, dict) and set(pairs) == expected_pairs,
             "Final-test validation is missing expected camera pairs")
    for name, record in pairs.items():
        frames = record.get("frames", [])
        indices = [f.get("source_frame") for f in frames]
        _require(record.get("status") == "PASS" and record.get("invalid_points") == 0,
                 f"Final-test pair {name} did not pass or contains invalid points")
        _require(len(indices) >= 3 and all(type(i) is int and i in test_indices for i in indices)
                 and len(set(indices)) == len(indices), f"Pair {name} lacks three distinct final-test observations")
        _require(_finite_nonnegative(record.get("mean_px")) and record["mean_px"] < 2
                 and _finite_nonnegative(record.get("p95_px")) and record["p95_px"] < 5,
                 f"Final-test pair {name} exceeds the required error limits")
    return {"reference": ref_name, "camera_count": num_cams,
            "validated_pairs": sorted(expected_pairs)}


def copy_calibration(source, destination, num_cams, reference, source_episode=None):
    """Validate before copying; preserve the previous destination in a backup."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    _require(source != destination and source not in destination.parents
             and destination not in source.parents, "Source and destination must be separate directories")
    validate_calibration(source, num_cams, reference)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".calibration-copy-", dir=destination.parent))
    try:
        shutil.copytree(source, staging, dirs_exist_ok=True)
        if source_episode is not None:
            provenance = {"source_episode": source_episode,
                          "copied_at": datetime.now(timezone.utc).isoformat(),
                          "note": "Reused for an episode with the same physical rig configuration."}
            (staging / "calibration_source.json").write_text(json.dumps(provenance, indent=2) + "\n")
        validate_calibration(staging, num_cams, reference)
        return publish_calibration(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", required=True)
    parser.add_argument("--cams", required=True, type=int)
    parser.add_argument("--ref-cam", required=True, type=int)
    parser.add_argument("--copy-to", help="Copy validated calibration to this directory with a backup")
    parser.add_argument("--source-episode", help="Provenance label when copying")
    args = parser.parse_args(argv)
    try:
        if args.copy_to:
            backup = copy_calibration(args.calibration_dir, args.copy_to, args.cams,
                                      args.ref_cam, args.source_episode)
            print(f"Validated calibration copied to {args.copy_to}")
            if backup:
                print(f"Previous destination backed up: {backup}")
        else:
            result = validate_calibration(args.calibration_dir, args.cams, args.ref_cam)
            print(f"Calibration reuse checks passed: {result['camera_count']} cameras, "
                  f"reference {result['reference']}, all final-test pairs passed")
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"Calibration cannot be reused: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
