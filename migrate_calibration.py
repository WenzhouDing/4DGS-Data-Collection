#!/usr/bin/env python3
"""Migrate calibration extrinsics to OpenCV world-to-camera coordinates.

Missing convention tags mean the repository's legacy camera-to-world format.
The entire calibration directory is backed up before any changed JSON is replaced.
Intrinsics, images, and other assets are preserved; distortion is not recalibrated.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np

from calibration_geometry import (
    EXTRINSICS_CONVENTION, fundamental_from_KRT, world_to_camera,
)


def _camera_name(value):
    if not isinstance(value, str) or re.fullmatch(r"cam[1-9][0-9]*", value) is None:
        raise ValueError(f"Invalid camera name: {value!r}")
    return value


def _read(path):
    with path.open() as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _same_numeric(first, second, label):
    try:
        a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
        equal = a.shape == b.shape and np.allclose(a, b, rtol=1e-10, atol=1e-10)
    except (TypeError, ValueError):
        equal = False
    if not equal:
        raise ValueError(f"Individual and combined calibration disagree: {label}")


def _euler_degrees(rotation):
    """Angles of stored R under the column-vector R = Rz @ Ry @ Rx order."""
    sy = np.hypot(rotation[0, 0], rotation[1, 0])
    if sy > 1e-8:
        rx = np.arctan2(rotation[2, 1], rotation[2, 2])
        ry = np.arctan2(-rotation[2, 0], sy)
        rz = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        rx = np.arctan2(-rotation[1, 2], rotation[1, 1])
        ry = np.arctan2(-rotation[2, 0], sy)
        rz = 0.0
    return {key: round(float(np.degrees(angle)), 6)
            for key, angle in zip(("rx", "ry", "rz"), (rx, ry, rz))}


def prepare_migration(calibration_dir):
    """Read and validate every input, returning changed-file bytes without writing."""
    directory = Path(calibration_dir).resolve()
    if not directory.is_dir():
        raise ValueError(f"Calibration directory does not exist: {directory}")
    combined_path = directory / "calibration_all_cameras.json"
    combined = _read(combined_path) if combined_path.exists() else {"cameras": {}}
    cameras = deepcopy(combined.get("cameras", {}))
    if not isinstance(cameras, dict) or any(not isinstance(c, dict) for c in cameras.values()):
        raise ValueError("Combined cameras must be a mapping of camera names to objects")
    for name in cameras:
        _camera_name(name)

    top_convention = combined.get("extrinsics_convention")
    if top_convention not in (None, "camera_to_world", EXTRINSICS_CONVENTION):
        raise ValueError(f"Unknown combined extrinsics convention: {top_convention!r}")
    if combined.get("schema_version", 1) not in (1, 2):
        raise ValueError("Unsupported combined calibration schema version")
    if combined.get("schema_version") == 2 and top_convention != EXTRINSICS_CONVENTION:
        raise ValueError("Schema version 2 requires world_to_camera convention metadata")

    individual_ext = {}
    for path in sorted(directory.glob("cam*_intrinsics.json")):
        name = _camera_name(path.name.removesuffix("_intrinsics.json"))
        intr = _read(path)
        entry = cameras.setdefault(name, {})
        for key, value in intr.items():
            if key in entry:
                if key in ("K", "dist", "image_size", "rms_error_px"):
                    _same_numeric(entry[key], value, f"{name}.{key}")
                elif entry[key] != value:
                    raise ValueError(f"Individual and combined calibration disagree: {name}.{key}")
            else:
                entry[key] = deepcopy(value)
    for path in sorted(directory.glob("cam*_extrinsics.json")):
        name = _camera_name(path.name.removesuffix("_extrinsics.json"))
        if name not in cameras:
            raise ValueError(f"No intrinsics for {name}")
        individual_ext[name] = _read(path)

    references = set()
    if combined.get("reference_camera") is not None:
        references.add(_camera_name(combined["reference_camera"]))
    for name, entry in cameras.items():
        ext = entry.get("extrinsics")
        if ext is not None and not isinstance(ext, dict):
            raise ValueError(f"Invalid combined extrinsics for {name}")
        if ext and top_convention is not None:
            declared = ext.get("convention", "camera_to_world")
            if declared != top_convention:
                raise ValueError(f"Combined convention contradicts {name} extrinsics")
        for candidate in (ext, individual_ext.get(name)):
            if candidate:
                references.add(_camera_name(candidate.get("reference")))
                if candidate.get("target", name) != name:
                    raise ValueError(f"Extrinsics target does not match file/entry {name}")
    if len(references) != 1:
        raise ValueError(f"Expected one consistent reference camera, found {sorted(references)}")
    reference = references.pop()
    if reference not in cameras:
        raise ValueError(f"Reference {reference} has no intrinsics")

    for name, entry in cameras.items():
        K = np.asarray(entry.get("K"), dtype=float)
        if K.shape != (3, 3) or not np.isfinite(K).all() or abs(np.linalg.det(K)) < 1e-12:
            raise ValueError(f"Invalid or missing intrinsic matrix for {name}")

    updated = {}
    Kref = np.asarray(cameras[reference]["K"], dtype=float)
    for name, entry in cameras.items():
        old_combined = entry.get("extrinsics")
        old_individual = individual_ext.get(name)
        if old_combined and old_individual:
            Rc, tc = world_to_camera(old_combined, expected_reference=reference)
            Ri, ti = world_to_camera(old_individual, expected_reference=reference)
            _same_numeric(Rc, Ri, f"{name}.R")
            _same_numeric(tc, ti, f"{name}.T")
        original = deepcopy(old_combined or {})
        original.update(deepcopy(old_individual or {}))
        pose_source = old_individual or old_combined
        if not original:
            if name != reference:
                continue  # A camera with intrinsics only remains uncalibrated.
            original = {"reference": reference, "target": name,
                        "convention": EXTRINSICS_CONVENTION,
                        "R": np.eye(3).tolist(), "T": [0., 0., 0.],
                        "method": "reference", "path": [int(reference[3:])]}
            pose_source = original
        R, t = world_to_camera(pose_source, expected_reference=reference)
        # Legacy inversion returns a transposed view. Use the same memory layout
        # as a JSON reload so BLAS rounding cannot change F on the second run.
        R = np.array(R, dtype=float, order="C", copy=True)
        t = np.asarray(t).reshape(3)
        if name == reference:
            if not np.allclose(R, np.eye(3), atol=1e-10, rtol=0) or not np.allclose(t, 0, atol=1e-10, rtol=0):
                raise ValueError("Reference camera must have identity rotation and zero translation")
            R, t = np.eye(3), np.zeros(3)
            F = None  # There is no stereo fundamental matrix for a camera with itself.
        else:
            F = fundamental_from_KRT(Kref, np.asarray(entry["K"]), R, t.reshape(3, 1)).tolist()
        original.update({
            "reference": reference, "target": name,
            "convention": EXTRINSICS_CONVENTION,
            "R": R.tolist(), "T": t.tolist(), "F": F,
            "baseline_m": round(float(np.linalg.norm(t)), 6),
            "euler_deg": _euler_degrees(R),
            "euler_rotation_order": "Rz @ Ry @ Rx",
        })
        entry["extrinsics"] = original
        updated[f"{name}_extrinsics.json"] = original

    combined.update({"schema_version": 2, "extrinsics_convention": EXTRINSICS_CONVENTION,
                     "reference_camera": reference, "cameras": cameras})
    updated["calibration_all_cameras.json"] = combined
    changes = {}
    for filename, data in updated.items():
        encoded = (json.dumps(data, indent=2, allow_nan=False) + "\n").encode()
        destination = directory / filename
        if not destination.exists() or destination.read_bytes() != encoded:
            changes[filename] = encoded
    return directory, changes


def migrate_calibration(calibration_dir):
    """Validate, back up, and replace changed JSON; no-op when already migrated."""
    directory, changes = prepare_migration(calibration_dir)
    if not changes:
        return {"changed": False, "backup_dir": None, "files_updated": []}

    # Prepare all output before touching any calibration file. The backup sits
    # beside the source directory, so it cannot recursively include itself.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    backup = directory.with_name(f"{directory.name}_backup_{stamp}")
    with tempfile.TemporaryDirectory(prefix=".calibration-migration-", dir=directory.parent) as temp:
        staged = Path(temp)
        for filename, content in changes.items():
            with (staged / filename).open("wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        shutil.copytree(directory, backup, symlinks=True)
        replaced = []
        try:
            for filename in changes:
                os.replace(staged / filename, directory / filename)
                replaced.append(filename)
        except BaseException:
            for filename in reversed(replaced):
                old = backup / filename
                if old.exists():
                    shutil.copy2(old, directory / filename)
                else:
                    (directory / filename).unlink()
            raise
    return {"changed": True, "backup_dir": str(backup),
            "files_updated": sorted(changes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", required=True,
                        help="Directory containing calibration JSON files")
    args = parser.parse_args()
    try:
        result = migrate_calibration(args.calibration_dir)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Migration failed: {error}\n")
    if result["changed"]:
        print(f"Updated {len(result['files_updated'])} JSON files to world_to_camera.")
        print(f"Backup: {result['backup_dir']}")
    else:
        print("Already migrated; no files changed.")
    print("Intrinsics and distortion coefficients are unchanged; distortion was not recalibrated.")


if __name__ == "__main__":
    main()
