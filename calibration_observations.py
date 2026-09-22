"""Portable, source-checked checkerboard detections for repeatable calibration."""
import json
from pathlib import Path

import numpy as np
from calibration_frame_offsets import normalize_frame_offsets


def source_signature(synced_dir, cameras):
    result = {}
    for cam in cameras:
        p = Path(synced_dir) / f"cam{cam}_synced.mp4"
        stat = p.stat()
        result[str(cam)] = {"name": p.name, "size": stat.st_size,
                            "mtime_ns": stat.st_mtime_ns}
    return result


def save_corner_cache(path, cam_corners, image_size, board_size, scan_info, synced_dir,
                      frame_offsets=None):
    cameras = sorted(cam_corners)
    offsets = normalize_frame_offsets(
        {f"cam{cam}": value for cam, value in (frame_offsets or {}).items()}, len(cameras))
    metadata = {"schema_version": 1, "image_size": list(image_size),
                "board_size": list(board_size), "cameras": cameras,
                "frame_offsets": {f"cam{cam}": offsets[cam] for cam in cameras},
                "scan_info": scan_info,
                "sources": source_signature(synced_dir, cameras)}
    arrays = {"metadata": np.array(json.dumps(metadata))}
    for cam in cameras:
        names = sorted(cam_corners[cam])
        arrays[f"cam{cam}_frames"] = np.array([int(n[6:-4]) for n in names], dtype=np.int64)
        arrays[f"cam{cam}_corners"] = np.array(
            [cam_corners[cam][n][0] for n in names], dtype=np.float32
        ).reshape(len(names), int(np.prod(board_size)), 1, 2)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        np.savez_compressed(f, **arrays)


def load_corner_cache(path, synced_dir, board_size, num_cams, frame_offsets=None):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"]))
        cameras = list(range(1, num_cams + 1))
        if (meta.get("schema_version") != 1 or meta["board_size"] != list(board_size)
                or meta["cameras"] != cameras):
            raise ValueError("Corner cache does not match board/camera configuration")
        if meta["sources"] != source_signature(synced_dir, cameras):
            raise ValueError("Corner cache source videos have changed")
        expected_offsets = normalize_frame_offsets(
            {f"cam{cam}": value for cam, value in (frame_offsets or {}).items()}, num_cams)
        if normalize_frame_offsets(meta.get("frame_offsets", {}), num_cams) != expected_offsets:
            raise ValueError("Corner cache frame offsets do not match the requested alignment")
        result = {}
        for cam in cameras:
            frames = data[f"cam{cam}_frames"]
            corners = data[f"cam{cam}_corners"]
            expected = (len(frames), int(np.prod(board_size)), 1, 2)
            if (frames.ndim != 1 or not np.issubdtype(frames.dtype, np.integer)
                    or np.any(frames < 0)
                    or np.any(frames + expected_offsets[cam] < 0)
                    or len(np.unique(frames)) != len(frames)
                    or (len(frames) and np.any(np.diff(frames) <= 0))
                    or corners.shape != expected
                    or not np.isfinite(corners).all()):
                raise ValueError(f"Invalid cached observations for cam{cam}")
            result[cam] = {f"frame_{int(n):06d}.jpg": (p.copy(), None)
                           for n, p in zip(frames, corners)}
        scan = {int(k): v for k, v in meta["scan_info"].items()}
        return result, tuple(meta["image_size"]), scan


def split_observations(cam_corners):
    """Reserve every fifth source observation globally, before fitting any camera."""
    names = sorted(set().union(*(set(c) for c in cam_corners.values())))
    held_out = set(names[4::5])
    train = {cam: {n: p for n, p in rows.items() if n not in held_out}
             for cam, rows in cam_corners.items()}
    validation = {cam: {n: p for n, p in rows.items() if n in held_out}
                  for cam, rows in cam_corners.items()}
    return train, validation, sorted(held_out)
