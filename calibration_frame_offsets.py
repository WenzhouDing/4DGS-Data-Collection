"""Validated per-camera source-frame offsets shared by calibration and evaluation."""

from collections.abc import Mapping
import json
from numbers import Integral
from pathlib import Path
import re


def normalize_frame_offsets(mapping, num_cams):
    """Return integer camera IDs mapped to offsets, filling omitted cameras with 0.

    Input keys are canonical ``cam1`` ... ``camN`` strings. A logical frame n
    reads source frame n + offset for that camera. Values must be integers;
    booleans and floating-point values, including 1.0, are rejected.
    """
    if isinstance(num_cams, bool) or not isinstance(num_cams, Integral) or num_cams < 1:
        raise ValueError("num_cams must be a positive integer")
    if not isinstance(mapping, Mapping):
        raise ValueError("Frame offsets must be a JSON object mapping camN to integer offsets")
    result = {cam: 0 for cam in range(1, int(num_cams) + 1)}
    for key, value in mapping.items():
        if not isinstance(key, str) or re.fullmatch(r"cam[1-9][0-9]*", key) is None:
            raise ValueError(f"Invalid frame-offset camera name: {key!r}")
        cam = int(key[3:])
        if cam not in result:
            raise ValueError(f"Frame-offset camera {key} is outside 1..{num_cams}")
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"Frame offset for {key} must be an integer, not {value!r}")
        result[cam] = int(value)
    return result


def load_frame_offsets(path, num_cams):
    """Load a JSON camN->integer mapping; no file means all-zero offsets."""
    if path is None:
        return normalize_frame_offsets({}, num_cams)
    with Path(path).open() as stream:
        mapping = json.load(stream)
    return normalize_frame_offsets(mapping, num_cams)
