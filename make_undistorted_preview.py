#!/usr/bin/env python3
"""Make a labeled, synchronized grid from completed undistorted video exports."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

from export_calibrated_videos import probe_video, read_json, signature


def ffmpeg(arguments):
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                    *map(str, arguments)], check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=".")
    parser.add_argument("--episode", default="episode_0004")
    parser.add_argument("--decode-accel", choices=["auto", "none", "videotoolbox"], default="auto")
    args = parser.parse_args(argv)
    episode = Path(args.base).resolve() / "output" / args.episode
    video_dir = episode / "synced_undistorted"
    manifest_path = video_dir / "export_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("image_space") != "undistorted" or manifest.get("partial_export"):
        raise ValueError("A complete undistorted export is required")
    names = sorted(manifest["cameras"], key=lambda name: int(name.removeprefix("cam")))
    if not names:
        raise ValueError("No cameras in export manifest")
    paths = {name: video_dir / f"{name}_synced_undistorted.mp4" for name in names}
    sources = {name: signature(path) for name, path in paths.items()}
    for name, path in paths.items():
        if sources[name] != manifest["output_signatures"][path.name]:
            raise ValueError(f"Export changed since validation: {path}")
    probes = {name: probe_video(path) for name, path in paths.items()}
    source_fps = Fraction(manifest["fps"])
    source_frames = manifest["frame_count"]
    for name, probe in probes.items():
        if probe["fps"] != str(source_fps) or probe["frame_count"] != source_frames:
            raise ValueError(f"Camera does not share the export timeline: {name}")
    stride = max(1, math.ceil(source_fps / 30))
    fps = source_fps / stride
    frames = math.ceil(source_frames / stride)
    cols = len(names) if len(names) <= 3 else math.ceil(len(names) / 2)
    rows = math.ceil(len(names) / cols)
    width, height = 640, 360
    reference = read_json(episode / "calibration_undistorted/calibration_all_cameras.json")["reference_camera"]
    if reference not in names:
        raise ValueError("Reference camera is missing from the preview")
    output = episode / f"{args.episode}_undistorted_preview.mp4"
    poster = output.with_suffix(".jpg")
    report_path = output.with_suffix(".json")
    fingerprint = {"export_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                   "sources": sources, "audio_reference_camera": reference, "preview_version": 1}
    if output.is_file() and poster.is_file() and report_path.is_file():
        previous = read_json(report_path)
        if (previous.get("fingerprint") == fingerprint
                and previous.get("video_signature") == signature(output)
                and previous.get("poster_signature") == signature(poster)):
            print(f"Up to date: {output}")
            return 0
    acceleration = ("videotoolbox" if platform.system() == "Darwin" else "none") if args.decode_accel == "auto" else args.decode_accel
    staging = Path(tempfile.mkdtemp(prefix=".preview-staging-", dir=episode))

    def make_proxy(name):
        command = ["-threads", "2"]
        if acceleration != "none":
            command += ["-hwaccel", acceleration]
        command += ["-i", paths[name], "-map", "0:v:0", "-an",
                    "-vf", f"select=not(mod(n\\,{stride})),setpts=N*{fps.denominator}/({fps.numerator}*TB),"
                    f"scale={width}:{height}:force_original_aspect_ratio=decrease:force_divisible_by=2:out_range=limited,"
                    f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p",
                    "-r", fps, "-fps_mode", "cfr", "-frames:v", frames,
                    "-c:v", "libx264", "-threads", "2", "-preset", "veryfast", "-crf", "18",
                    "-bf", "0", "-color_range", "tv", "-colorspace", "bt709",
                    "-color_primaries", "bt709", "-color_trc", "bt709",
                    "-video_track_timescale", fps.numerator, staging / f"{name}.mp4"]
        ffmpeg(command)
        print(f"Preview tile ready: {name}", flush=True)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for future in as_completed([pool.submit(make_proxy, name) for name in names]):
                future.result()
        # Labels are rasterized once because FFmpeg builds may omit drawtext.
        labels = np.zeros((rows * height, cols * width, 4), dtype=np.uint8)
        for index, name in enumerate(names):
            x, y = index % cols * width, index // cols * height
            cv2.rectangle(labels, (x + 8, y + 8), (x + 158, y + 48), (12, 12, 12, 255), -1)
            cv2.putText(labels, f"CAM {name[3:]}", (x + 20, y + 37), cv2.FONT_HERSHEY_SIMPLEX,
                        .85, (255, 255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(staging / "labels.png"), labels)
        command = ["-filter_complex_threads", "2"]
        for name in names:
            command += ["-threads", "1", "-i", staging / f"{name}.mp4"]
        command += ["-i", staging / "labels.png"]
        has_audio = probes[reference]["has_audio"]
        if has_audio:
            command += ["-i", paths[reference]]
        inputs = "".join(f"[{index}:v]" for index in range(len(names)))
        layout = "|".join(f"{index % cols * width}_{index // cols * height}" for index in range(len(names)))
        grid = f"{inputs}xstack=inputs={len(names)}:layout={layout}:fill=black[grid]" if len(names) > 1 else "[0:v]null[grid]"
        command += ["-filter_complex", f"{grid};[grid][{len(names)}:v]overlay=0:0:format=auto,format=yuv420p[v]", "-map", "[v]"]
        if has_audio:
            command += ["-map", f"{len(names) + 1}:a:0", "-c:a", "copy"]
        command += ["-frames:v", frames, "-r", fps, "-fps_mode", "cfr",
                    "-c:v", "libx264", "-threads", "4", "-preset", "fast", "-crf", "19",
                    "-bf", "0", "-color_range", "tv", "-colorspace", "bt709",
                    "-color_primaries", "bt709", "-color_trc", "bt709",
                    "-video_track_timescale", fps.numerator, "-movflags", "+faststart", staging / output.name]
        ffmpeg(command)
        checked = probe_video(staging / output.name)
        if (checked["frame_count"], checked["fps"], checked["width"], checked["height"]) != (frames, str(fps), cols * width, rows * height):
            raise ValueError("Preview does not match its intended grid/timeline")
        if sources != {name: signature(path) for name, path in paths.items()}:
            raise ValueError("Source exports changed while creating the preview")
        ffmpeg(["-i", staging / output.name, "-frames:v", "1", "-q:v", "2", staging / poster.name])
        (staging / output.name).replace(output)
        (staging / poster.name).replace(poster)
        report = {"fingerprint": fingerprint, "source_directory": str(video_dir),
                  "cameras_row_major": names, "grid": [cols, rows], "tile_size": [width, height],
                  "source_fps": str(source_fps), "source_frame_count": source_frames,
                  "source_frame_stride": stride, "source_frame_mapping": "preview frame n uses source frame n * stride",
                  "audio_camera": reference if has_audio else None, "output_probe": checked,
                  "video_signature": signature(output), "poster_signature": signature(poster)}
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    except BaseException:
        print(f"Preview diagnostics retained: {staging}")
        raise
    else:
        shutil.rmtree(staging)
    print(f"Preview complete: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
