#!/usr/bin/env python3
"""Export a common video timeline with lens distortion removed, preserving sources."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time

import cv2
import numpy as np

from calibration_geometry import distortion_diagnostics
from calibration_validation import validate_calibration
from run_eval_epipolar import infer_reference, resolve_frame_offsets


def read_json(path):
    return json.loads(Path(path).read_text())


def signature(path):
    stat = Path(path).stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def probe_video(path):
    """Count displayed packets, excluding MP4 discard/preroll samples.

    Header nb_frames includes discarded samples in stream-copied GoPro files.
    Require a constant presentation grid; validate actual decoded output later.
    """
    info = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)]))
    video = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    if video is None:
        raise ValueError(f"No video stream: {path}")
    fps = Fraction(video["r_frame_rate"])
    if fps <= 0 or Fraction(video["avg_frame_rate"]) != fps:
        raise ValueError(f"Expected constant frame rate: {path}")
    packets = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_packets",
        "-show_entries", "packet=pts,flags", "-of", "json", str(path)]))["packets"]
    pts = sorted(int(p["pts"]) for p in packets
                 if "pts" in p and int(p["pts"]) >= 0 and "D" not in p.get("flags", ""))
    tick = Fraction(video["time_base"])
    if not pts or len(set(pts)) != len(pts):
        raise ValueError(f"Missing or duplicate displayed timestamps: {path}")
    if any(abs(float(p * tick - i / fps)) > float(tick) + 1e-8
           for i, p in enumerate(pts)):
        raise ValueError(f"Video must have a regular presentation grid starting at zero: {path}")
    # The BGR8 processing path must never silently reduce a higher bit-depth input.
    if video.get("pix_fmt") not in {"yuv420p", "yuvj420p", "nv12", "yuv422p",
                                    "yuvj422p", "yuv444p", "yuvj444p", "rgb24", "bgr24"}:
        raise ValueError(f"Unsupported pixel format {video.get('pix_fmt')}: this exporter preserves 8-bit inputs")
    return {"width": video["width"], "height": video["height"],
            "fps": str(fps), "frame_count": len(pts),
            "header_frame_count": video.get("nb_frames"),
            "pixel_format": video.get("pix_fmt"), "color_range": video.get("color_range", "tv"),
            "color_space": video.get("color_space", "bt709"),
            "color_primaries": video.get("color_primaries", "bt709"),
            "color_transfer": video.get("color_transfer", "bt709"),
            "bit_rate": int(video.get("bit_rate", 0)),
            "has_audio": any(s["codec_type"] == "audio" for s in info["streams"])}


def common_frame_range(probes, offsets, max_frames=None):
    rates = {p["fps"] for p in probes.values()}
    if len(rates) != 1:
        raise ValueError("Camera frame rates differ; integer offsets cannot synchronize them")
    start = max(0, -min(offsets.values()))
    stop = min(p["frame_count"] - offsets[cam] for cam, p in probes.items())
    if max_frames is not None:
        if max_frames < 1:
            raise ValueError("--max-frames must be positive")
        stop = min(stop, start + max_frames)
    if stop <= start:
        raise ValueError("No common video frames remain after alignment")
    return start, stop


def read_frame(stream, size):
    chunks = bytearray()
    while len(chunks) < size:
        part = stream.read(size - len(chunks))
        if not part:
            break
        chunks.extend(part)
    if not chunks:
        return None
    if len(chunks) != size:
        raise ValueError("Decoder returned a truncated frame")
    return chunks


def export_camera(source, output, intrinsics, probe, source_start, frame_count,
                  encoder, decode_accel="none", progress_label="video"):
    """Decode displayed frames, bilinearly remap with OpenCV, and encode at source FPS."""
    source, output = Path(source), Path(output)
    width, height = probe["width"], probe["height"]
    if intrinsics["image_size"] != [width, height]:
        raise ValueError(f"Calibration/video dimensions differ: {source}")
    K, dist = np.array(intrinsics["K"]), np.array(intrinsics["dist"])
    maps = cv2.initUndistortRectifyMap(K, dist, None, K, (width, height), cv2.CV_32FC1)
    fps = Fraction(probe["fps"])
    duration = frame_count / float(fps)
    decode = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "2"]
    if decode_accel != "none":
        decode += ["-hwaccel", decode_accel]
    decode += ["-i", str(source), "-map", "0:v:0", "-an", "-sn", "-dn",
               "-vf", f"trim=start_frame={source_start}:end_frame={source_start+frame_count},setpts=PTS-STARTPTS",
               "-frames:v", str(frame_count), "-fps_mode", "passthrough",
               "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
    encode = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
              "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", f"{width}x{height}",
              "-framerate", str(fps), "-i", "pipe:0"]
    if probe["has_audio"]:
        encode += ["-i", str(source), "-map", "0:v:0", "-map", "1:a:0",
                   "-af", f"atrim=start={source_start/float(fps):.12f},asetpts=PTS-STARTPTS,apad,atrim=duration={duration:.12f}",
                   "-c:a", "aac", "-b:a", "192k"]
    else:
        encode += ["-map", "0:v:0", "-an"]
    color_range = "pc" if probe["color_range"] == "pc" else "tv"
    out_range = "full" if color_range == "pc" else "limited"
    matrix = probe["color_space"] if probe["color_space"] in {"bt709", "bt470bg", "smpte170m"} else "bt709"
    encode += ["-vf", f"scale=in_range=full:out_range={out_range}:out_color_matrix={matrix},format=yuv420p",
               "-c:v", encoder, "-pix_fmt", "yuv420p", "-color_range", color_range,
               "-colorspace", matrix, "-color_primaries", probe["color_primaries"],
               "-color_trc", probe["color_transfer"], "-g", str(round(float(fps))), "-bf", "0"]
    if encoder == "hevc_videotoolbox":
        bitrate = probe["bit_rate"] or round(width*height*float(fps)*.1)
        encode += ["-b:v", str(bitrate), "-tag:v", "hvc1"]
        if matrix == "bt709" and probe["color_primaries"] == "bt709" and probe["color_transfer"] == "bt709":
            encode += ["-bsf:v", "hevc_metadata="
                       f"video_full_range_flag={int(color_range == 'pc')}:"
                       "colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1"]
    else:
        encode += ["-preset", "fast", "-crf", "17"]
    encode += ["-frames:v", str(frame_count), "-fps_mode", "passthrough",
               "-video_track_timescale", str(fps.numerator),
               "-map_metadata", "-1", "-movflags", "+faststart", str(output)]
    processes = []
    with tempfile.TemporaryFile() as decode_log, tempfile.TemporaryFile() as encode_log:
        try:
            decoder = subprocess.Popen(decode, stdout=subprocess.PIPE, stderr=decode_log)
            processes.append(decoder)
            writer = subprocess.Popen(encode, stdin=subprocess.PIPE, stderr=encode_log)
            processes.append(writer)
            last = time.monotonic()
            for index in range(frame_count):
                raw = read_frame(decoder.stdout, width*height*3)
                if raw is None:
                    raise ValueError(f"{source.name}: ended at {index}/{frame_count} requested frames")
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
                corrected = cv2.remap(frame, *maps, interpolation=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT)
                writer.stdin.write(corrected.tobytes())
                if time.monotonic() - last > 10:
                    print(f"{progress_label}: {index+1}/{frame_count} frames", flush=True)
                    last = time.monotonic()
            writer.stdin.close()
            decoder.stdout.close()
            if decoder.wait() != 0 or writer.wait() != 0:
                raise RuntimeError("FFmpeg failed")
        except BaseException as exc:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                process.wait()
            decode_log.seek(0); encode_log.seek(0)
            details = decode_log.read().decode(errors="replace") + encode_log.read().decode(errors="replace")
            raise RuntimeError(f"{progress_label}: {exc}\n{details[-6000:]}") from exc
    checked = probe_video(output)
    if (checked["frame_count"] != frame_count or checked["fps"] != probe["fps"]
            or (checked["width"], checked["height"]) != (width, height)):
        raise ValueError(f"Exported video does not match the common timeline: {output}")
    print(f"{progress_label}: verified {frame_count} undistorted frames", flush=True)
    return checked


def write_output_calibration(combined, target, source_dir, manifest):
    """Same pixel K and world-to-camera pose; exported pixels require zero distortion."""
    target = Path(target); target.mkdir(parents=True, exist_ok=True)
    result = deepcopy(combined)
    result["image_space"] = "undistorted"
    result["source_calibration"] = str(source_dir)
    result["calibration_source_episode"] = combined.get("source_episode")
    result["source_episode"] = manifest["episode"]
    result["source_frame_offsets"] = {name: 0 for name in result["cameras"]}
    result["video_directory"] = "../synced_undistorted"
    result["export_timeline"] = {k: manifest[k] for k in ("logical_start_frame", "frame_count", "fps", "frame_offsets")}
    for name, entry in result["cameras"].items():
        entry["dist"] = [0.] * 5
        entry["distortion_model"] = "none"
        entry["distortion_validation"] = distortion_diagnostics(entry["K"], entry["dist"], entry["image_size"])
        intr = {key: value for key, value in entry.items() if key != "extrinsics"}
        (target/f"{name}_intrinsics.json").write_text(json.dumps(intr, indent=2)+"\n")
        (target/f"{name}_extrinsics.json").write_text(json.dumps(entry["extrinsics"], indent=2)+"\n")
    (target/'calibration_all_cameras.json').write_text(json.dumps(result, indent=2)+"\n")


def publish_export(staged_videos, staged_calibration, video_target, calibration_target):
    """Replace both output directories with rollback; preserve every previous output."""
    pairs = [(Path(staged_videos), Path(video_target)), (Path(staged_calibration), Path(calibration_target))]
    for _, target in pairs:
        if target.exists() and any(target.iterdir()):
            marker = target/'export_manifest.json' if target == pairs[0][1] else target/'calibration_all_cameras.json'
            if not marker.is_file():
                raise ValueError(f"Refusing to replace an unrelated directory: {target}")
    backups, published = [], []
    stamp = time.time_ns()
    try:
        for staged, target in pairs:
            if target.exists():
                backup = target.with_name(target.name+f".backup-{stamp}")
                target.rename(backup); backups.append((backup, target))
            staged.rename(target); published.append(target)
    except BaseException:
        for target in reversed(published):
            shutil.rmtree(target)
        for backup, target in reversed(backups):
            backup.rename(target)
        raise
    return [str(backup) for backup, _ in backups]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='.')
    parser.add_argument('--episode', default='episode_0001')
    parser.add_argument('--cams', type=int, default=12)
    parser.add_argument('--frame-offsets')
    parser.add_argument('--encoder', choices=['auto', 'libx264', 'hevc_videotoolbox'], default='auto')
    parser.add_argument('--decode-accel', choices=['auto', 'none', 'videotoolbox'], default='auto')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--max-frames', type=int, help='Diagnostic partial export only; omit for full videos')
    args = parser.parse_args(argv)
    if args.workers < 1:
        raise ValueError('--workers must be positive')
    episode = Path(args.base).resolve()/'output'/args.episode
    calibration = episode/'calibration'
    reference = infer_reference(str(calibration), args.cams, None)
    validate_calibration(calibration, args.cams, reference)
    combined = read_json(calibration/'calibration_all_cameras.json')
    offsets, offset_source = resolve_frame_offsets(args.frame_offsets, str(calibration), args.episode, args.cams)
    encoder = ('hevc_videotoolbox' if platform.system() == 'Darwin' else 'libx264') if args.encoder == 'auto' else args.encoder
    acceleration = ('videotoolbox' if platform.system() == 'Darwin' else 'none') if args.decode_accel == 'auto' else args.decode_accel
    sources = {cam: episode/'synced_raw'/f'cam{cam}_synced.mp4' for cam in range(1,args.cams+1)}
    source_signatures = {f'cam{cam}': signature(path) for cam,path in sources.items()}
    probes = {cam: probe_video(path) for cam,path in sources.items()}
    start, stop = common_frame_range(probes, offsets, args.max_frames)
    fingerprint = {"sources": source_signatures,
                   "calibration_sha256": hashlib.sha256((calibration/'calibration_all_cameras.json').read_bytes()).hexdigest(),
                   "frame_offsets": {f'cam{cam}': v for cam,v in offsets.items()},
                   "logical_start_frame": start, "frame_count": stop-start,
                   "encoder": encoder, "export_version": 1}
    video_target, calibration_target = episode/'synced_undistorted', episode/'calibration_undistorted'
    existing = video_target/'export_manifest.json'
    if existing.is_file() and (calibration_target/'calibration_all_cameras.json').is_file():
        previous = read_json(existing)
        if previous.get('fingerprint') == fingerprint and all(
                (video_target/name).is_file() and signature(video_target/name) == value
                for name,value in previous.get('output_signatures',{}).items()
        ) and len(previous.get('output_signatures',{})) == args.cams and all(
                (calibration_target/name).is_file()
                and hashlib.sha256((calibration_target/name).read_bytes()).hexdigest() == value
                for name,value in previous.get('calibration_hashes',{}).items()
        ) and len(previous.get('calibration_hashes',{})) == 2*args.cams+1:
            print(f'Up to date: {video_target}')
            return 0
    stage = Path(tempfile.mkdtemp(prefix='.undistort-staging-',dir=episode))
    staged_videos, staged_calibration = stage/'videos', stage/'calibration'
    staged_videos.mkdir()
    cv2.setNumThreads(4)
    manifest = {"schema_version": 1, "episode": args.episode, "fingerprint": fingerprint,
                "logical_start_frame": start, "frame_count": stop-start, "fps": probes[reference]['fps'],
                "frame_offsets": fingerprint['frame_offsets'], "offset_source": offset_source,
                "partial_export": args.max_frames is not None, "interpolation": "OpenCV INTER_LINEAR",
                "image_space": "undistorted", "output_K_policy": "original K; original resolution; reduced field of view",
                "calibration_directory": '../calibration_undistorted', "cameras": {}}
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            jobs = {pool.submit(export_camera, path, staged_videos/f'cam{cam}_synced_undistorted.mp4',
                    combined['cameras'][f'cam{cam}'], probes[cam], start+offsets[cam], stop-start,
                    encoder, acceleration, f'cam{cam}'): cam for cam,path in sources.items()}
            for future in as_completed(jobs):
                cam = jobs[future]
                manifest['cameras'][f'cam{cam}'] = {"source_video": str(sources[cam]),
                    "source_start_frame": start+offsets[cam], "input_probe": probes[cam],
                    "output_probe": future.result()}
        if source_signatures != {f'cam{cam}':signature(path) for cam,path in sources.items()}:
            raise ValueError('Source videos changed while exporting')
        manifest['output_signatures'] = {path.name:signature(path) for path in staged_videos.glob('*.mp4')}
        write_output_calibration(combined, staged_calibration, calibration, manifest)
        manifest['calibration_hashes'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                         for path in staged_calibration.glob('*.json')}
        (staged_videos/'export_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        backups = publish_export(staged_videos, staged_calibration, video_target, calibration_target)
        print(f'Export complete: {video_target}; matching calibration: {calibration_target}')
        for backup in backups:
            print(f'Previous output backed up: {backup}')
    except BaseException:
        print(f'Unpublished diagnostics retained: {stage}')
        raise
    else:
        stage.rmdir()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
