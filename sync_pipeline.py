#!/usr/bin/env python3
"""
GoPro Multi-Camera Multi-Session Sync Pipeline
================================================
Discovers recording sessions by pairing files in order across camera folders,
uses audio cross-correlation (clap sync) for frame-accurate alignment, then
outputs per-session synced raw video, a side-by-side preview, sync report,
and metadata.

Usage:
    python sync_pipeline.py [--base DIR] [--cams N] [--search-window SEC]

Defaults:
    --base          .               (directory containing camera folders 1/ 2/ ... N/)
    --cams          5
    --search-window 15              (seconds of audio to search for clap)
"""

import argparse
import numpy as np
import subprocess
import os
import sys
import json
import glob
import shutil
from datetime import datetime


# ─── CLI ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Multi-camera GoPro audio sync pipeline")
    p.add_argument("--base", default=".", help="Base directory containing camera folders 1/ 2/ ... N/")
    p.add_argument("--cams", type=int, default=12, help="Number of cameras")
    p.add_argument("--ref-cam", type=int, default=1,
                   help="Reference camera (cross-correlation reference + preview audio source)")
    p.add_argument("--search-window", type=int, default=15, help="Seconds of audio to search for clap")
    p.add_argument("--preview-height", type=int, default=360, help="Preview per-camera height in px")
    p.add_argument("--preview-max-sec", type=int, default=30, help="Max preview duration in seconds")
    return p.parse_args()


# ─── CONSTANTS ────────────────────────────────────────────────────
SAMPLE_RATE = 48000


def probe_creation_time(video_path):
    """Probe creation_time from format tags via ffprobe.

    Returns a timezone-aware datetime, or None if absent/unparseable.
    GoPro embeds this when the camera clock is set (Precision Time QR or GPS).
    """
    from datetime import datetime
    result = subprocess.run(
        ["ffprobe", "-v", "quiet",
         "-show_entries", "format_tags=creation_time",
         "-of", "csv=p=0", video_path],
        capture_output=True, text=True,
    )
    ts = result.stdout.strip()
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def group_chapters(file_paths, durations, creation_times, tol_sec=2.0):
    """Group chronologically-adjacent files into logical recordings.

    A GoPro auto-splits long recordings into chapter files. Within a camera,
    chapter N+1's creation_time ≈ chapter N's creation_time + chapter N's
    duration. Files within `tol_sec` of that expectation are merged.

    Inputs are parallel lists, all aligned by index. Files with missing
    creation_time are placed in their own group (no merging possible).

    Returns: list of dicts with keys:
      paths        — list of file paths in chapter order
      durations    — list of per-file durations (parallel to paths)
      total_dur    — sum of durations
      start_time   — datetime of first chapter (or None)
    """
    from datetime import timedelta
    n = len(file_paths)
    if n == 0:
        return []
    # Sort by creation_time (None values go to the end as standalone groups).
    order = sorted(range(n),
                   key=lambda i: (creation_times[i] is None,
                                  creation_times[i] or 0))
    groups = []
    current = None
    for i in order:
        ct = creation_times[i]
        if current is None or ct is None or current["start_time"] is None:
            if current is not None:
                groups.append(current)
            current = {"paths": [file_paths[i]],
                       "durations": [durations[i]],
                       "start_time": ct}
            continue
        prev_end = current["start_time"] + timedelta(seconds=sum(current["durations"]))
        if abs((ct - prev_end).total_seconds()) <= tol_sec:
            current["paths"].append(file_paths[i])
            current["durations"].append(durations[i])
        else:
            groups.append(current)
            current = {"paths": [file_paths[i]],
                       "durations": [durations[i]],
                       "start_time": ct}
    if current is not None:
        groups.append(current)
    for g in groups:
        g["total_dur"] = sum(g["durations"])
    return groups


def write_concat_list(paths, list_path):
    """Write an ffmpeg concat-demuxer list file. Returns list_path."""
    with open(list_path, "w") as f:
        for p in paths:
            # ffmpeg concat demuxer requires single-quoted absolute paths.
            f.write(f"file '{os.path.abspath(p)}'\n")
    return list_path


def ffmpeg_input_args(paths, work_dir, tag):
    """Return ffmpeg input args for one logical recording.

    Single file: ['-i', path]. Multiple chapters: concat demuxer.
    """
    if len(paths) == 1:
        return ["-i", paths[0]]
    list_path = os.path.join(work_dir, f"concat_{tag}.txt")
    write_concat_list(paths, list_path)
    return ["-f", "concat", "-safe", "0", "-i", list_path]


def probe_video_fps(video_path):
    """Probe actual FPS from video via ffprobe (e.g. 59.94, 119.88)."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0",
         video_path],
        capture_output=True, text=True,
    )
    fps_str = result.stdout.strip().split("\n")[0]
    if not fps_str:
        raise RuntimeError(
            f"ffprobe could not read fps from {video_path} "
            f"(rc={result.returncode}, stderr={result.stderr.strip()!r})")
    try:
        if "/" in fps_str:
            num, den = fps_str.split("/")
            return float(num) / float(den)
        return float(fps_str)
    except ValueError as e:
        raise RuntimeError(
            f"ffprobe returned unparseable fps {fps_str!r} for {video_path}") from e


# ─── HELPERS ──────────────────────────────────────────────────────
def extract_audio(video_paths, wav_path, work_dir, tag):
    """Extract mono 48 kHz WAV from one logical recording (one or more chapters).

    Multiple chapter files are concatenated via the ffmpeg concat demuxer.
    """
    paths = [video_paths] if isinstance(video_paths, str) else list(video_paths)
    input_args = ffmpeg_input_args(paths, work_dir, f"{tag}_audio")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "quiet"] + input_args
        + ["-vn", "-acodec", "pcm_s16le",
           "-ac", "1", "-ar", str(SAMPLE_RATE), wav_path],
        check=True,
    )


def load_audio(wav_path):
    """Load WAV and peak-normalise to [-1, 1]."""
    import scipy.io.wavfile as wavmod
    rate, data = wavmod.read(wav_path)
    data = data.astype(np.float64)
    mx = np.abs(data).max()
    if mx > 0:
        data /= mx
    return rate, data


def cross_correlate_offset(ref_audio, other_audio, search_samples=None):
    """Return (lag_samples, confidence).

    Positive lag means *other* started recording AFTER *ref*.
    Uses scipy.signal.correlate(ref, other, mode='full').
    """
    if search_samples:
        a = ref_audio[:search_samples]
        b = other_audio[:search_samples]
    else:
        a = ref_audio
        b = other_audio

    from scipy.signal import correlate
    corr = correlate(a, b, mode="full")
    peak_idx = np.argmax(np.abs(corr))
    lag_samples = peak_idx - (len(b) - 1)

    max_corr = np.abs(corr[peak_idx])
    norm = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    confidence = max_corr / norm if norm > 0 else 0
    return lag_samples, confidence


def find_clap_peak(audio_data, sample_rate, search_seconds=15):
    """Find the loudest transient in the first N seconds."""
    search_range = int(sample_rate * search_seconds)
    chunk = np.abs(audio_data[:search_range])
    window = int(sample_rate * 0.005)  # 5 ms smoothing
    smoothed = np.convolve(chunk, np.ones(window) / window, mode="same")
    peak_sample = np.argmax(smoothed)
    peak_amplitude = smoothed[peak_sample]
    return int(peak_sample), float(peak_amplitude)


def get_video_metadata(video_path):
    """Extract ffprobe metadata."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", "-show_streams", video_path],
        capture_output=True, text=True,
    )
    try:
        data = json.loads(result.stdout)
        meta = {}
        for s in data.get("streams", []):
            if s["codec_type"] == "video":
                meta["width"] = int(s["width"])
                meta["height"] = int(s["height"])
                meta["codec"] = s["codec_name"]
                meta["fps"] = s.get("r_frame_rate", "?")
                meta["creation_time"] = s.get("tags", {}).get("creation_time", "N/A")
                meta["timecode"] = s.get("tags", {}).get("timecode", "N/A")
                meta["nb_frames"] = s.get("nb_frames", "N/A")
            elif s["codec_type"] == "audio":
                meta["audio_codec"] = s["codec_name"]
                meta["audio_rate"] = s.get("sample_rate", "?")
                meta["audio_channels"] = s.get("channels", "?")
        meta["duration"] = float(data["format"]["duration"])
        meta["size_bytes"] = int(data["format"]["size"])
        meta["format_creation_time"] = data["format"].get("tags", {}).get("creation_time", "N/A")
        return meta
    except Exception as e:
        return {"error": str(e)}


# ─── MAIN ─────────────────────────────────────────────────────────
def main():
    args = parse_args()

    VIDEO_BASE = os.path.abspath(args.base)
    OUTPUT_BASE = os.path.join(VIDEO_BASE, "output")
    WORK_DIR = os.path.join(VIDEO_BASE, ".sync_work")
    NUM_CAMS = args.cams
    REF_CAM = args.ref_cam
    SEARCH_WINDOW_SEC = args.search_window
    PREVIEW_HEIGHT = args.preview_height
    PREVIEW_MAX_SEC = args.preview_max_sec

    if not (1 <= REF_CAM <= NUM_CAMS):
        print(f"ERROR: --ref-cam {REF_CAM} must be in 1..{NUM_CAMS}")
        raise SystemExit(1)

    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # ── 1. Discover and pair sessions ─────────────────────────────
    print("=" * 70)
    print("STEP 1: Discovering and pairing recording sessions")
    print("=" * 70)

    # Per-camera: discover raw files, probe durations + creation_times,
    # then group adjacent chapters (a single auto-split recording) into one
    # logical recording. After grouping, cam_recordings[cam] is a list whose
    # length is the number of *logical* recordings for that camera.
    cam_recordings = {}
    for cam in range(1, NUM_CAMS + 1):
        mp4s = sorted(glob.glob(os.path.join(VIDEO_BASE, str(cam), "*GX*.MP4")))
        if not mp4s:
            cam_recordings[cam] = []
            print(f"  Camera {cam}: 0 files")
            continue
        durations = []
        creation_times = []
        for f in mp4s:
            dur_out = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries",
                 "format=duration", "-of", "csv=p=0", f],
                capture_output=True, text=True,
            ).stdout.strip()
            if not dur_out:
                raise RuntimeError(f"ffprobe could not read duration from {f}")
            durations.append(float(dur_out))
            creation_times.append(probe_creation_time(f))
        groups = group_chapters(mp4s, durations, creation_times)
        cam_recordings[cam] = groups
        merged = sum(1 for g in groups if len(g["paths"]) > 1)
        print(f"  Camera {cam}: {len(mp4s)} files -> {len(groups)} logical "
              f"recording(s){' (' + str(merged) + ' merged from chapters)' if merged else ''}")
        for gi, g in enumerate(groups):
            names = [os.path.basename(p) for p in g["paths"]]
            ts = g["start_time"].isoformat() if g["start_time"] else "no-creation_time"
            print(f"    rec {gi+1}: {names}  {g['total_dur']:.2f}s  start={ts}")

    empty_cams = [cam for cam, recs in cam_recordings.items() if not recs]
    if empty_cams:
        print(f"\nERROR: no GX*.MP4 files found in folder(s) {empty_cams} "
              f"under {VIDEO_BASE}.")
        print("  Check --base, --cams, and that each camera's SD card has "
              "been copied into the matching numbered folder.")
        raise SystemExit(1)

    # Pair logical recordings across cameras by index (sorted earliest first).
    # If counts differ, truncate to the minimum and warn.
    counts = {cam: len(r) for cam, r in cam_recordings.items()}
    if len(set(counts.values())) > 1:
        print(f"\n  WARNING: cameras have different recording counts {counts}. "
              f"Truncating to the minimum.")
    num_sessions = min(counts.values())
    print(f"\n  -> {num_sessions} complete session(s) (all {NUM_CAMS} cameras present)")

    sessions = {}
    for s in range(num_sessions):
        session_name = f"session_{s + 1:02d}"
        session_files = {}
        for cam in range(1, NUM_CAMS + 1):
            rec = cam_recordings[cam][s]
            primary_name = os.path.basename(rec["paths"][0])
            session_files[cam] = {
                "paths": rec["paths"],
                "filename": primary_name,
                "filenames": [os.path.basename(p) for p in rec["paths"]],
                "duration": rec["total_dur"],
                "start_time": rec["start_time"],
            }

        for cam in range(1, NUM_CAMS + 1):
            # LRV proxy: only first chapter (preview is short anyway)
            mp4_name = session_files[cam]["filenames"][0]
            lrv_name = mp4_name.replace("GX", "GL").replace(".MP4", ".LRV")
            lrv_path = os.path.join(VIDEO_BASE, str(cam), lrv_name)
            session_files[cam]["lrv_path"] = lrv_path if os.path.exists(lrv_path) else None
            thm_name = mp4_name.replace(".MP4", ".THM")
            thm_path = os.path.join(VIDEO_BASE, str(cam), thm_name)
            session_files[cam]["thm_path"] = thm_path if os.path.exists(thm_path) else None

        sessions[session_name] = session_files

        dur_vals = [session_files[c]["duration"] for c in range(1, NUM_CAMS + 1)]
        dur_spread = max(dur_vals) - min(dur_vals)
        print(f"\n  {session_name}:")
        for cam in range(1, NUM_CAMS + 1):
            f = session_files[cam]
            chap_tag = "" if len(f["paths"]) == 1 else f" (+{len(f['paths'])-1} chapters)"
            print(f"    Cam {cam}: {f['filename']}{chap_tag}  {f['duration']:.2f}s  "
                  f"LRV={'Y' if f['lrv_path'] else 'N'}  THM={'Y' if f['thm_path'] else 'N'}")
        print(f"    Raw duration spread: {dur_spread:.2f}s {'!! LARGE SPREAD' if dur_spread > 5 else 'OK'}")

    # ── 2. Process each session ───────────────────────────────────
    for session_name, session_files in sessions.items():
        print(f"\n{'=' * 70}")
        print(f"PROCESSING: {session_name}")
        print(f"{'=' * 70}")

        session_out = os.path.join(OUTPUT_BASE, session_name)
        synced_dir = os.path.join(session_out, "synced_raw")
        meta_dir = os.path.join(session_out, "metadata")
        os.makedirs(synced_dir, exist_ok=True)
        os.makedirs(meta_dir, exist_ok=True)

        # Probe FPS from first camera's first chapter (all cams share settings)
        FPS = probe_video_fps(session_files[1]["paths"][0])
        print(f"\n  Video FPS: {FPS:.4f}")

        # Extract audio (concat chapters if multi-file)
        print("\n  Extracting audio...")
        audio_data = {}
        for cam in range(1, NUM_CAMS + 1):
            wav_path = os.path.join(WORK_DIR, f"{session_name}_cam{cam}.wav")
            extract_audio(
                session_files[cam]["paths"],
                wav_path,
                WORK_DIR,
                f"{session_name}_cam{cam}",
            )
            rate, data = load_audio(wav_path)
            audio_data[cam] = data
            print(f"    Cam {cam}: {len(data)} samples ({len(data) / rate:.2f}s)")

        # Cross-correlation sync
        print(f"\n  Cross-correlating (Cam {REF_CAM} = reference)...")
        ref_cam = REF_CAM
        search_samples = int(SAMPLE_RATE * SEARCH_WINDOW_SEC)
        offsets = {ref_cam: (0, 1.0)}

        for cam in range(1, NUM_CAMS + 1):
            if cam == ref_cam:
                continue
            lag, conf = cross_correlate_offset(
                audio_data[ref_cam], audio_data[cam], search_samples
            )
            offsets[cam] = (lag, conf)
            lag_ms = lag / SAMPLE_RATE * 1000
            lag_frames = lag / SAMPLE_RATE * FPS
            print(f"    Cam {ref_cam} -> Cam {cam}: {lag:+d} samples "
                  f"({lag_ms:+.2f}ms, {lag_frames:+.2f} frames) conf={conf:.4f}")

        # Clap/peak detection
        print("\n  Detecting transient peaks...")
        peaks_info = {}
        for cam in range(1, NUM_CAMS + 1):
            peak_sample, peak_amp = find_clap_peak(
                audio_data[cam], SAMPLE_RATE, SEARCH_WINDOW_SEC
            )
            peak_time = peak_sample / SAMPLE_RATE
            peaks_info[cam] = {
                "sample": peak_sample, "time_sec": peak_time, "amplitude": peak_amp
            }
            print(f"    Cam {cam}: peak at {peak_time:.4f}s "
                  f"(sample {peak_sample}, amp={peak_amp:.4f})")

        # Compute trim offsets
        # Positive cross-corr lag = other started AFTER ref.
        # Camera with LARGEST offset started EARLIEST -> trim most from its head.
        offset_sec = {cam: offsets[cam][0] / SAMPLE_RATE for cam in range(1, NUM_CAMS + 1)}
        max_offset = max(offset_sec.values())
        trim_sec = {cam: max_offset - offset_sec[cam] for cam in range(1, NUM_CAMS + 1)}

        effective_dur = {cam: session_files[cam]["duration"] - trim_sec[cam]
                         for cam in range(1, NUM_CAMS + 1)}
        common_dur = min(effective_dur.values())

        print("\n  Trim plan:")
        for cam in range(1, NUM_CAMS + 1):
            print(f"    Cam {cam}: trim {trim_sec[cam]:.4f}s from start, "
                  f"effective dur: {effective_dur[cam]:.2f}s")
        print(f"    Common synced duration: {common_dur:.2f}s "
              f"({common_dur * FPS:.0f} frames)")

        # Trim raw video (stream copy — no re-encode). Multi-chapter recordings
        # go through the concat demuxer; -ss is applied as input-side seek.
        print("\n  Trimming synced raw video...")
        for cam in range(1, NUM_CAMS + 1):
            inp_args = ffmpeg_input_args(
                session_files[cam]["paths"], WORK_DIR,
                f"{session_name}_cam{cam}_trim",
            )
            out = os.path.join(synced_dir, f"cam{cam}_synced.mp4")
            cmd = ["ffmpeg", "-y", "-v", "quiet"]
            if trim_sec[cam] > 0.0001:
                cmd += ["-ss", f"{trim_sec[cam]:.6f}"]
            cmd += inp_args + ["-t", f"{common_dur:.6f}", "-c", "copy", out]
            subprocess.run(cmd, check=True)
            out_size = os.path.getsize(out) / (1024 * 1024)
            print(f"    Cam {cam}: -> {os.path.basename(out)} ({out_size:.1f} MB)")

        # Side-by-side preview (3+2 grid from LRV files, capped at PREVIEW_MAX_SEC)
        print("\n  Generating synced preview...")
        preview_inputs = []
        for cam in range(1, NUM_CAMS + 1):
            # Preview is short (<= PREVIEW_MAX_SEC), so first chapter is enough.
            src = session_files[cam].get("lrv_path") or session_files[cam]["paths"][0]
            preview_inputs.append(src)

        filter_str = ""
        for idx, cam in enumerate(range(1, NUM_CAMS + 1)):
            ss = trim_sec[cam]
            cap = min(common_dur, PREVIEW_MAX_SEC)
            trim_f = (f"trim=start={ss:.6f}:duration={cap},setpts=PTS-STARTPTS,"
                      if ss > 0.0001 else
                      f"trim=duration={cap},setpts=PTS-STARTPTS,")
            filter_str += f"[{idx}:v]{trim_f}scale=-2:{PREVIEW_HEIGHT}[v{cam}];\n"

        # N-camera grid: bias toward a wider-than-tall layout so each pane
        # stays readable on a 16:9 screen. For N <= 3 use a single row; else
        # cols = ceil(N/2), rows = ceil(N/cols). Examples:
        #   N=2  -> 2x1   N=5  -> 3x2   N=8  -> 4x2   N=12 -> 6x2
        # Cam 1 is top-left; cameras fill row-major (left-to-right, top-to-bottom).
        import math
        if NUM_CAMS <= 3:
            grid_cols = NUM_CAMS
            grid_rows = 1
        else:
            grid_cols = math.ceil(NUM_CAMS / 2)
            grid_rows = math.ceil(NUM_CAMS / grid_cols)
        row_labels = []
        cam_iter = iter(range(1, NUM_CAMS + 1))
        for r in range(grid_rows):
            row_cams = []
            for _ in range(grid_cols):
                c = next(cam_iter, None)
                if c is None:
                    break
                row_cams.append(c)
            label = f"row{r}"
            if len(row_cams) == 1:
                # hstack=inputs=1 is invalid; pass through with a copy
                filter_str += f"[v{row_cams[0]}]copy[{label}_raw];\n"
            else:
                filter_str += "".join(f"[v{c}]" for c in row_cams)
                filter_str += f"hstack=inputs={len(row_cams)}[{label}_raw];\n"
            if len(row_cams) < grid_cols:
                # pad short last row to full width so vstack works
                filter_str += (f"[{label}_raw]pad=iw*{grid_cols}/{len(row_cams)}"
                               f":ih:0:0:black[{label}];\n")
            else:
                filter_str += f"[{label}_raw]copy[{label}];\n"
            row_labels.append(label)
        if len(row_labels) == 1:
            filter_str += f"[{row_labels[0]}]copy[out];\n"
        else:
            filter_str += "".join(f"[{lbl}]" for lbl in row_labels)
            filter_str += f"vstack=inputs={len(row_labels)}[out];\n"

        # Audio from the reference camera, trimmed to match the synced video
        a_ss = trim_sec[REF_CAM]
        a_dur = min(common_dur, PREVIEW_MAX_SEC)
        ref_audio_idx = REF_CAM - 1  # ffmpeg input index (preview_inputs is 0-indexed)
        if a_ss > 0.0001:
            filter_str += (f"[{ref_audio_idx}:a]atrim=start={a_ss:.6f}:duration={a_dur},"
                           f"asetpts=PTS-STARTPTS[aout]")
        else:
            filter_str += f"[{ref_audio_idx}:a]atrim=duration={a_dur},asetpts=PTS-STARTPTS[aout]"

        preview_path = os.path.join(session_out, f"{session_name}_preview.mp4")
        cmd = ["ffmpeg", "-y", "-v", "quiet"]
        for src in preview_inputs:
            cmd += ["-i", src]
        cmd += [
            "-filter_complex", filter_str,
            "-map", "[out]", "-map", "[aout]",
            "-t", str(a_dur),
            "-c:v", "libx264", "-preset", "fast", "-crf", "28",
            "-c:a", "aac", "-b:a", "128k",
            preview_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                prev_size = os.path.getsize(preview_path) / (1024 * 1024)
                print(f"    -> {os.path.basename(preview_path)} ({prev_size:.1f} MB)")
            else:
                print(f"    WARNING: preview failed, falling back to cam{REF_CAM} only")
                fallback = session_files[REF_CAM].get("lrv_path") or session_files[REF_CAM]["paths"][0]
                subprocess.run([
                    "ffmpeg", "-y", "-v", "quiet",
                    "-ss", f"{trim_sec[REF_CAM]:.6f}", "-i", fallback,
                    "-t", str(min(common_dur, PREVIEW_MAX_SEC)),
                    "-vf", f"scale=-2:{PREVIEW_HEIGHT}",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "28",
                    "-c:a", "aac", "-b:a", "128k",
                    preview_path,
                ], check=True)
        except subprocess.TimeoutExpired:
            print("    WARNING: preview generation timed out, skipping")

        # Save metadata
        print("\n  Saving metadata...")
        session_meta = {
            "session_name": session_name,
            "fps": FPS,
            "sample_rate": SAMPLE_RATE,
            "reference_camera": ref_cam,
            "common_duration_sec": common_dur,
            "common_duration_frames": int(common_dur * FPS),
            "cameras": {},
        }
        for cam in range(1, NUM_CAMS + 1):
            # Metadata is from the first chapter; record all chapter filenames.
            cam_meta = get_video_metadata(session_files[cam]["paths"][0])
            cam_meta["source_file"] = session_files[cam]["filename"]
            cam_meta["source_files"] = session_files[cam]["filenames"]
            cam_meta["trim_sec"] = trim_sec[cam]
            cam_meta["trim_frames"] = trim_sec[cam] * FPS
            cam_meta["offset_samples"] = offsets[cam][0]
            cam_meta["offset_ms"] = offsets[cam][0] / SAMPLE_RATE * 1000
            cam_meta["offset_frames"] = offsets[cam][0] / SAMPLE_RATE * FPS
            cam_meta["correlation_confidence"] = offsets[cam][1]
            cam_meta["clap_peak"] = peaks_info[cam]
            session_meta["cameras"][f"cam{cam}"] = cam_meta

            if session_files[cam]["thm_path"]:
                shutil.copy2(
                    session_files[cam]["thm_path"],
                    os.path.join(meta_dir,
                                 f"cam{cam}_{session_files[cam]['filename'].replace('.MP4', '.THM')}"),
                )

        meta_json_path = os.path.join(meta_dir, f"{session_name}_metadata.json")
        with open(meta_json_path, "w") as f:
            json.dump(session_meta, f, indent=2, default=str)
        print(f"    -> {os.path.basename(meta_json_path)}")

        # Sync report
        print("\n  Generating sync report...")
        report = []
        report.append(f"# Sync Report: {session_name}")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append("## Session Overview")
        report.append(f"- Reference camera: Cam {ref_cam}")
        report.append(f"- FPS: {FPS:.4f}")
        report.append(f"- Audio sample rate: {SAMPLE_RATE} Hz")
        report.append(f"- Common synced duration: {common_dur:.2f}s ({int(common_dur * FPS)} frames)")
        report.append(f"- Number of cameras: {NUM_CAMS}\n")

        report.append("## Source Files")
        report.append("| Camera | Source File | Original Duration | Trim Amount | Confidence |")
        report.append("|--------|------------|-------------------|-------------|------------|")
        for cam in range(1, NUM_CAMS + 1):
            f = session_files[cam]
            conf = offsets[cam][1]
            label = f["filename"]
            if len(f["paths"]) > 1:
                label += f" (+{len(f['paths'])-1} chapters)"
            report.append(
                f"| Cam {cam} | {label} | {f['duration']:.2f}s | "
                f"{trim_sec[cam]:.4f}s ({trim_sec[cam] * FPS:.2f}f) | {conf:.4f} |"
            )

        report.append(f"\n## Audio Sync Offsets (relative to Cam {ref_cam})")
        report.append("| Camera | Samples | Milliseconds | Frames | Confidence |")
        report.append("|--------|---------|-------------|--------|------------|")
        for cam in range(1, NUM_CAMS + 1):
            lag = offsets[cam][0]
            lag_ms = lag / SAMPLE_RATE * 1000
            lag_frames = lag / SAMPLE_RATE * FPS
            conf = offsets[cam][1]
            report.append(f"| Cam {cam} | {lag:+d} | {lag_ms:+.2f}ms | {lag_frames:+.2f} | {conf:.4f} |")

        report.append("\n## Clap/Transient Detection")
        report.append("| Camera | Peak Time | Peak Sample | Amplitude |")
        report.append("|--------|-----------|-------------|-----------|")
        for cam in range(1, NUM_CAMS + 1):
            p = peaks_info[cam]
            report.append(f"| Cam {cam} | {p['time_sec']:.4f}s | {p['sample']} | {p['amplitude']:.4f} |")

        report.append("\n## Sanity Checks")
        non_ref = [c for c in range(1, NUM_CAMS + 1) if c != REF_CAM]
        confidences = [offsets[cam][1] for cam in non_ref]
        min_conf = min(confidences)
        max_offset_ms = max(abs(offsets[cam][0]) / SAMPLE_RATE * 1000
                            for cam in non_ref)
        dur_spread = max(session_files[c]["duration"] for c in range(1, NUM_CAMS + 1)) - \
                     min(session_files[c]["duration"] for c in range(1, NUM_CAMS + 1))
        checks = [
            ("Cross-correlation confidence > 0.3",
             "PASS" if min_conf > 0.3 else "FAIL", f"min={min_conf:.4f}"),
            ("Max offset < 5000ms",
             "PASS" if max_offset_ms < 5000 else "WARN", f"{max_offset_ms:.1f}ms"),
            ("Raw duration spread < 10s",
             "PASS" if dur_spread < 10 else "WARN", f"{dur_spread:.2f}s"),
            ("All cameras present",
             "PASS" if len(session_files) == NUM_CAMS else "FAIL",
             f"{len(session_files)}/{NUM_CAMS}"),
        ]
        report.append("| Check | Result | Detail |")
        report.append("|-------|--------|--------|")
        for name, res, detail in checks:
            mark = "PASS" if res == "PASS" else ("WARN" if res == "WARN" else "FAIL")
            report.append(f"| {name} | {mark} | {detail} |")

        report.append(f"\n## Output Files")
        report.append("```")
        report.append(f"{session_name}/")
        report.append("  synced_raw/        # Frame-synced raw video (stream-copied)")
        for cam in range(1, NUM_CAMS + 1):
            report.append(f"    cam{cam}_synced.mp4")
        report.append(f"  {session_name}_preview.mp4  # Side-by-side preview (first {PREVIEW_MAX_SEC}s)")
        report.append("  metadata/          # Sync metadata + thumbnails")
        report.append(f"    {session_name}_metadata.json")
        report.append("    cam*_*.THM")
        report.append("```")

        report_path = os.path.join(session_out, f"{session_name}_sync_report.md")
        with open(report_path, "w") as f:
            f.write("\n".join(report))
        print(f"    -> {os.path.basename(report_path)}")
        print(f"\n  Done: {session_name}")

    # ── Final summary ─────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("ALL SESSIONS COMPLETE")
    print(f"{'=' * 70}")
    print(f"Output directory: {OUTPUT_BASE}")
    for sn in sessions:
        print(f"  {sn}/")
    print("\nDone!")


if __name__ == "__main__":
    main()
