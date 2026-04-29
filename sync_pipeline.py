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
    p.add_argument("--cams", type=int, default=5, help="Number of cameras")
    p.add_argument("--search-window", type=int, default=15, help="Seconds of audio to search for clap")
    p.add_argument("--preview-height", type=int, default=360, help="Preview per-camera height in px")
    p.add_argument("--preview-max-sec", type=int, default=30, help="Max preview duration in seconds")
    return p.parse_args()


# ─── CONSTANTS ────────────────────────────────────────────────────
SAMPLE_RATE = 48000


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
def extract_audio(video_path, wav_path):
    """Extract mono 48 kHz WAV from video."""
    subprocess.run([
        "ffmpeg", "-y", "-v", "quiet",
        "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(SAMPLE_RATE),
        wav_path
    ], check=True)


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
    SEARCH_WINDOW_SEC = args.search_window
    PREVIEW_HEIGHT = args.preview_height
    PREVIEW_MAX_SEC = args.preview_max_sec

    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # ── 1. Discover and pair sessions ─────────────────────────────
    print("=" * 70)
    print("STEP 1: Discovering and pairing recording sessions")
    print("=" * 70)

    cam_files = {}
    for cam in range(1, NUM_CAMS + 1):
        mp4s = sorted(glob.glob(os.path.join(VIDEO_BASE, str(cam), "*GX*.MP4")))
        cam_files[cam] = mp4s
        print(f"  Camera {cam}: {len(mp4s)} recordings — {[os.path.basename(f) for f in mp4s]}")

    # Pair by recording order (not filename — cameras may have different numbering)
    empty_cams = [cam for cam, mp4s in cam_files.items() if not mp4s]
    if empty_cams:
        print(f"\nERROR: no GX*.MP4 files found in folder(s) {empty_cams} "
              f"under {VIDEO_BASE}.")
        print("  Check --base, --cams, and that each camera's SD card has "
              "been copied into the matching numbered folder.")
        raise SystemExit(1)
    num_sessions = min(len(v) for v in cam_files.values())
    print(f"\n  -> {num_sessions} complete sessions (all {NUM_CAMS} cameras present)")

    sessions = {}
    for s in range(num_sessions):
        session_name = f"session_{s + 1:02d}"
        session_files = {}
        for cam in range(1, NUM_CAMS + 1):
            fpath = cam_files[cam][s]
            fname = os.path.basename(fpath)
            dur_out = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries",
                 "format=duration", "-of", "csv=p=0", fpath],
                capture_output=True, text=True,
            ).stdout.strip()
            if not dur_out:
                raise RuntimeError(f"ffprobe could not read duration from {fpath}")
            dur = float(dur_out)
            session_files[cam] = {"path": fpath, "filename": fname, "duration": dur}

        for cam in range(1, NUM_CAMS + 1):
            mp4_name = session_files[cam]["filename"]
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
            print(f"    Cam {cam}: {f['filename']}  {f['duration']:.2f}s  "
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

        # Probe FPS from first camera's video (all cams share same settings)
        FPS = probe_video_fps(session_files[1]["path"])
        print(f"\n  Video FPS: {FPS:.4f}")

        # Extract audio
        print("\n  Extracting audio...")
        audio_data = {}
        for cam in range(1, NUM_CAMS + 1):
            wav_path = os.path.join(WORK_DIR, f"{session_name}_cam{cam}.wav")
            extract_audio(session_files[cam]["path"], wav_path)
            rate, data = load_audio(wav_path)
            audio_data[cam] = data
            print(f"    Cam {cam}: {len(data)} samples ({len(data) / rate:.2f}s)")

        # Cross-correlation sync
        print("\n  Cross-correlating (Cam 1 = reference)...")
        ref_cam = 1
        search_samples = int(SAMPLE_RATE * SEARCH_WINDOW_SEC)
        offsets = {ref_cam: (0, 1.0)}

        for cam in range(2, NUM_CAMS + 1):
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

        # Trim raw video (stream copy — no re-encode)
        print("\n  Trimming synced raw video...")
        for cam in range(1, NUM_CAMS + 1):
            inp = session_files[cam]["path"]
            out = os.path.join(synced_dir, f"cam{cam}_synced.mp4")
            cmd = ["ffmpeg", "-y", "-v", "quiet"]
            if trim_sec[cam] > 0.0001:
                cmd += ["-ss", f"{trim_sec[cam]:.6f}"]
            cmd += ["-i", inp, "-t", f"{common_dur:.6f}", "-c", "copy", out]
            subprocess.run(cmd, check=True)
            out_size = os.path.getsize(out) / (1024 * 1024)
            print(f"    Cam {cam}: -> {os.path.basename(out)} ({out_size:.1f} MB)")

        # Side-by-side preview (3+2 grid from LRV files, capped at PREVIEW_MAX_SEC)
        print("\n  Generating synced preview...")
        preview_inputs = []
        for cam in range(1, NUM_CAMS + 1):
            src = session_files[cam].get("lrv_path") or session_files[cam]["path"]
            preview_inputs.append(src)

        filter_str = ""
        for idx, cam in enumerate(range(1, NUM_CAMS + 1)):
            ss = trim_sec[cam]
            cap = min(common_dur, PREVIEW_MAX_SEC)
            trim_f = (f"trim=start={ss:.6f}:duration={cap},setpts=PTS-STARTPTS,"
                      if ss > 0.0001 else
                      f"trim=duration={cap},setpts=PTS-STARTPTS,")
            filter_str += f"[{idx}:v]{trim_f}scale=-2:{PREVIEW_HEIGHT}[v{cam}];\n"

        filter_str += "[v1][v2][v3]hstack=inputs=3[top];\n"
        filter_str += "[v4][v5]hstack=inputs=2[bot_raw];\n"
        filter_str += "[bot_raw]pad=iw*3/2:ih:0:0:black[bot];\n"
        filter_str += "[top][bot]vstack=inputs=2[out];\n"

        # Audio from cam1 (reference), trimmed to match the synced video
        a_ss = trim_sec[1]
        a_dur = min(common_dur, PREVIEW_MAX_SEC)
        if a_ss > 0.0001:
            filter_str += (f"[0:a]atrim=start={a_ss:.6f}:duration={a_dur},"
                           f"asetpts=PTS-STARTPTS[aout]")
        else:
            filter_str += f"[0:a]atrim=duration={a_dur},asetpts=PTS-STARTPTS[aout]"

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
                print(f"    WARNING: preview failed, falling back to cam1 only")
                fallback = session_files[1].get("lrv_path") or session_files[1]["path"]
                subprocess.run([
                    "ffmpeg", "-y", "-v", "quiet",
                    "-ss", f"{trim_sec[1]:.6f}", "-i", fallback,
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
            cam_meta = get_video_metadata(session_files[cam]["path"])
            cam_meta["source_file"] = session_files[cam]["filename"]
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
            report.append(
                f"| Cam {cam} | {f['filename']} | {f['duration']:.2f}s | "
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
        confidences = [offsets[cam][1] for cam in range(2, NUM_CAMS + 1)]
        min_conf = min(confidences)
        max_offset_ms = max(abs(offsets[cam][0]) / SAMPLE_RATE * 1000
                            for cam in range(2, NUM_CAMS + 1))
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
