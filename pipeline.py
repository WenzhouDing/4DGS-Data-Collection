#!/usr/bin/env python3
"""Run synchronization, validated calibration, undistortion, and previews from saved settings."""
import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parent


def episode_name(value):
    if isinstance(value, bool):
        raise ValueError("Episode numbers must be positive integers")
    match = re.fullmatch(r"(?:episode_)?([0-9]+)", str(value))
    if not match or int(match[1]) < 1:
        raise ValueError(f"Invalid episode {value!r}; use a number such as 4")
    return f"episode_{int(match[1]):04d}"


def integer(value, label, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def keys(table, allowed, label):
    if not isinstance(table, dict):
        raise ValueError(f"{label} must be a TOML table")
    unknown = set(table) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown {label} setting(s): {', '.join(sorted(unknown))}")


@dataclass
class Settings:
    config: Path
    data_dir: Path
    cameras: int
    reference: int
    calibration_episode: str
    board: str
    square_size: float
    sample_every: int
    max_frames: int
    episodes: list
    frame_offsets: dict


def load_settings(path, episodes=None):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    keys(config, ["project", "calibration", "run", "frame_offsets"], "top-level")
    project, calibration, run = (config.get(name, {}) for name in ("project", "calibration", "run"))
    keys(project, ["data_dir", "cameras", "reference_camera"], "project")
    keys(calibration, ["episode", "board", "square_size_m", "sample_every", "max_frames"], "calibration")
    keys(run, ["episodes"], "run")
    cameras = integer(project.get("cameras"), "project.cameras", 2)
    reference = integer(project.get("reference_camera"), "project.reference_camera", 1)
    if reference > cameras:
        raise ValueError("project.reference_camera exceeds the camera count")
    data_dir = project.get("data_dir", ".")
    if not isinstance(data_dir, str) or not data_dir.strip():
        raise ValueError("project.data_dir must be a nonempty path")
    board = calibration.get("board", "")
    if not isinstance(board, str) or not re.fullmatch(r"[0-9]+x[0-9]+", board) or min(map(int, board.split("x"))) < 3:
        raise ValueError("calibration.board must count squares, e.g. '9x12' (at least 3 per side)")
    square = calibration.get("square_size_m")
    if type(square) not in (float, int) or not math.isfinite(square) or square <= 0:
        raise ValueError("calibration.square_size_m must be a positive finite measurement in metres")
    targets = episodes if episodes is not None else run.get("episodes")
    if not isinstance(targets, list) or not targets:
        raise ValueError("run.episodes must be a nonempty list, e.g. [4]")
    targets = list(dict.fromkeys(episode_name(value) for value in targets))
    offsets = config.get("frame_offsets", {})
    if not isinstance(offsets, dict):
        raise ValueError("frame_offsets must be a table of episode tables")
    normalized = {}
    for name, values in offsets.items():
        canonical = episode_name(name)
        if name != canonical:
            raise ValueError(f"Use [frame_offsets.{canonical}] for timing corrections")
        keys(values, [f"cam{i}" for i in range(1, cameras + 1)], f"frame_offsets.{name}")
        if any(type(value) is not int for value in values.values()):
            raise ValueError(f"frame_offsets.{name} values must be integers")
        normalized[name] = {f"cam{i}": values.get(f"cam{i}", 0) for i in range(1, cameras + 1)}
    return Settings(path, (path.parent / Path(data_dir).expanduser()).resolve(), cameras,
                    reference, episode_name(calibration.get("episode")), board, float(square),
                    integer(calibration.get("sample_every", 30), "calibration.sample_every", 1),
                    integer(calibration.get("max_frames", 0), "calibration.max_frames", 0), targets, normalized)


def required_episodes(settings):
    return list(dict.fromkeys([settings.calibration_episode, *settings.episodes]))


def synced_files(settings, episode):
    directory = settings.data_dir / "output" / episode / "synced_raw"
    return [directory / f"cam{i}_synced.mp4" for i in range(1, settings.cameras + 1)]


def missing_inputs(settings):
    missing = []
    for episode in required_episodes(settings):
        files = synced_files(settings, episode)
        if all(path.is_file() and path.stat().st_size > 0 for path in files):
            continue
        directory = settings.data_dir / "output" / episode
        if directory.exists() and any(directory.iterdir()):
            raise ValueError(f"{episode} has incomplete synchronized inputs. Restore all {settings.cameras} "
                             f"camN_synced.mp4 files in {directory / 'synced_raw'}, or move the incomplete episode "
                             "folder aside before retrying. Existing files were preserved.")
        missing.append(episode)
    return missing


def check_environment(cpu=False):
    problems = []
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            problems.append(f"Install {executable} and put it on PATH (both come with FFmpeg)")
    for module in ("cv2", "numpy", "scipy", "matplotlib", "plotly"):
        if importlib.util.find_spec(module) is None:
            problems.append(f"Missing Python dependency {module}; run 'uv sync' in {ROOT}")
    if not problems:
        encoders = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"], stderr=subprocess.DEVNULL, text=True)
        for encoder in ["libx264"] + (["hevc_videotoolbox"] if platform.system() == "Darwin" and not cpu else []):
            if not re.search(rf"\b{encoder}\b", encoders):
                problems.append(f"FFmpeg lacks {encoder}; install a build with this encoder" + (" or use --cpu" if encoder != "libx264" else ""))
        filters = subprocess.check_output(["ffmpeg", "-hide_banner", "-filters"], stderr=subprocess.DEVNULL, text=True)
        if any(not re.search(rf"\b{name}\b", filters) for name in ("xstack", "overlay", "scale", "select")):
            problems.append("FFmpeg needs xstack, overlay, scale, and select filters for the preview")
    if problems:
        raise ValueError("Environment needs attention:\n  " + "\n  ".join(problems))
    print("Environment OK: Python dependencies, FFmpeg, FFprobe, encoders and preview filters.")


def print_plan(settings, missing, cpu=False):
    print(f"Settings: {settings.config}\nData: {settings.data_dir}")
    print(f"Rig: {settings.cameras} cameras; reference cam{settings.reference}; board {settings.board} squares, {settings.square_size:g} m")
    print("1. " + ("Synchronize missing recordings: " + ", ".join(missing) if missing else "Reuse existing synchronized inputs"))
    print(f"2. Validate/reuse calibration from {settings.calibration_episode}; fit if needed")
    print("3. Export synced + undistorted videos: " + ", ".join(settings.episodes))
    print("4. Make all-camera previews and camera-pose viewers")
    print("Video processing: " + ("CPU" if cpu else "automatic (hardware on macOS)"))


class Runner:
    def __init__(self, log_path):
        self.log_path = Path(log_path)

    def run(self, script, *arguments):
        command = [sys.executable, str(ROOT / script), *map(str, arguments)]
        print(f"\n▶ {script}", flush=True)
        environment = {**os.environ, "OPENCV_OPENCL_DEVICE": "disabled", "PYTHONUNBUFFERED": "1"}
        with self.log_path.open("a") as log:
            log.write("\n$ " + shlex.join(command) + "\n")
            with subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, errors="replace") as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                code = process.wait()
            if code:
                raise RuntimeError(f"{script} failed (exit {code}). See {self.log_path}. "
                                   "Fix the reported issue, then rerun the same command to reuse completed stages.")


def verify_inputs(settings, episode, base=None):
    from export_calibrated_videos import probe_video
    episode_dir = (base or settings.data_dir) / "output" / episode
    metadata_path = episode_dir / "metadata" / f"{episode}_metadata.json"
    if metadata_path.is_file():
        validation = json.loads(metadata_path.read_text()).get("sync_validation")
        if validation is not None and (not isinstance(validation, dict) or validation.get("passed") is not True):
            raise ValueError(f"{episode}: saved synchronization validation failed; resolve the sync diagnostics before reuse")
    directory = episode_dir / "synced_raw"
    probes = [probe_video(directory / f"cam{i}_synced.mp4") for i in range(1, settings.cameras + 1)]
    if len({probe["fps"] for probe in probes}) != 1:
        raise ValueError(f"{episode}: camera frame rates differ")
    if len({(probe["width"], probe["height"]) for probe in probes}) != 1:
        raise ValueError(f"{episode}: camera frame sizes differ; use identical recording settings")


def synchronize_missing(settings, episodes, runner):
    if not episodes:
        return
    for index in range(1, settings.cameras + 1):
        directory = settings.data_dir / str(index)
        if not directory.is_dir() or not list(directory.glob("*GX*.MP4")):
            raise ValueError(f"Missing raw footage in {directory}. Copy this camera's GX*.MP4 recordings "
                             "there, or restore the synchronized episode inputs.")
    stage = Path(tempfile.mkdtemp(prefix=".pipeline-sync-", dir=settings.data_dir / "output"))
    for index in range(1, settings.cameras + 1):
        (stage / str(index)).symlink_to(settings.data_dir / str(index), target_is_directory=True)
    try:
        runner.run("sync_pipeline.py", "--base", stage, "--cams", settings.cameras,
                   "--ref-cam", settings.reference, "--episodes", *[int(name[8:]) for name in episodes])
        # Validate every requested episode before publishing any of them.
        for name in episodes:
            metadata = json.loads((stage / "output" / name / "metadata" / f"{name}_metadata.json").read_text())
            if metadata.get("sync_validation", {}).get("passed") is not True:
                raise ValueError(f"{name}: synchronization confidence did not pass")
            verify_inputs(settings, name, stage)
            destination = settings.data_dir / "output" / name
            if destination.exists() and any(destination.iterdir()):
                raise ValueError(f"Refusing to replace existing episode {destination}")
        for name in episodes:
            destination = settings.data_dir / "output" / name
            if destination.exists():
                destination.rmdir()
            (stage / "output" / name).rename(destination)
    except BaseException:
        print(f"Synchronization diagnostics retained: {stage}")
        raise
    else:
        shutil.rmtree(stage)


def calibration_matches(settings, directory):
    from calibration_validation import validate_calibration
    try:
        validate_calibration(directory, settings.cameras, settings.reference)
        combined = json.loads((directory / "calibration_all_cameras.json").read_text())
        board = combined["checkerboard"]
        if (board["board_squares"] != list(map(int, settings.board.split("x")))
                or not math.isclose(board["square_size_m"], settings.square_size, rel_tol=1e-10)
                or combined.get("source_episode") != settings.calibration_episode):
            return False
        expected = settings.frame_offsets.get(settings.calibration_episode, {})
        saved = combined.get("source_frame_offsets", {})
        return all(saved.get(f"cam{i}", 0) == expected.get(f"cam{i}", 0)
                   for i in range(1, settings.cameras + 1))
    except (ValueError, KeyError, TypeError, OSError):
        return False


def offset_arguments(settings, episode, work):
    path = work / f"{episode}_frame_offsets.json"
    configured = settings.frame_offsets.get(episode, {})
    offsets = {f"cam{i}": configured.get(f"cam{i}", 0) for i in range(1, settings.cameras + 1)}
    path.write_text(json.dumps(offsets, indent=2) + "\n")
    return ["--frame-offsets", str(path)]


def input_signatures(settings, episode):
    return {path.name: {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in synced_files(settings, episode)}


def run_pipeline(settings, cpu=False, recalibrate=False):
    missing = missing_inputs(settings)
    check_environment(cpu)
    print_plan(settings, missing, cpu)
    logs = settings.data_dir / "output" / "pipeline_logs"
    logs.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}-", dir=logs))
    runner = Runner(work / "run.log")
    print(f"Run log: {runner.log_path}", flush=True)
    synchronize_missing(settings, missing, runner)
    for name in required_episodes(settings):
        verify_inputs(settings, name)
    source = settings.data_dir / "output" / settings.calibration_episode / "calibration"
    state_path = settings.data_dir / "output/pipeline_state.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    if not isinstance(state, dict):
        raise ValueError(f"Invalid saved pipeline state: {state_path}")
    current_inputs = input_signatures(settings, settings.calibration_episode)
    previous = state.get(settings.calibration_episode)
    if previous is not None and not isinstance(previous, dict):
        raise ValueError(f"Invalid saved calibration-source state: {state_path}")
    changed_inputs = previous is not None and previous.get("inputs") != current_inputs
    if changed_inputs:
        print("Calibration-source videos changed since the previous run; refitting calibration.", flush=True)
    if recalibrate or changed_inputs or settings.calibration_episode in missing or not calibration_matches(settings, source):
        runner.run("run_calibration.py", "--base", settings.data_dir, "--episode", settings.calibration_episode,
                   "--cams", settings.cameras, "--ref-cam", settings.reference, "--board", settings.board,
                   "--square-size", settings.square_size, "--every", settings.sample_every,
                   "--max-frames", settings.max_frames,
                   *offset_arguments(settings, settings.calibration_episode, work))
        if not calibration_matches(settings, source):
            raise ValueError(f"Calibration did not pass validation: {source}. See {runner.log_path}")
    else:
        print(f"Calibration passed validation; reusing {source}", flush=True)
    source_bytes = (source / "calibration_all_cameras.json").read_bytes()
    if current_inputs != input_signatures(settings, settings.calibration_episode):
        raise ValueError("Calibration-source videos changed during this run; retry with stable inputs")
    state[settings.calibration_episode] = {"inputs": current_inputs,
                                          "calibration_sha256": hashlib.sha256(source_bytes).hexdigest()}
    temporary_state = work / "pipeline_state.json"
    temporary_state.write_text(json.dumps(state, indent=2) + "\n")
    temporary_state.replace(state_path)
    for name in settings.episodes:
        target = settings.data_dir / "output" / name / "calibration"
        if target != source:
            same = calibration_matches(settings, target) and (target / "calibration_all_cameras.json").read_bytes() == source_bytes
            if not same:
                runner.run("calibration_validation.py", "--calibration-dir", source,
                           "--cams", settings.cameras, "--ref-cam", settings.reference,
                           "--copy-to", target, "--source-episode", settings.calibration_episode)
        runner.run("export_calibrated_videos.py", "--base", settings.data_dir, "--episode", name,
                   "--cams", settings.cameras, *offset_arguments(settings, name, work),
                   *(["--encoder", "libx264", "--decode-accel", "none"] if cpu else []))
        runner.run("make_undistorted_preview.py", "--base", settings.data_dir, "--episode", name,
                   *(["--decode-accel", "none"] if cpu else []))
        runner.run("viz_calibration.py", "--base", settings.data_dir, "--episode", name)
    print("\nDone. Use each episode's videos with its matching zero-distortion calibration:")
    for name in settings.episodes:
        directory = settings.data_dir / "output" / name
        print(f"  {name}\n    Videos: {directory / 'synced_undistorted'}\n"
              f"    Calibration: {directory / 'calibration_undistorted'}\n"
              f"    Preview: {directory / (name + '_undistorted_preview.mp4')}")
    print(f"Run log: {runner.log_path}")


def print_status(settings):
    print(f"Data: {settings.data_dir}\nFile availability (run validates contents before reuse):")
    for name in required_episodes(settings):
        directory = settings.data_dir / "output" / name
        synced = sum(path.is_file() and path.stat().st_size > 0 for path in synced_files(settings, name))
        exported = sum((directory / "synced_undistorted" / f"cam{i}_synced_undistorted.mp4").is_file()
                       for i in range(1, settings.cameras + 1))
        calibration = (directory / "calibration/calibration_all_cameras.json").is_file()
        preview = (directory / f"{name}_undistorted_preview.mp4").is_file()
        role = "calibration source" if name == settings.calibration_episode else "output episode"
        print(f"  {name} ({role}): synced {synced}/{settings.cameras}; calibration {'present' if calibration else 'missing'}; "
              f"undistorted {exported}/{settings.cameras}; preview {'present' if preview else 'missing'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog="Default: run using pipeline.toml. Repeat the same command to reuse completed outputs.")
    parser.add_argument("command", nargs="?", choices=["run", "doctor", "status"], default="run")
    parser.add_argument("--config", type=Path, default=ROOT / "pipeline.toml", help="Saved settings (default: pipeline.toml beside this script)")
    parser.add_argument("--episode", nargs="+", help="Override output episodes, e.g. --episode 4 5")
    parser.add_argument("--dry-run", action="store_true", help="Show the plan without writing files or running stages")
    parser.add_argument("--cpu", action="store_true", help="Use software video encoding/decoding instead of macOS acceleration")
    parser.add_argument("--recalibrate", action="store_true", help="Refit the source calibration even when validated files exist")
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.config, args.episode)
        if args.command == "status":
            print_status(settings)
        elif args.dry_run:
            print_plan(settings, missing_inputs(settings), args.cpu)
            if args.recalibrate:
                print("Source calibration will be refitted (--recalibrate).")
            print("Dry run: no files changed.")
        elif args.command == "doctor":
            check_environment(args.cpu)
            print_plan(settings, missing_inputs(settings), args.cpu)
            print_status(settings)
        else:
            run_pipeline(settings, args.cpu, args.recalibrate)
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped. Rerun the same command to reuse completed stages.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
