# GoPro Multi-Camera 3D Vision Rig

Frame-accurate sync, intrinsic calibration, and extrinsic (stereo) calibration pipeline for a multi-camera GoPro Hero 10 rig used in 3D vision work.

## What This Does

**Sync pipeline** (`sync_pipeline.py`) — Takes raw footage from N GoPro cameras that were started manually (no genlock), finds the audio clap in each recording via cross-correlation, and trims all cameras to a common timeline. Outputs frame-synced raw video (stream-copied, no re-encode), a side-by-side preview grid with audio, per-session metadata JSON, and a sync report with sanity checks. FPS is probed from the actual video (supports 60fps, 120fps, etc.).

**Calibration pipeline** (`run_calibration.py`) — Two-phase calibration from a synced checkerboard session:

1. **Intrinsic calibration** — All cameras are read in lockstep (same frame numbers) via OpenCV. For each sampled frame, checkerboard detection runs in parallel across cameras using `findChessboardCornersSB` (sector-based, faster than classic) on downscaled frames (~960px wide), then refines at full resolution via `cornerSubPix`. Early stopping fires after `--max-frames` (default 60) frames where ALL cameras detected the board. Each camera's detections then go through `cv2.calibrateCamera()`. Outputs minimal intrinsics JSON (K matrix, distortion coefficients, image size, RMS error).

2. **Extrinsic calibration** — Finds frames where the checkerboard was detected in both the reference camera (configurable via `--ref-cam`) and each other camera, then runs `cv2.stereoCalibrate()` with fixed intrinsics. Because detection uses the same frame numbers across cameras (lockstep), shared frames are guaranteed to be truly time-synced. Outputs per-pair rotation, translation, essential/fundamental matrices, baseline distance, and stereo RMS — all expressed in the reference camera's coordinate frame (origin = `--ref-cam`).

Both phases share the same corner detections. Source frame numbers are embedded in filenames (e.g. `frame_000120.jpg` = source frame 120), so matching by filename guarantees temporal correspondence across synced cameras. The lockstep approach ensures all cameras process identical frame numbers — this is critical for stereo calibration accuracy. Detected frames go to a system temp directory and are automatically cleaned up.

## Directory Layout

```
.
├── sync_pipeline.py                # Multi-session audio sync
├── run_calibration.py              # Intrinsic + extrinsic calibration (with bridging)
├── run_eval_epipolar.py            # Epipolar-geometry validation
├── viz_calibration.py              # 3D camera-pose visualizer (Plotly HTML)
├── sync_vis.py                     # Interactive ±2-frame sync fine-tune
├── pyproject.toml                  # uv project metadata + deps
├── uv.lock                         # uv lockfile
├── gopro_hero10_3d_rig_config.txt  # Camera settings + QR code reference
├── 1/                              # Camera 1 raw footage (gitignored)
│   ├── GX010004.MP4
│   ├── GL010004.LRV
│   └── ...
├── 2/ ... N/                       # Camera 2–N raw footage (12 cams in this project)
└── output/                         # All pipeline output (gitignored)
    ├── session_01/
    │   ├── synced_raw/
    │   │   ├── cam1_synced.mp4 ... camN_synced.mp4
    │   ├── session_01_preview.mp4
    │   ├── session_01_sync_report.md
    │   ├── sync_adjustments.json   # written by sync_vis.py (optional)
    │   └── metadata/
    │       ├── session_01_metadata.json
    │       └── *.THM
    ├── session_02/ ...
    └── calibration/
        ├── cam1_intrinsics.json ... camN_intrinsics.json
        ├── cam{i}_extrinsics.json   # one per non-ref camera, in --ref-cam's frame
        ├── calibration_all_cameras.json
        ├── checkerboard_config.json
        ├── frame_extraction_log.json   # source frame numbers used per cam
        └── validation/
            ├── cam1/ ... camN/
            │   ├── corners_frame_*.jpg     # detected vs reprojected corners
            │   └── reproj_error_per_frame.png
            ├── rms_all_cameras.png         # intrinsic RMS comparison
            └── stereo/
                ├── pair_{REF}_*_cam*.jpg   # epipolar line overlays (REF = --ref-cam)
                └── stereo_rms.png          # stereo RMS comparison
```

Raw footage folders (`1/`, `2/`, ...) and `output/` are gitignored — only the scripts, config, and docs are tracked.

## Prerequisites

- Python 3.12+ (managed via [uv](https://docs.astral.sh/uv/))
- ffmpeg and ffprobe on PATH
- GoPro Labs firmware on all cameras (for QR code configuration)

```bash
uv sync   # creates .venv/ from pyproject.toml + uv.lock
```

Run any script with `uv run python <script>.py …` (or activate `.venv/` manually).

## Camera Setup

All cameras must share identical settings so that footage is directly comparable. The file `gopro_hero10_3d_rig_config.txt` has the full parameter list and a GoPro Labs QR code command string you can flash to every camera:

```
mVr4p120e0!NfW0thS0dR0aScFd1b1w45i4M4S180x0sLa
```

Key settings: 4K 120fps, Wide lens, Flat color, 10-bit, ISO locked at 400, 180-degree shutter, Hypersmooth off. The pipeline auto-detects the actual FPS from video metadata.

Generate a scannable QR at:
```
https://gopro.github.io/labs/control/set/?cmd=mVr4p120e0!NfW0thS0dR0aScFd1b1w45i4M4S180x0sLa
```

See the config file for per-camera naming QR codes (`!MBASE`) and time sync options (Precision Time QR, GPS).

## Shooting Workflow

1. Flash the QR code to every camera (identical settings).
2. Optionally flash per-camera naming QR codes (`CAM01`, `CAM02`, ...).
3. Start all cameras recording.
4. **Clap once** clearly within the first ~10 seconds — this is the sync reference.
5. Shoot the scene. For calibration, hold a checkerboard visible to all cameras.
6. Stop all cameras.
7. Copy each camera's SD card into its own numbered folder (`1/`, `2/`, etc.).

## Running the Sync Pipeline

From the project root (the directory containing folders `1/`, `2/`, ...):

```bash
uv run python sync_pipeline.py --base . --cams 12
```

| Flag | Default | Description |
|------|---------|-------------|
| `--base` | `.` | Directory containing camera folders `1/` through `N/` |
| `--cams` | `12` | Number of cameras |
| `--ref-cam` | `1` | Reference camera (cross-correlation reference + preview audio source) |
| `--search-window` | `15` | Seconds of audio to search for the clap |
| `--preview-height` | `360` | Per-camera height in the preview grid (px) |
| `--preview-max-sec` | `30` | Max duration of the preview clip |

The pipeline:
1. Discovers GoPro MP4s in each camera folder and pairs them **by recording order** (not filename — cameras may start numbering differently and may even have different file counts; mismatched extras are dropped).
2. Extracts audio, runs `scipy.signal.correlate()` against the reference camera (`--ref-cam`).
3. Computes per-camera trim offsets and a common duration.
4. Stream-copies (`-c copy`) each camera's video with the computed trim — no quality loss.
5. Generates an N-camera grid preview from LRV proxy files (falls back to scaled MP4 if LRV missing). Layout is biased wider-than-tall: `N≤3` → single row, otherwise `cols = ceil(N/2)`, `rows = ceil(N/cols)` (so 12 → 6×2, 5 → 3×2, 8 → 4×2). Cam 1 is top-left, filling row-major.
6. Writes a sync report with sanity checks (confidence, offset magnitude, duration spread).

### How Sync Works

Each camera is started manually, so recording start times differ by several seconds. A single clap provides a sharp audio transient visible in all recordings. Cross-correlation finds the precise sample offset between the reference camera's audio (`--ref-cam`, default cam 1) and every other camera. The camera that started earliest gets the most trimmed from its head; the one that started latest gets trimmed least. After trimming, frame 0 in all cameras corresponds to the same physical moment.

Key detail: `scipy.signal.correlate(ref, other)` returns a positive lag when `other` started **after** `ref`. The trim formula is `trim[cam] = max_offset - offset[cam]` — the camera with the largest offset started earliest and needs the most cut.

## Running Calibration

Record a session where a checkerboard is clearly visible from all cameras, then sync it first:

```bash
uv run python sync_pipeline.py --base . --cams 12
uv run python run_calibration.py --board 9x12 --square-size 0.03 --base . --session session_01 --cams 12 --ref-cam 3
```

| Flag | Default | Description |
|------|---------|-------------|
| `--board` | *required* | Board size as COLSxROWS in squares, e.g. `9x12` |
| `--square-size` | *required* | Checkerboard square side in metres (e.g. `0.03` for 30 mm) |
| `--base` | `.` | Project root |
| `--session` | `session_01` | Which synced session to calibrate from |
| `--cams` | `12` | Number of cameras |
| `--ref-cam` | `1` | Reference camera (its frame is the world origin; all extrinsics are expressed in it) |
| `--every` | `30` | Process every Nth frame (30 = ~4fps at 120fps, 1 = all) |
| `--max-frames` | `60` | Stop after this many frames where ALL cameras detected the board (0 = no limit) |

### Phase 1: Intrinsic Calibration

All cameras are read in lockstep — every camera advances to the same frame number together. For each sampled frame (`--every`, default 30), all cameras retrieve and decode the frame, then `cv2.findChessboardCornersSB` runs in parallel across cameras via `ThreadPoolExecutor`. For 4K frames, detection runs on a downscaled image (~960px wide), then corners are refined at full resolution via `cornerSubPix`. Early stopping fires after `--max-frames` (default 60) frames where ALL cameras detected the board — this guarantees enough shared frames for stereo calibration. Only frames where the checkerboard is detected are saved to disk. The board size is specified via `--board` (e.g. `9x12` = 9 columns × 12 rows of squares → 8×11 inner corners).

Detected frames go through `cv2.calibrateCamera()`. The output per camera is a minimal JSON:

```json
{
  "image_size": [3840, 2160],
  "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "dist": [k1, k2, p1, p2, k3],
  "rms_error_px": 0.2855
}
```

`K` is the 3x3 camera matrix (focal lengths fx/fy in pixels, principal point cx/cy). `dist` is the 5-coefficient distortion vector (radial k1/k2/k3, tangential p1/p2). These two are everything needed for undistortion via `cv2.undistort(img, K, dist)` or downstream stereo work. Anything else (optimal new camera matrix, undistort ROI) is recomputable from K and dist, so it's not stored.

### Phase 2: Extrinsic (Stereo) Calibration

Using the corners already detected in Phase 1, the script finds "shared frames" — frames where both the reference camera (`--ref-cam`) and camera N detected the board at the same source frame number. Because Phase 1 processes all cameras in lockstep on the same frame numbers, shared frames are guaranteed to be truly time-synced (same physical moment, board in same pose). With `--max-frames 60`, at least 60 such shared frames are guaranteed for every camera pair.

For each candidate pair, `cv2.stereoCalibrate()` runs with the `CALIB_FIX_INTRINSIC` flag — it trusts the per-camera K and dist from Phase 1 and only solves for the rotation R and translation T between cameras. The raw stereoCalibrate output is inverted so that R and T express **camN's pose in the reference camera's coordinate frame** (ref-cam = origin).

**Bridging for wide-baseline rigs.** With many cameras (e.g. a 12-cam 2×6 grid), some `(ref, N)` pairs rarely share enough frames — `stereoCalibrate` either skips for too few shared frames or returns a poor RMS. Phase 2 first computes **every viable `(i, j)` pair** (~`N(N-1)/2` calls; ~1 min added for 12 cams) and builds a graph of edges that meet a quality bar (`shared ≥ 8 frames` AND `rms ≤ 1.5 px`). For each non-ref camera, the script prefers the direct `(ref, N)` pair if it meets the quality bar; otherwise it BFS-searches the **shortest hop-count path** from `ref` to `N` through good edges and chains the transforms. Each per-cam JSON records the path used:

```json
"method": "bridged via 3->5->6",
"path": [3, 5, 6],
"path_rms": [0.928, 0.955]
```

The output per pair:

```json
{
  "reference": "cam3",
  "target": "cam5",
  "R": [[...], [...], [...]],
  "T": [tx, ty, tz],
  "E": [[...], [...], [...]],
  "F": [[...], [...], [...]],
  "stereo_rms_px": 0.4096,
  "baseline_m": 0.0999,
  "euler_deg": {"rx": -0.87, "ry": 0.01, "rz": -0.29},
  "shared_frames": 41
}
```

`R` is the 3x3 rotation of camN relative to the reference. `T` is camN's optical center position in the reference's coordinate frame (metres; OpenCV convention: +X right, +Y down, +Z forward from the reference's viewpoint). `baseline_m` is `‖T‖`. `E` and `F` are the essential and fundamental matrices (from the original stereoCalibrate, not inverted). `stereo_rms_px` is the stereo reprojection error. `euler_deg` decomposes R as `R = Rx(rx) · Ry(ry) · Rz(rz)` (extrinsic XYZ / intrinsic ZYX, applied to a column vector with Rz first) — for quick sanity checking only.

The `calibration_all_cameras.json` combines both intrinsics and extrinsics for all cameras in one file, alongside the checkerboard parameters. Each camera's `extrinsics` block also carries `method` and `path` so you can audit which pairs were direct vs. bridged.

## 3D Camera-Pose Visualizer (`viz_calibration.py`)

Reads `calibration_all_cameras.json` and renders every camera as a frustum in the reference camera's coordinate frame, with hover tooltips showing K, intrinsic/stereo RMS, baseline, and (for bridged cams) the chain of intermediates used.

```bash
uv run python viz_calibration.py --base . [--show]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--base` | `.` | Project root |
| `--out` | `output/calibration/camera_poses.html` | Output HTML path |
| `--show` | off | Open in browser after generating |
| `--frustum-depth` | `0.05` | Frustum length in metres (visualization scale only) |

The output is a self-contained HTML file (Plotly with embedded JS — works fully offline). Camera color-coding:

| Color | Meaning |
|---|---|
| 🔴 red | reference camera (origin) |
| 🟢 green | direct stereoCalibrate, RMS ≤ 1.5 px |
| 🟡 amber | bridged via intermediate cameras |
| ⚪ gray | direct used despite poor RMS (no bridge available) |

Use this to **qualitatively verify** the calibration: a well-calibrated 6×2 GoPro grid should appear as two evenly-spaced rows of frustums all roughly facing the same direction. Outlier baselines or bizarre orientations are immediately visible.

### What to Look For

**Intrinsic RMS** should be under ~0.5 px for GoPro Wide at 4K. Values above 1px suggest poor board visibility, motion blur, or too few frames.

**Stereo RMS** should be under ~1 px for a well-calibrated pair. A large stereo RMS (like 20+ px) indicates a problem — common causes: the board wasn't fully visible to both cameras simultaneously, there's a sync error, or the camera was at a very oblique angle to the board.

**Baselines** are the Euclidean distance from the reference camera (`--ref-cam`) to each other camera, in metres. They should match your physical rig geometry — e.g. for a linear 5-cam rig with 10cm spacing and `--ref-cam 1`, expect baselines `0.10, 0.20, 0.30, 0.40 m`; with `--ref-cam 3` expect `0.20, 0.10, 0.10, 0.20 m` (cam1, cam2, cam4, cam5). The 12-cam 6×2 grid will produce per-pair baselines that match the in-rig distance from your `--ref-cam` to each other camera.

**Euler angles** should be small if cameras are roughly parallel. Large rotations (> 5-10 degrees) might indicate a tilted camera or a calibration issue.

### Validation Output

The `validation/` folder contains visual sanity checks:

**Intrinsic validation** (`validation/cam{N}/`):
- `corners_frame_*.jpg` — 4 sample frames per camera showing detected corners (green, from `drawChessboardCorners`) overlaid with reprojected corners (red circles). Green and red should overlap tightly.
- `reproj_error_per_frame.png` — bar chart of per-frame reprojection error, color-coded green (< 0.06px) / yellow (< 0.1px) / red (> 0.1px) with a mean line.

**Intrinsic summary** (`validation/rms_all_cameras.png`) — cross-camera RMS comparison bar chart.

**Stereo validation** (`validation/stereo/`):
- `pair_{REF}_{N}_frame_*_cam{REF}.jpg` and `pair_{REF}_{N}_frame_*_camN.jpg` — epipolar line overlays on 2 sample shared frames per pair (`REF` = `--ref-cam`). For each coloured dot (a detected checkerboard corner), the corresponding epipolar line is drawn in the other camera's image. The dot in the second image should sit on or very near the line. Large deviations mean the stereo geometry is off.
- `stereo_rms.png` — cross-pair stereo RMS bar chart. Outlier pairs are immediately visible.

## Standalone Epipolar Eval (`run_eval_epipolar.py`)

A separate, post-calibration sanity check that re-detects the checkerboard on a different sample of synced-video frames and measures **point-to-line epipolar distance** for every pair `(ref-cam, camN)`. This is independent of the calibration step's own validation: it reads the saved `cam{N}_intrinsics.json` + `cam{N}_extrinsics.json`, recomputes `F` from `K1, K2, R, T`, undistorts both images and points, and reports `mean / median / p95 / max` of the per-corner epipolar distances.

```bash
uv run python run_eval_epipolar.py --base . --session session_01 --cams 12 --ref-cam 1
```

| Flag | Default | Description |
|------|---------|-------------|
| `--base` | `.` | Project root |
| `--session` | `session_01` | Synced session to evaluate |
| `--cams` | `12` | Number of cameras |
| `--ref-cam` | `1` | Must match the value used at calibration time |
| `--board` | `9x12` | Board size as COLSxROWS in squares |
| `--num-frames` | `10` | Frames to sample (uniformly across the middle 10%–90% of the video) |

Output: `output/calibration/validation/epipolar_eval/`
- `pair_{REF}_{N}_f<frame>_cam*.jpg` — annotated overlays (green ≤ 1px, yellow ≤ 2px, red > 2px)
- `epipolar_eval_summary.json` — per-pair statistics

Verdict: **mean < 1 px = good, < 2 px = acceptable, > 2 px = poor**. Frames whose worst point exceeds 10 px are auto-rejected (likely a bad detection or a partially occluded board).

## GoPro File Types

| Extension | Description |
|-----------|-------------|
| `GX*.MP4` | Main 4K video |
| `GL*.LRV` | Low-res proxy (used for preview generation) |
| `*.THM`   | JPEG thumbnail |

File numbering differs across cameras — that's why pairing is done by recording order, not filename. Within a single camera, GoPro auto-splits long recordings into chapter files (`GX01XXXX.MP4`, `GX02XXXX.MP4`, ...). The sync pipeline detects these via abutting `creation_time` tags and merges them into one logical recording (using ffmpeg's `concat` demuxer for audio extraction and the stream-copy trim).

## Sync Fine-Tune (`sync_vis.py`)

After running `sync_pipeline.py`, audio-clap sync is usually accurate to within one frame, but residual sub-frame drift can leave a camera 1–2 frames off the visually-correct moment. `sync_vis.py` runs a small **local web server** with a browser-based UI that lets you nudge each camera by ±2 frames using the macOS hardware HEVC decoder for fast, smooth scrubbing:

```bash
uv run python sync_vis.py --base . --session session_01 --cams 12 --ref-cam 1
```

A browser tab opens automatically (Ctrl+C to stop the server). Architecture:
- **Flask** backend serves the synced MP4s with HTTP Range support — the browser only fetches what it needs.
- **HTML5 `<video>` elements** in a CSS grid use the browser's hardware HEVC decoder. Setting `video.currentTime` is much faster than `cv2.VideoCapture` seeking, and the GPU compositor keeps the multi-camera grid smooth.
- **Single-page UI** (no React, no build step) — just an embedded HTML/CSS/JS template in `sync_vis.py`.

| Flag | Default | Description |
|------|---------|-------------|
| `--base` | `.` | Project root |
| `--session` | `session_01` | Synced session to fine-tune |
| `--cams` | `12` | Number of cameras |
| `--ref-cam` | `1` | Reference camera (its offset stays at 0) |
| `--port` | `8765` | Local port for the web UI |
| `--no-browser` | off | Skip auto-opening a browser tab |
| `--no-proxy` | off | Serve original 4K HEVC instead of generating a 720p H.264 proxy |

Controls:
- **Frame slider** at the top — scrubs across the synced timeline
- **Audio-peak markers** above the slider — one tick per camera at its detected clap moment (cam-1 / ref tick is red); click any tick to jump to that frame. If sync is good, all ticks stack at the same position.
- **Per-camera ±2 ticks** — click `−2 / −1 / 0 / +1 / +2` for each camera; only that camera reseeks
- **Save adjustments** — writes `output/<session>/sync_adjustments.json`; re-launching resumes from the saved values
- **Apply to videos** — re-trims `cam{N}_synced.mp4` in place using the current offsets, regenerates proxies, and resets offsets to 0 (the previously-applied values are kept under `previous_offsets` in the JSON for audit). Updates `metadata.json`'s trim values.

Keyboard shortcuts: <kbd>←</kbd>/<kbd>→</kbd> step ±1 frame · <kbd>shift</kbd>+arrow ±10 · <kbd>home</kbd>/<kbd>end</kbd> · <kbd>1</kbd>–<kbd>9</kbd> focus a cam's offset · <kbd>s</kbd> save.

Performance: on first launch the tool generates a 720p H.264 proxy per camera (parallel, ~30 s/cam). Browser HEVC decode of full-res 4K is slow on rapid scrubbing; H.264 720p decodes ~10× faster on macOS hardware. Proxy duration is preserved so frame indices map 1-to-1 with the original full-res `cam{N}_synced.mp4`. Use `--no-proxy` to bypass.

## Next Steps (Not Yet Implemented)

- **Undistortion** — Apply the calibrated intrinsics to remove lens distortion before 3D reconstruction.
- **Stereo rectification** — Use extrinsics to compute rectification transforms (`cv2.stereoRectify`) for aligned epipolar geometry.
- **Dense matching / depth** — Feed undistorted, rectified, synced frames into a stereo or multi-view stereo pipeline.

## License

Unlicensed — private project.
