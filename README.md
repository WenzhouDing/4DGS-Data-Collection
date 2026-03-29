# GoPro Multi-Camera 3D Vision Rig

Frame-accurate sync and intrinsic calibration pipeline for a multi-camera GoPro Hero 10 rig used in 3D vision work.

## What This Does

**Sync pipeline** — Takes raw footage from N GoPro cameras that were started manually (no genlock), finds the audio clap in each recording via cross-correlation, and trims all cameras to a common timeline. Outputs frame-synced raw video (stream-copied, no re-encode), a side-by-side preview grid, per-session metadata JSON, and a sync report with sanity checks.

**Calibration pipeline** — Extracts frames from a session where a checkerboard was visible, auto-detects the board geometry, and runs OpenCV's `calibrateCamera()` per camera. Outputs per-camera intrinsic matrices, distortion coefficients, and a combined calibration file.

## Directory Layout

```
.
├── sync_pipeline.py          # Multi-session audio sync
├── run_calibration.py        # Checkerboard intrinsic calibration
├── requirements.txt
├── gopro_hero10_3d_rig_config.txt   # Camera settings + QR code reference
├── 1/                        # Camera 1 raw footage (gitignored)
│   ├── GX010004.MP4
│   ├── GL010004.LRV
│   └── ...
├── 2/                        # Camera 2 raw footage
├── ...
└── output/                   # All pipeline output (gitignored)
    ├── session_01/
    │   ├── synced_raw/
    │   │   ├── cam1_synced.mp4
    │   │   └── ...
    │   ├── session_01_preview.mp4
    │   ├── session_01_sync_report.md
    │   └── metadata/
    │       ├── session_01_metadata.json
    │       └── *.THM
    ├── session_02/
    ├── ...
    └── calibration/
        ├── calibration_all_cameras.json
        ├── cam1_intrinsics.json
        ├── ...
        └── checkerboard_config.json
```

Raw footage folders (`1/`, `2/`, ...) and `output/` are gitignored — only the scripts, config, and docs are tracked.

## Prerequisites

- Python 3.8+
- ffmpeg and ffprobe on PATH
- GoPro Labs firmware on all cameras (for QR code configuration)

```bash
pip install -r requirements.txt
```

## Camera Setup

All cameras must share identical settings so that footage is directly comparable. The file `gopro_hero10_3d_rig_config.txt` has the full parameter list and a GoPro Labs QR code command string you can flash to every camera:

```
mVr4p60e0!NfW0thS0dR0aScFd1b1w45i4M4S180x0sLa
```

Key settings: 4K 60fps, Wide lens, Flat color, 10-bit, ISO locked at 400, 180-degree shutter, Hypersmooth off.

Generate a scannable QR at:
```
https://gopro.github.io/labs/control/set/?cmd=mVr4p60e0!NfW0thS0dR0aScFd1b1w45i4M4S180x0sLa
```

See the config file for per-camera naming QR codes (`!MBASE`) and time sync options (Precision Time QR, GPS).

## Shooting Workflow

1. Flash the QR code to every camera (identical settings).
2. Optionally flash per-camera naming QR codes (`CAM01`, `CAM02`, ...).
3. Start all cameras recording.
4. **Clap once** clearly within the first ~10 seconds — this is the sync reference.
5. Shoot the scene.
6. Stop all cameras.
7. Copy each camera's SD card into its own numbered folder (`1/`, `2/`, etc.).

## Running the Sync Pipeline

From the project root (the directory containing folders `1/`, `2/`, ...):

```bash
python sync_pipeline.py --base . --cams 5
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--base` | `.` | Directory containing camera folders `1/` through `N/` |
| `--cams` | `5` | Number of cameras |
| `--search-window` | `15` | Seconds of audio to search for the clap |
| `--preview-height` | `360` | Per-camera height in the preview grid (px) |
| `--preview-max-sec` | `30` | Max duration of the preview clip |

The pipeline:
1. Discovers GoPro MP4s in each camera folder and pairs them **by recording order** (not filename — cameras may start numbering differently).
2. Extracts audio, runs `scipy.signal.correlate()` against Camera 1 as reference.
3. Computes per-camera trim offsets and a common duration.
4. Stream-copies (`-c copy`) each camera's video with the computed trim — no quality loss.
5. Generates a 3+2 grid preview from LRV proxy files (falls back to scaled MP4 if LRV missing).
6. Writes a sync report with sanity checks (confidence, offset magnitude, duration spread).

### How Sync Works

Each camera is started manually, so recording start times differ by several seconds. A single clap provides a sharp audio transient visible in all recordings. Cross-correlation finds the precise sample offset between Camera 1's audio and every other camera. The camera that started earliest gets the most trimmed from its head; the one that started latest gets trimmed least. After trimming, frame 0 in all cameras corresponds to the same physical moment.

Key detail: `scipy.signal.correlate(ref, other)` returns a positive lag when `other` started **after** `ref`. The trim formula is `trim[cam] = max_offset - offset[cam]` — the camera with the largest offset started earliest and needs the most cut.

## Running Calibration

Record a session where a checkerboard is clearly visible from all cameras, then sync it first:

```bash
python sync_pipeline.py --base . --cams 5
python run_calibration.py --base . --session session_04 --square-size 0.03
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--base` | `.` | Project root |
| `--session` | `session_04` | Which synced session to calibrate from |
| `--cams` | `5` | Number of cameras |
| `--square-size` | `0.03` | Checkerboard square side in metres |
| `--fps-extract` | `0.5` | Frame extraction rate (0.5 = one frame every 2 s) |

The script auto-detects the checkerboard inner-corner count by trying several candidates. It uses subpixel corner refinement and reports RMS reprojection error per camera.

Output lands in `output/calibration/`:
- `camN_intrinsics.json` — camera matrix K, distortion coefficients, optimal undistortion matrix, per-frame errors, frames used.
- `calibration_all_cameras.json` — all cameras combined with checkerboard metadata.
- `checkerboard_config.json` — board geometry for downstream stereo calibration.

## GoPro File Types

| Extension | Description |
|-----------|-------------|
| `GX*.MP4` | Main 4K video |
| `GL*.LRV` | Low-res proxy (used for preview generation) |
| `*.THM`   | JPEG thumbnail |

File numbering differs across cameras — that's why pairing is done by recording order, not filename.

## Next Steps (Not Yet Implemented)

- **Stereo / extrinsic calibration** — Use the intrinsics + a shared checkerboard session to compute relative camera poses via `cv2.stereoCalibrate()`.
- **Undistortion** — Apply the calibrated intrinsics to remove lens distortion before 3D reconstruction.
- **Dense matching / depth** — Feed undistorted, synced frames into a stereo or multi-view stereo pipeline.

## License

Unlicensed — private project.
