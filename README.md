# GoPro Multi-Camera 3D Vision Rig

Synchronization, validated intrinsic/extrinsic calibration, and undistorted video export for a multi-camera GoPro Hero 10 rig used in 3D vision work.

## What This Does

**Sync pipeline** (`sync_pipeline.py`) — Takes raw footage from N GoPro cameras that were started manually (no genlock), finds the audio clap in each recording via cross-correlation, and trims all cameras to a common timeline. Outputs frame-synced raw video (stream-copied, no re-encode), a side-by-side sync preview with audio that retains lens distortion, per-episode metadata JSON, and a sync report with sanity checks. FPS is probed from the actual video (supports 60fps, 120fps, etc.).

**Calibration pipeline** (`run_calibration.py`) — Two-phase calibration from a synced checkerboard episode:

1. **Intrinsic calibration** — Reads cameras in lockstep, detects checkerboards in parallel on downscaled frames, then refines corners at full resolution. It globally separates training, model-selection, and final-test frames, fits several OpenCV lens-model candidates, and requires both a valid distortion mapping and validation observations before selecting a model. Outputs include K, distortion coefficients, image dimensions, fit errors, and validation diagnostics.

2. **Extrinsic calibration** — Uses shared training frames with fixed intrinsics, calibrates viable camera pairs, and bridges through intermediate cameras where necessary. Outputs use the usual OpenCV **world-to-camera** convention: `X_cam = R @ X_ref + T`, with the reference camera defining world coordinates. Fundamental matrices always map reference-image points to target-image epipolar lines.

Both phases share the same corner detections. Source frame numbers identify corresponding frames in already-synchronized videos; reading in lockstep does not correct any residual synchronization error. Model-selection and final-test frames never enter intrinsic or stereo parameter fitting, and the final-test set is also excluded from model selection. Calibration is built in a staging directory and published only when every required camera and final-test geometry check passes. A failed run retains diagnostic artifacts and leaves an existing published calibration unchanged.

**Video export** (`export_calibrated_videos.py`) — Applies validated lens calibration and any measured residual frame offsets to produce `synced_undistorted/` and matching `calibration_undistorted/`. The exported videos keep the source resolution and frame rate; their calibration retains K and OpenCV extrinsics and sets distortion to zero. `synced_raw/` remains the original, lens-distorted intermediate. The episode organizer runs this export by default.

**Undistorted preview** (`make_undistorted_preview.py`) — Builds a labeled grid of every camera from the completed undistorted export. Every tile uses the same sampled frame indices, with audio from the saved reference camera. The organizer generates this preview after each episode's export.

## Directory Layout

```
.
├── sync_pipeline.py                # Multi-episode audio sync
├── run_calibration.py              # Intrinsic + extrinsic calibration (with bridging)
├── run_eval_epipolar.py            # Epipolar-geometry validation
├── export_calibrated_videos.py     # Synchronized, undistorted videos + matching calibration
├── make_undistorted_preview.py     # All-camera grid from completed undistorted videos
├── calibration_geometry.py        # Convention, projection, and lens validity helpers
├── calibration_fit.py             # Lens candidates and held-out model selection
├── calibration_observations.py    # Source-checked detections and global frame split
├── calibration_frame_offsets.py  # Validated logical-to-source frame offsets
├── calibration_validation.py     # Reuse checks and backed-up episode copies
├── migrate_calibration.py         # Backed-up migration of legacy extrinsics
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
    ├── episode_0001/                # KITTI-style self-contained episode
    │   ├── synced_raw/              # Synchronized intermediate; lens distortion remains
    │   │   ├── cam1_synced.mp4 ... camN_synced.mp4
    │   ├── synced_undistorted/      # Final videos, with lens correction and frame offsets applied
    │   ├── calibration_undistorted/ # Matching K, zero distortion, unchanged OpenCV extrinsics
    │   ├── episode_0001_preview.mp4 # Initial sync preview; lens distortion remains
    │   ├── episode_0001_undistorted_preview.mp4 # Final all-camera preview
    │   ├── episode_0001_undistorted_preview.jpg # Preview poster
    │   ├── episode_0001_undistorted_preview.json # Inputs, frame mapping, and verification
    │   ├── episode_0001_sync_report.md
    │   ├── sync_adjustments.json   # written by sync_vis.py (optional)
    │   ├── metadata/
    │   │   ├── episode_0001_metadata.json
    │   │   └── *.THM
    │   └── calibration/             # per-episode calibration (KITTI-style;
    │       ├── cam1_intrinsics.json ... camN_intrinsics.json
    │       ├── cam{i}_extrinsics.json   # world-to-camera R,T; reference has identity pose
    │       ├── calibration_all_cameras.json
    │       ├── checkerboard_config.json
    │       ├── frame_extraction_log.json   # source frame numbers used per cam
    │       ├── observation_split.json     # training, selection, and final-test frames
    │       ├── cam{i}_model_selection.json # candidate fits and selection diagnostics
    │       ├── heldout_epipolar.json      # untouched final-test pair measurements
    │       ├── camera_poses.html       # written by viz_calibration.py
    │       └── validation/
    │           ├── cam1/ ... camN/
    │           │   ├── corners_frame_*.jpg
    │           │   └── reproj_error_per_frame.png
    │           ├── rms_all_cameras.png
    │           └── stereo/
    │               ├── pair_{REF}_*_cam*.jpg
    │               └── stereo_rms.png
    └── episode_0002/ ...             # other episodes can share calibration —
                                      # organize_episodes.sh duplicates the
                                      # calibration/ folder into each episode
                                      # in a group (see "Episode Grouping").
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
5. Generates an initial N-camera sync preview from LRV proxy files (falls back to scaled MP4 if LRV missing), retaining lens distortion and using at most 30 seconds by default. Layout is biased wider-than-tall: `N≤3` → single row, otherwise `cols = ceil(N/2)`, `rows = ceil(N/cols)` (so 12 → 6×2, 5 → 3×2, 8 → 4×2). Cam 1 is top-left, filling row-major. This initial preview can fall back to the reference camera alone if the grid fails; it does not show the later undistorted export.
6. Writes a sync report with sanity checks (confidence, offset magnitude, duration spread).

### How Sync Works

Each camera is started manually, so recording start times differ by several seconds. A single clap provides a sharp audio transient visible in all recordings. Cross-correlation finds the precise sample offset between the reference camera's audio (`--ref-cam`, default cam 1) and every other camera. The camera that started earliest gets the most trimmed from its head; the one that started latest gets trimmed least. After trimming, frame 0 in all cameras corresponds to the same physical moment.

Key detail: `scipy.signal.correlate(ref, other)` returns a positive lag when `other` started **after** `ref`. The trim formula is `trim[cam] = max_offset - offset[cam]` — the camera with the largest offset started earliest and needs the most cut.

## Episode Grouping (`organize_episodes.sh`)

KITTI-style packaging: each selected `output/episode_NNNN/` keeps synchronized source video, metadata, and calibration, and exports final `synced_undistorted/` videos with matching `calibration_undistorted/`. Some recordings reuse calibration from another episode of the same shoot. `organize_episodes.sh` runs synchronization, calibrates each group's source episode, copies validated calibration into the other group members, and then exports every selected episode exactly once, including when an episode appears in multiple groups. It creates that episode's all-camera undistorted preview immediately after the export succeeds.

```bash
./organize_episodes.sh --base . --cams 12 --board 9x12 --square-size 0.03 \
    --ref-cam 3 --groups "1" "2,3,4"
```

Each `--groups` arg is a comma-separated list of episode indices; the **first index in each group is the calibration source** for the entire group. So `--groups "1" "2,3,4"` means: episode 1 calibrates from itself; episodes 2, 3, and 4 share calibration captured during episode 2.

| Flag | Default | Description |
|------|---------|-------------|
| `--base` | `.` | Project root |
| `--cams` | *required* | Number of cameras |
| `--board` | *required* | Checkerboard size, e.g. `9x12` |
| `--square-size` | *required* | Square side in metres |
| `--ref-cam` | `1` | Reference camera for calibration |
| `--max-frames` | `0` | Calibration `--max-frames` (0 = no cap) |
| `--groups` | *required* | One arg per group; comma-separated episode indices, first is calib source |
| `--skip-sync` | off | Don't run `sync_pipeline.py` (assume episodes already synced) |
| `--skip-calib` | off | Reuse only existing calibration that passes validation; fail without copying if missing/invalid |
| `--skip-export` | off | Skip final video export and its undistorted preview for a calibration-only workflow |

The final reconstruction inputs are each selected episode's `synced_undistorted/` and `calibration_undistorted/`. Keep those together. An export or undistorted-preview failure stops the organizer with a nonzero status; `--skip-calib` still exports and makes the preview when existing calibration passes validation. Add `--skip-export` to perform only synchronization/calibration work.

Existing source calibration is checked before reuse: camera coverage, intrinsics and full-sensor lens validity, explicit conventions, matching individual/combined files, directional F, and passing independent final-test evidence. Without `--skip-calib`, an invalid source is recalibrated; with it, the script stops before copying. Destination copies are staged and validated, and existing calibration is preserved in a `calibration.backup-*` directory. Migration alone cannot make an invalid lens model or missing validation evidence pass these checks.

## Exporting Synchronized, Undistorted Video

To export an episode whose `calibration/` has already passed validation:

```bash
uv run python export_calibrated_videos.py --base . --episode episode_0004 --cams 10
```

The exporter reads `synced_raw/` and `calibration/`, then writes `synced_undistorted/` and `calibration_undistorted/`. Source videos and raw calibration stay intact. It exports the common valid frame range at the source resolution and frame rate. Use the matching output calibration, whose distortion coefficients are zero; applying the original distortion coefficients again would distort the corrected images.

The current exporter supports 8-bit input video and writes 8-bit output. It rejects 10-bit input instead of silently reducing its bit depth; the camera-settings section above does not imply 10-bit export support.

Output K equals each camera's original calibrated K. Keeping that image canvas crops some wide-angle edge content; the exporter does not estimate a wider-field replacement K. Camera poses retain the OpenCV `X_cam = R @ X_ref + T` convention. This is lens undistortion, not pairwise stereo rectification: calibrated multi-view reconstruction can use these videos and camera poses directly. Rectification is needed only for downstream algorithms that specifically require rectified stereo images.

Residual timing corrections use `source_frame = logical_frame + offset`. Saved `source_frame_offsets` are inferred only when the calibration's `source_episode` matches the exported episode. A copied calibration does not establish another episode's timing. To apply offsets measured for that episode, supply an explicit file:

```bash
uv run python export_calibrated_videos.py --base . --episode episode_0004 --cams 10 \
    --frame-offsets episode_0004_offsets.json
```

The file is a JSON mapping such as `{"cam4": -1, "cam6": -1}`; omitted cameras use zero. Use values measured from that episode rather than copying these example values. The processed calibration resets `source_frame_offsets` to zero because those shifts are already applied to the exported videos. Its `export_timeline` preserves the applied `frame_offsets`, starting logical frame, frame count, and FPS.

`--encoder auto` selects HEVC `hevc_videotoolbox` on macOS and H.264 `libx264` elsewhere. Use `--encoder hevc_videotoolbox` or `--encoder libx264` to select one explicitly.

### All-Camera Undistorted Preview

After a complete export, generate or refresh its preview with:

```bash
uv run python make_undistorted_preview.py --base . --episode episode_0004
```

The script infers camera count from `synced_undistorted/export_manifest.json`, verifies the completed videos against that manifest, and requires a shared frame count and FPS. It reads the exported videos directly, so their lens correction and timing offsets are already applied. No additional calibration or offset file is needed. `--decode-accel auto` uses VideoToolbox on macOS; `none` and `videotoolbox` select decoding explicitly.

Every camera appears in numeric order, left to right and then top to bottom, with a `CAM N` label. Tiles are 640×360, preserving aspect ratio with padding when needed. Ten cameras produce a 5×2, 3200×720 H.264 grid. Sampling uses the same source frame indices in every tile and limits preview FPS to at most 30; for 119.88 FPS sources, it selects frames 0, 4, 8, … at 30000/1001 FPS (about 29.97). The preview covers the full exported clip and copies audio from the calibration's reference camera when available (cam3 for episode 4).

Outputs are `output/<episode>/<episode>_undistorted_preview.mp4`, a matching JPEG poster, and a JSON record of camera order, source signatures, frame mapping, and verified output dimensions/FPS. Episode 4's full camera videos remain 3840×2160 at 119.88 FPS; only the viewing preview is reduced in size and frame rate. The earlier `<episode>_preview.mp4` is the separate, lens-distorted sync diagnostic.

## Running Calibration

Record an episode where a checkerboard is clearly visible from all cameras, then sync it first:

```bash
uv run python sync_pipeline.py --base . --cams 12
uv run python run_calibration.py --board 9x12 --square-size 0.03 --base . --episode episode_0001 --cams 12 --ref-cam 3
```

| Flag | Default | Description |
|------|---------|-------------|
| `--board` | *required* | Board size as COLSxROWS in squares, e.g. `9x12` |
| `--square-size` | *required* | Checkerboard square side in metres (e.g. `0.03` for 30 mm) |
| `--base` | `.` | Project root |
| `--episode` | `episode_0001` | Which synced episode to calibrate from |
| `--cams` | `12` | Number of cameras |
| `--ref-cam` | `1` | Reference camera (its frame is the world origin; all extrinsics are expressed in it) |
| `--every` | `30` | Process every Nth frame (30 = ~4fps at 120fps, 1 = all) |
| `--max-frames` | `60` | Stop after this many frames where ALL cameras detected the board (0 = no limit) |
| `--corners-cache` | *none* | Save or reuse corner detections after validating the source videos and detection configuration |
| `--output-dir` | `<episode>/calibration` | Optional calibration destination; publication still requires all checks to pass |
| `--frame-offsets` | *all zero* | JSON integer offsets; source frame = logical frame + camera offset |

### Phase 1: Intrinsic Calibration

All cameras advance to the same source frame together. For each sampled frame (`--every`, default 30), `cv2.findChessboardCornersSB` runs in parallel via `ThreadPoolExecutor`; 4K images are downscaled to approximately 960 pixels wide for detection, followed by full-resolution `cornerSubPix` refinement. Early stopping fires after `--max-frames` shared detections, or scanning ends at the available footage. The board size is specified in squares: `--board 9x12` means 8×11 inner corners.

Source frames are split globally across all cameras. Every fifth distinct frame in the union of detected observations becomes a final-test frame. Every fifth frame in the remaining pool becomes a model-selection frame, leaving approximately 64% training, 16% selection, and 20% final testing. No reserved frame enters intrinsic or stereo parameter fitting; final-test frames never enter model selection either. `observation_split.json` records the exact split.

Each camera needs at least five training and three model-selection detections. Training detections go through `cv2.calibrateCamera()` using `opencv5`, `opencv4` (k3 fixed), `rational_low`, `rational6`, and `rational8` candidates. A candidate must pass the distortion validity screen and have model-selection reprojection RMS at most 2 px; among scores within 0.05 px of the best, fewer free distortion coefficients are preferred. Rational candidates also must have no positive-radius denominator poles, even outside the sampled sensor domain: a nearly cancelled pole can hide unstable extrapolation behind a low fit error. This conservative rule also applies when reusing calibration. `camN_model_selection.json` records candidates and rejection reasons. A small training RMS alone cannot qualify a model. The selected intrinsics include these core fields:

```json
{
  "image_size": [3840, 2160],
  "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "dist": [k1, k2, p1, p2, k3],
  "rms_error_px": 0.2855
}
```

`K` contains focal lengths fx/fy and principal point cx/cy in pixels. OpenCV coefficient order is `[k1, k2, p1, p2]` for four coefficients, `[k1, k2, p1, p2, k3]` for five, and `[k1, k2, p1, p2, k3, k4, k5, k6]` for rational models. The full JSON also records the selected model and diagnostics. The validity screen checks for lens-mapping folds and failed inverse mapping across the sensor. Passing this numerical screen does not establish physical accuracy outside observed checkerboard coverage.

Use `--corners-cache PATH` to save/reuse a `.npz` detection cache. Reuse checks source video names, sizes, modification times, board dimensions, and camera IDs. A cache retains its original sampled observations; use a new cache when changing sampling settings. Use `--output-dir PATH` when evaluating a separate calibration destination. `run_calibration.py` estimates calibration; `export_calibrated_videos.py` produces the corresponding undistorted video, and the organizer invokes both stages by default.

For a measured residual timing offset, pass a JSON file such as `{"cam4": -1, "cam6": -1}` through `--frame-offsets`. At logical frame n, these cameras read source frame n−1; omitted cameras use zero. Keys must be `camN`, values must be integers, and reads are restricted to the common valid video range. This changes logical frame selection only: raw videos are not trimmed or rewritten. The detection cache checks offsets on reuse; changed offsets require a new cache. Combined calibration and split metadata record all cameras' offsets under `source_frame_offsets`, preserving the original `source_episode` when calibration is copied.

Runs build their outputs in `.calibration-staging-*` beside the destination. Publication requires valid intrinsics for every camera and a passing final check for every reference→target pair: at least three measurable final-test frames, mean epipolar error below 2 px, p95 below 5 px, and no invalid points. Results are in `heldout_epipolar.json`; no high residuals are discarded. Failed runs retain their outputs as `TARGET.failed-TIMENS`, including `camN_failure.json` when a camera fit fails. A successful replacement preserves the previous calibration as `TARGET.backup-TIMENS`.

### Phase 2: Extrinsic (Stereo) Calibration

Using the training corners from Phase 1, the script finds shared frames where both cameras detected the board at the same source frame number. Withheld frames remain excluded. The number of available training pairs therefore depends on detections, the validation split, and footage length; `--max-frames 60` does not guarantee 60 training observations per pair.

For each candidate pair, `cv2.stereoCalibrate()` runs with `CALIB_FIX_INTRINSIC`, solving rotation and translation using the selected K and distortion parameters. The final saved transform maps **reference/world coordinates into the target camera**, including when pair transforms must be reversed or chained.

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
  "convention": "world_to_camera",
  "R": [[...], [...], [...]],
  "T": [tx, ty, tz],
  "F": [[...], [...], [...]],
  "stereo_rms_px": 0.928,
  "baseline_m": 0.281,
  "euler_deg": {"rx": -0.87, "ry": 0.01, "rz": -0.29},
  "euler_rotation_order": "Rz @ Ry @ Rx",
  "method": "direct",
  "path": [3, 5],
  "path_rms": [0.928]
}
```

For column vectors, the coordinate definitions are:

```python
X_cam = R @ X_ref + T       # saved OpenCV world-to-camera transform
C_ref = -R.T @ T           # camera optical center in reference/world coordinates
X_ref = R.T @ X_cam + C_ref
```

`T` is the reference origin's position in target-camera coordinates, in metres. Each camera uses +X right, +Y down, +Z forward. `baseline_m = ‖T‖ = ‖C_ref‖`. The reference camera itself has `R = I`, `T = 0`, and no stereo `F`.

`F` uses undistorted pixel coordinates and satisfies `x_target.T @ F @ x_ref = 0`. It is recomputed from K and the final reference→target transform for every pair. This fixes the legacy direction bug for targets numbered below the reference: with cam3 as reference, the original `(1,3)` and `(2,3)` matrices mapped in the opposite direction. Reversing a pair requires transposing F as well as inverting its rigid transform.

`stereo_rms_px` is the direct stereo fit RMS, or the maximum link RMS for a bridged path. `method`, `path`, and `path_rms` retain that provenance. `euler_deg` describes the saved world-to-camera rotation with `R = Rz(rz) @ Ry(ry) @ Rx(rx)`; use R itself for calculations.

`calibration_all_cameras.json` combines intrinsics, extrinsics, and checkerboard parameters. New files declare `"schema_version": 2`, `"extrinsics_convention": "world_to_camera"`, and a `"reference_camera"` such as `"cam3"`. Every extrinsics block also declares `"convention": "world_to_camera"`.

### Migrating Existing Calibration

Readers explicitly support the repository's legacy files: a missing extrinsics `convention` means **camera-to-world**. Unknown conventions, contradictory metadata, and conflicting reference cameras are rejected. To rewrite existing JSONs into the new convention:

```bash
uv run python migrate_calibration.py --calibration-dir output/episode_0004/calibration
uv run python viz_calibration.py --base . --episode episode_0004
```

The migration validates inputs first, backs up the entire calibration directory to a timestamped sibling `calibration_backup_*`, converts legacy R/T, recomputes every target F, and updates individual and combined JSONs consistently. Repeating it is a no-op; it never inverts an already tagged world-to-camera transform a second time. Intrinsic files remain byte-for-byte unchanged. Migration does **not** refit distortion, improve calibration observations, or regenerate old error charts and epipolar overlays. Preserve the backup and rerun validation or calibration as needed.

## 3D Camera-Pose Visualizer (`viz_calibration.py`)

Reads either supported calibration convention, converts world-to-camera extrinsics to camera centers/axes for drawing, and renders frusta in the reference frame. Hover tooltips show K, intrinsic/stereo RMS, baseline, and bridging paths. Changing the stored convention preserves the displayed rig geometry.

```bash
# Generate camera_poses.html in EVERY episode's calibration folder
uv run python viz_calibration.py --base .

# Or render just one episode (and optionally open in a browser)
uv run python viz_calibration.py --base . --episode episode_0002 --show
```

`organize_episodes.sh` invokes the all-episodes form after video export, so calibrated episodes have `camera_poses.html` next to their raw-camera calibration JSONs. This rendering step also runs when `--skip-export` is used.

| Flag | Default | Description |
|------|---------|-------------|
| `--base` | `.` | Project root |
| `--episode` | *all* | Only this episode (otherwise loops over every `output/episode_*/` with calibration) |
| `--out` | `<episode>/calibration/camera_poses.html` | Output HTML path (only with `--episode`) |
| `--show` | off | Open in browser after generating (only with `--episode`) |
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

**Intrinsic RMS** summarizes the fitted observations, not full-frame lens validity. Inspect held-out errors and distortion diagnostics together with it: a subpixel fit can still fold near image edges. Large residuals can indicate poor board visibility, blur, an unsuitable model, or insufficient observations.

**Stereo RMS** should be under ~1 px for a well-calibrated pair. A large stereo RMS (like 20+ px) indicates a problem — common causes: the board wasn't fully visible to both cameras simultaneously, there's a sync error, or the camera was at a very oblique angle to the board. For **bridged pairs** the reported `stereo_rms_px` is the worst link in the chain (`max(path_rms)`); inspect `path_rms` in the JSON if a bridged pair looks borderline — one weak intermediate hop dominates the score.

**Baselines** are the Euclidean distance from the reference camera (`--ref-cam`) to each other camera, in metres. They should match your physical rig geometry — e.g. for a linear 5-cam rig with 10cm spacing and `--ref-cam 1`, expect baselines `0.10, 0.20, 0.30, 0.40 m`; with `--ref-cam 3` expect `0.20, 0.10, 0.10, 0.20 m` (cam1, cam2, cam4, cam5). The 12-cam 6×2 grid will produce per-pair baselines that match the in-rig distance from your `--ref-cam` to each other camera.

**Euler angles** should be small if cameras are roughly parallel. Large rotations (> 5-10 degrees) might indicate a tilted camera or a calibration issue.

### Validation Output

The `validation/` folder contains visual sanity checks:

**Intrinsic validation** (`validation/cam{N}/`):
- `corners_frame_*.jpg` — 4 sample frames per camera showing detected corners (green, from `drawChessboardCorners`) overlaid with reprojected corners (red circles). Green and red should overlap tightly.
- `reproj_error_per_frame.png` — per-frame RMS in pixels: `sqrt(sum(dx² + dy²) / N)`, equivalently L2 norm divided by `sqrt(N)`. Colors are green at ≤0.5 px, amber above 0.5 through 1 px, and red above 1 px. The legacy chart divided by N and understated RMS by `sqrt(88) ≈ 9.38` for this board; its old 0.06/0.1 thresholds are obsolete. Overall calibration RMS from OpenCV was not affected by that chart bug.

**Intrinsic summary** (`validation/rms_all_cameras.png`) — cross-camera RMS comparison bar chart.

**Stereo validation** (`validation/stereo/`):
- `pair_{REF}_{N}_frame_*_cam{REF}.jpg` and `pair_{REF}_{N}_frame_*_camN.jpg` — epipolar line overlays on 2 sample shared frames per pair (`REF` = `--ref-cam`). For each coloured dot (a detected checkerboard corner), the corresponding epipolar line is drawn in the other camera's image. The dot in the second image should sit on or very near the line. Large deviations mean the stereo geometry is off.
- `stereo_rms.png` — cross-pair stereo RMS bar chart. Outlier pairs are immediately visible.

## Standalone Epipolar Eval (`run_eval_epipolar.py`)

A post-calibration check that re-detects checkerboards and measures **point-to-line epipolar distance** for each reference→target pair. It reads saved intrinsics/extrinsics, validates lens mappings, recomputes F in the correct direction, and reports mean, median, p95, and maximum distance. Point undistortion checks numerical convergence by projecting back to the original pixels. Its video sampling can overlap calibration frames; the calibration pipeline's separate final-test split provides the explicitly held-out check.

```bash
uv run python run_eval_epipolar.py --base . --episode episode_0001 --cams 12
```

| Flag | Default | Description |
|------|---------|-------------|
| `--base` | `.` | Project root |
| `--episode` | `episode_0001` | Synced episode to evaluate |
| `--calibration-dir` | `<episode>/calibration` | Read another calibration directory without activating it for the episode |
| `--cams` | `12` | Number of cameras |
| `--ref-cam` | *infer from calibration* | Optional explicit reference; a mismatch with saved calibration fails |
| `--board` | `9x12` | Board size as COLSxROWS in squares |
| `--num-frames` | `10` | Frames to sample (uniformly across the middle 10%–90% of the video) |
| `--frame-indices` | *automatic sampling* | Explicit ascending, unique, nonnegative logical frame indices; overrides `--num-frames` |
| `--frame-offsets` | *source metadata or zero* | Explicit JSON mapping; otherwise saved offsets apply only when `source_episode` matches the evaluated episode |
| `--out` | `<episode>/calibration/validation/epipolar_eval` | Output directory for reports and overlays |
| `--allow-invalid-intrinsics` | off | Measure usable points for diagnosis despite an invalid lens model; the result remains FAILED |

Output: `output/<episode>/calibration/validation/epipolar_eval/`

- `pair_{REF}_{N}_f<frame>_cam*.jpg` — annotated overlays (green ≤ 1px, yellow ≤ 2px, red > 2px)
- `epipolar_eval_summary.json` — every expected pair, measurements, status/reasons, detection coverage, failed reads, invalid points, high-error counts, and lens diagnostics

All finite residuals remain in the statistics, including errors above 10 pixels; high-error frames are counted rather than discarded. A pair with mean below 1 px is GOOD, from 1 to below 2 px is OK, and at least 2 px or any frame above 10 px is POOR. Invalid geometry/lens mappings are FAILED, and pairs without usable measurements are INCOMPLETE. Missing pairs remain visible in the report instead of disappearing. Overall quality reflects the worst pair and coverage; POOR, FAILED, and INCOMPLETE return nonzero exit codes. Diagnostic mode cannot turn invalid intrinsics into a passing calibration.

Frame-offset metadata describes timing in the calibration's source episode. Reusing episode 2 calibration for episode 4 does not imply that episode 4 has the same timing offset or has been resynchronized. The evaluator uses zero offsets across different episodes unless an explicit `--frame-offsets` file is supplied.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The suite covers independent projection geometry, invalid lens models, convention migration, globally separate cached observations, evaluator failures, publication rollback, validated episode reuse/copy, and positive/negative frame offsets. Tests use synthetic geometry and temporary files; they do not alter captured videos.

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
uv run python sync_vis.py --base . --episode episode_0001 --cams 12 --ref-cam 1
```

A browser tab opens automatically (Ctrl+C to stop the server). Architecture:
- **Flask** backend serves the synced MP4s with HTTP Range support — the browser only fetches what it needs.
- **HTML5 `<video>` elements** in a CSS grid, decoded via the browser's hardware video pipeline. By default the server serves a **720p H.264 proxy** per camera (lazily generated on first launch); the browser decodes these ~10× faster than 4K HEVC, which is what makes scrubbing across N cams smooth. Frame indices map 1-to-1 with the original full-res `cam{N}_synced.mp4` because the proxy preserves duration. `--no-proxy` falls back to serving the original (slower).
- **Per-cam seek queue**: only one outstanding `currentTime` set per `<video>` at a time; the latest target is queued and flushed on `seeked`. Prevents the freeze that rapid slider drags would otherwise cause.
- **Single-page UI** (no React, no build step) — just an embedded HTML/CSS/JS template in `sync_vis.py`.

| Flag | Default | Description |
|------|---------|-------------|
| `--base` | `.` | Project root |
| `--episode` | `episode_0001` | Synced episode to fine-tune |
| `--cams` | `12` | Number of cameras |
| `--ref-cam` | `1` | Reference camera (its offset stays at 0) |
| `--port` | `8765` | Local port for the web UI |
| `--no-browser` | off | Skip auto-opening a browser tab |
| `--no-proxy` | off | Serve original 4K HEVC instead of generating a 720p H.264 proxy |

Controls:
- **Frame slider** at the top — scrubs across the synced timeline
- **Audio-peak markers** above the slider — one tick per camera at its detected clap moment (cam-1 / ref tick is red); click any tick to jump to that frame. If sync is good, all ticks stack at the same position.
- **Per-camera ±2 ticks** — click `−2 / −1 / 0 / +1 / +2` for each camera; only that camera reseeks
- **Save adjustments** — writes `output/<episode>/sync_adjustments.json`; re-launching resumes from the saved values
- **Apply to videos** — re-trims `cam{N}_synced.mp4` in place using the current offsets, regenerates proxies, and resets offsets to 0 (the previously-applied values are kept under `previous_offsets` in the JSON for audit). Updates `metadata.json`'s trim values.

Keyboard shortcuts: <kbd>←</kbd>/<kbd>→</kbd> step ±1 frame · <kbd>shift</kbd>+arrow ±10 · <kbd>home</kbd>/<kbd>end</kbd> · <kbd>1</kbd>–<kbd>9</kbd> focus a cam's offset · <kbd>s</kbd> save.

Performance: on first launch the tool generates a 720p H.264 proxy per camera (parallel, ~30 s/cam). Subsequent launches reuse the cached proxies (proxy is regenerated only if its source `cam{N}_synced.mp4` is newer). Use `--no-proxy` to bypass and serve originals.

## Downstream Reconstruction

- **Multi-view reconstruction / 3DGS** — Use `synced_undistorted/` with `calibration_undistorted/`; the output pixels already have lens correction applied.
- **Optional stereo rectification** — For a method requiring rectified stereo pairs, compute pair-specific transforms from the saved K/R/T. This step is not required for general calibrated multi-view reconstruction and is not performed by the exporter.
- **Dense matching / depth** — Feed the exported synchronized frames and matching cameras into the chosen reconstruction pipeline.

## License

Unlicensed — private project.
