# GoPro Hero 10 — Multi-Camera 3D Vision Project

## Full Context Dump for Agent Handoff

---

## 1. PROJECT OVERVIEW

The user has multiple GoPro Hero 10 cameras used for a 3D vision project. Two core challenges:
1. **Matching settings** across all cameras (ISO, exposure, white balance, etc.)
2. **Time-syncing** the cameras for frame-accurate multi-view capture

The shooting environment is **indoors with controlled lighting**.

---

## 2. CAMERA CONFIGURATION

All cameras must share identical settings. GoPro Labs firmware is required on every camera.

### QR Code Configuration (GoPro Labs)

**Final QR command string:**
```
mVr4p120e0!NfW0thS0dR0aScFd1b1w45i4M4S180x0sLa
```

**QR Code URL (scan with camera or open in browser):**
```
https://gopro.github.io/labs/control/set/?cmd=mVr4p120e0!NfW0thS0dR0aScFd1b1w45i4M4S180x0sLa
```

**QR Code Creator page (interactive config tool):**
```
https://gopro.github.io/labs/control/custom/
```

### Full Settings Breakdown

| Setting | Value | QR Code | Rationale |
|---|---|---|---|
| Camera Mode | Video | `mV` | — |
| Video Resolution | 4K (16:9) | `r4` | — |
| Frame Rate | 120 fps | `p120` | High temporal resolution for sync and motion capture |
| Hypersmooth | Off | `e0` | Stabilization crops/warps differently per camera — destroys stereo geometry |
| Lens / FOV | **Wide** | `fW` | User explicitly chose Wide over Linear. NOTE: Wide introduces barrel distortion — must apply lens undistortion in the 3D pipeline using GoPro intrinsic calibration params |
| Hindsight | Off | `hS0` | Not needed |
| Duration | Off | `dR0` | Records until manually stopped |
| Wind Reduction | Off | `!N` / `W1` | Indoor environment, no wind |
| Protune Color | Flat | `cF` | Maximum dynamic range, easiest to color-match in post |
| Color Depth | 10-bit | `d1` | More headroom for color grading |
| Bit Rate | High | `b1` | Max quality |
| White Balance | 4500K (fixed) | `w45` | General indoor default. Adjust to match actual lights: tungsten ~3200K, fluorescent ~4000K, daylight LED ~5500K |
| ISO Min | 400 | `i4` | Locked (min = max) for matched exposure across all cameras |
| ISO Max | 400 | `M4` | Same as min — locks ISO |
| Lock Shutter | 180° (= 1/240s at 120fps) | `S180` | Standard cinematic motion blur. For sharper frames (better for CV/feature matching), consider 90° or 45° |
| EV Compensation | 0 | `x0` | Neutral |
| Sharpness | Low | `sL` | In-camera sharpening varies per unit — do it in post |
| RAW Audio | Off | `a` | Not needed for vision project |

### Per-Camera File Naming

Flash a separate QR per camera to label output files:

```
Camera 01:  https://gopro.github.io/labs/control/set/?cmd=!MBASE=%22CAM01%22
Camera 02:  https://gopro.github.io/labs/control/set/?cmd=!MBASE=%22CAM02%22
Camera 03:  https://gopro.github.io/labs/control/set/?cmd=!MBASE=%22CAM03%22
Camera 04:  https://gopro.github.io/labs/control/set/?cmd=!MBASE=%22CAM04%22
Camera 05:  https://gopro.github.io/labs/control/set/?cmd=!MBASE=%22CAM05%22
Camera 06:  https://gopro.github.io/labs/control/set/?cmd=!MBASE=%22CAM06%22
Camera 07:  https://gopro.github.io/labs/control/set/?cmd=!MBASE=%22CAM07%22
Camera 08:  https://gopro.github.io/labs/control/set/?cmd=!MBASE=%22CAM08%22
Camera 09:  https://gopro.github.io/labs/control/set/?cmd=!MBASE=%22CAM09%22
Camera 10:  https://gopro.github.io/labs/control/set/?cmd=!MBASE=%22CAM10%22
Camera 11:  https://gopro.github.io/labs/control/set/?cmd=!MBASE=%22CAM11%22
Camera 12:  https://gopro.github.io/labs/control/set/?cmd=!MBASE=%22CAM12%22
```

Files become e.g. `CAM01GH01xxxx.MP4`.

---

## 3. TIME SYNCHRONIZATION

### Method 1: Precision Time QR (Indoor — Recommended)

Display animated QR on laptop screen, scan with each camera right before shooting. Syncs internal clocks to millisecond precision.

```
https://gopro.github.io/labs/control/precisiontime/
```

### Method 2: GPS Time Sync (Outdoor Only)

Hero 10 supports GPS-based timecode sync via GoPro Labs. If cameras have GPS lock, timecodes auto-align to atomic time. Not usable indoors.

### Method 3: Audio Clap Sync (Fallback / Verification)

Standard clapperboard approach. Do a sharp clap in view of all cameras after starting recording.

---

## 4. POST-PROCESSING FRAME SYNC PIPELINE

### Step 1 — Extract audio from all videos

```bash
ffmpeg -i camera_01.mp4 -vn -acodec pcm_s16le camera_01.wav
ffmpeg -i camera_02.mp4 -vn -acodec pcm_s16le camera_02.wav
# repeat for all cameras
```

### Step 2 — Detect clap spike in each audio track

```python
import numpy as np
import scipy.io.wavfile as wav

def find_clap_sample(wav_path, search_seconds=10):
    rate, data = wav.read(wav_path)
    if data.ndim > 1:
        data = data[:, 0]  # mono
    search_range = int(rate * search_seconds)
    chunk = np.abs(data[:search_range].astype(np.float64))
    clap_sample = np.argmax(chunk)
    return clap_sample, rate

# For each camera
for cam in cameras:
    sample, rate = find_clap_sample(f"{cam}.wav")
    clap_time = sample / rate
    print(f"{cam}: clap at {clap_time:.4f}s (sample {sample})")
```

### Step 3 — Compute frame offsets

```python
fps = 120  # recording fps
ref_clap_time = clap_times["camera_01"]

for cam, t in clap_times.items():
    delta_sec = t - ref_clap_time
    delta_frames = round(delta_sec * fps)
    print(f"{cam}: offset = {delta_frames} frames")
```

### Step 4 — Trim all videos to common start

```bash
# Example: camera_02 is 15 frames (0.25s) ahead of camera_01
ffmpeg -ss 0.25 -i camera_02.mp4 -c copy camera_02_synced.mp4
```

### Step 5 — Verify sync

Play back synced videos and confirm the clap frame aligns visually across all cameras. Do clap at end of episode too to check for drift.

### Accuracy Notes

- Audio sync at 48kHz sample rate → ~0.02ms per sample, well within one frame at 120fps (~8.3ms)
- Cross-correlation (`scipy.signal.correlate`) is more robust than peak detection in noisy environments
- GoPro Hero 10 has NO genlock — over long recordings (10+ min) cameras can drift by 1-2 frames. Clap at both ends to detect drift. If significant, resample/interpolate.
- Calibration reads synchronized videos in lockstep (the same source frame number for all cameras). This preserves correspondence but does not fix residual synchronization error. Detection uses parallel `findChessboardCornersSB` on downscaled images (~960px), with `cornerSubPix` refinement at full resolution.
- **Stored extrinsics use OpenCV world-to-camera coordinates**: `X_cam = R @ X_ref + T`, with `--ref-cam` defining the world frame. The optical center in world coordinates is `C_ref = -R.T @ T`; saved T is the world origin expressed in camera coordinates. Reference-camera extrinsics are identity/zero. Individual and nested extrinsics declare `convention: world_to_camera`; combined files declare `schema_version: 2`, `extrinsics_convention: world_to_camera`, and `reference_camera: camN`.
- Legacy untagged extrinsics mean camera-to-world and remain readable. `migrate_calibration.py --calibration-dir DIR` validates the input, backs up the entire directory into a timestamped sibling, converts R/T once, recomputes F, and updates combined/individual JSONs. A second migration is a no-op. Intrinsic files stay byte-identical; migration does not correct distortion or regenerate old validation plots.
- **F direction** is always reference→target: `x_target.T @ F @ x_ref = 0` on undistorted pixels. Derive F from final K/R/T for direct and bridged pairs. The legacy code inverted R/T but retained forward F when a pair was looked up backward; with reference cam3 this affected cam1 and cam2. Pose inversion and F direction are separate from the storage convention change.
- **Intrinsic fitting**: globally reserve every fifth distinct detected source frame for final testing, then every fifth frame of the remaining pool for model selection (approximately 64% training / 16% selection / 20% test). Model-selection and test frames never enter intrinsic/stereo fitting; final-test frames never enter model selection. Compare `opencv5`, `opencv4`, `rational_low`, and `rational8`; require at least five training and three model-selection detections per camera, a valid lens mapping, and selection RMS ≤ 2 px. Prefer fewer free coefficients among candidates within 0.05 px of the best score. `observation_split.json` records the split; `camN_model_selection.json` records candidates. A low training RMS is insufficient.
- **Cache and publication**: `--corners-cache PATH` reuses `.npz` detections after checking source names/sizes/modification times, board dimensions, and camera IDs; use a new cache for a different sampling schedule. `--output-dir PATH` chooses a separate destination. Build in `.calibration-staging-*`; publish only if all intrinsics are valid and every reference→target pair passes the untouched final test (≥3 measurable frames, mean <2 px, p95 <5 px, no invalid points). `heldout_epipolar.json` retains all finite residuals. Failure artifacts remain in `TARGET.failed-TIMENS`, with camera-specific `camN_failure.json` when applicable; the previous published calibration remains intact. Successful replacement backs it up to `TARGET.backup-TIMENS`. These workflow checks do not imply that any existing episode has been successfully refitted.
- **Logical timing offsets**: calibration, evaluation, and video export accept `--frame-offsets FILE`, a JSON map such as `{"cam4": -1, "cam6": -1}`. Camera N reads source frame `logical_frame + offset`; omitted cameras use zero, and keys/values must be canonical `camN`/integers. Raw video bytes remain unchanged. Cache reuse requires matching offsets. Combined calibration and split metadata save `source_frame_offsets` for all cameras; copies preserve the original `source_episode`. Evaluation and export infer saved offsets only when that source episode matches `--episode`; otherwise they use zero unless explicitly overridden. Do not claim episode 4 is resynchronized because episode 2's calibration used timing corrections.
- **Final video output**: `synced_raw/` is a synchronized intermediate and still contains lens distortion. `export_calibrated_videos.py --base DIR --episode EP --cams N` reads it with validated `calibration/`, applies the episode's residual integer frame offsets and lens correction, and writes `synced_undistorted/` plus matching `calibration_undistorted/`. It preserves source resolution/FPS over the common valid range, retains the original K and OpenCV extrinsics, and sets output distortion to zero. Keeping the original K crops some outer wide-angle content; the output is not pairwise stereo-rectified. Stereo rectification is optional for algorithms that specifically require it, not a prerequisite for general calibrated multi-view reconstruction. Sources remain intact. Only 8-bit input/output is currently supported; 10-bit input is rejected instead of silently reduced, regardless of the camera settings listed above. `--encoder auto` selects HEVC `hevc_videotoolbox` on macOS and H.264 `libx264` elsewhere; either encoder can also be selected explicitly.
- **Exported timing metadata**: the processed calibration resets `source_frame_offsets` to zero because the exported videos already include the corrections. `export_timeline` preserves the applied `frame_offsets`, starting logical frame, frame count, and FPS. Reusing lens calibration across episodes does not transfer timing corrections: automatic offsets remain scoped to the original calibration's `source_episode`.
- **Final all-camera preview**: `make_undistorted_preview.py --base DIR --episode EP` infers cameras from the completed export manifest, checks source signatures and a shared frame count/FPS, and reads `synced_undistorted/` without applying offsets or lens correction again. It places every camera in numeric row order with `CAM N` labels, using 640×360 aspect-preserving tiles (5×2, 3200×720 for 10 cameras). All tiles sample identical source indices; 119.88 FPS input uses frames 0, 4, 8, … at 30000/1001 FPS, covering the full clip, with the saved reference camera's audio when present. Episode 4's full camera videos remain 3840×2160 at 119.88 FPS. Outputs are `EP_undistorted_preview.mp4`, a JPEG poster, and JSON provenance inside `output/EP/`. `--decode-accel auto` uses VideoToolbox on macOS; `none` or `videotoolbox` override it. The old `EP_preview.mp4` remains a separate, lens-distorted initial sync diagnostic.
- **Error reporting**: per-frame RMS is `L2 / sqrt(N)`; the old `L2 / N` chart understated RMS by `sqrt(88) ≈ 9.38`. Its 0.06/0.1 chart thresholds are obsolete. Saved OpenCV overall RMS was unaffected. Standalone epipolar evaluation infers the saved reference (and rejects an explicit mismatch), retains high errors, and records every pair including missing/failed ones. Invalid intrinsics or incomplete coverage cannot yield GOOD; `--allow-invalid-intrinsics` is diagnostic only, with a failing result. `--out DIR` chooses its report destination.
- Within a single camera, GoPro auto-splits long recordings into chapter files (`GX01XXXX.MP4`, `GX02XXXX.MP4`, ...). The sync pipeline detects chapters by abutting `creation_time` tags (next.start ≈ prev.start + prev.duration) and merges them into one logical recording before audio extraction and stream-copy trim, using ffmpeg's `concat` demuxer.
- **Bridging for wide-baseline rigs**: with many cameras (e.g. 12 in 2×6), some opposite-end pairs barely share training frames or fail `stereoCalibrate`. The script computes viable `(i, j)` pairs, builds a graph with `shared ≥ 8` and `rms ≤ 1.5 px`, then BFS-finds the shortest hop-count path through good edges when needed. Composition remains `R_AC = R_BC @ R_AB`, `T_AC = R_BC @ T_AB + T_BC`. Per-camera JSONs record `method`, `path`, and `path_rms`; F derives from the final reference→target transform.

---

## 5. GOPRO LABS FEATURES USED / AVAILABLE

| Feature | Use | Link |
|---|---|---|
| QR Code Configuration | Flash identical settings to all cameras | https://gopro.github.io/labs/control/custom/ |
| Precision Time Sync | ms-accurate timecode alignment via animated QR | https://gopro.github.io/labs/control/precisiontime/ |
| GPS Time Sync | Automatic timecode sync outdoors via GPS atomic time | Built into Labs firmware |
| Altered File Naming (`!MBASE`) | Label each camera's files (CAM01, CAM02, etc.) | https://gopro.github.io/labs/control/ |
| USB Power Trigger | Start/stop all cameras via shared USB hub switch | Available in Labs |
| Sound Pressure Level Trigger | Auto-start recording on loud sound (has latency) | Available in Labs |

### Key Limitation

**No hardware genlock or USB-based frame sync** on Hero 10. The old Dual Hero System (Hero 3+ era) had wired sync but was dropped for waterproofing. Sensor clocks run independently per camera.

For sub-frame accuracy with fast motion, consider cameras with actual genlock: Blackmagic Micro, FLIR/Basler machine vision cameras.

---

## 6. IMPORTANT NOTES

- **Hero 10 LCD must be ON** for QR code scanning to work
- **Flat color profile** preserves dynamic range — apply LUT or color grade in post
- **Wide lens** introduces barrel distortion — must undistort using GoPro intrinsic calibration parameters before stereo matching / 3D reconstruction
- ISO locked at 400 for controlled indoor lighting. Adjust both min and max together if scene is too dark/bright
- 180° shutter = 1/240s at 120fps. For **sharper frames with less motion blur** (better for computer vision feature matching), consider using 90° (1/480s) or 45° (1/960s) shutter angle instead
- High bit rate → larger files but maximum quality
- Always do a clap at start AND end of each recording episode to verify sync and detect drift

---

## 7. WORKFLOW SUMMARY

1. Install GoPro Labs firmware on all Hero 10 cameras
2. Open QR Code Creator, display the settings QR → scan with every camera
3. Flash individual `!MBASE` naming QR to each camera (CAM01, CAM02, etc.)
4. Before each shoot: display Precision Time QR on laptop → scan with every camera
5. Start all cameras recording
6. Do a sharp clap (visible and audible to all cameras)
7. Shoot your scene
8. Do another clap at the end
9. Stop recording
10. In post: extract audio → detect clap peaks → compute frame offsets → trim to sync → verify

---

## 8. FILES DELIVERED

- `sync_pipeline.py` — Multi-episode audio sync pipeline (cross-correlation clap sync, stream-copy trim, chapter merging by `creation_time` + ffmpeg concat demuxer, configurable `--ref-cam`). Its `EP_preview.mp4` uses original LRV/MP4 inputs, retains lens distortion, defaults to a 30-second cap, and may fall back to the reference camera alone if the grid fails. It is an initial sync diagnostic, not a preview of the final undistorted outputs.
- `run_calibration.py` — Intrinsic candidate fitting with global held-out frames and lens validity checks; stereo calibration with all-pairs bridging; source-validated corner cache; staged publication after validation.
- `run_eval_epipolar.py` — Standalone epipolar validation with reference inference, lens validity checks, retained high-error measurements, and explicit failed/incomplete pair records. No high-error rejection that could hide bad calibration.
- `export_calibrated_videos.py` — Final synchronized, undistorted video export with matching zero-distortion calibration. Preserves full resolution/FPS, original K, and OpenCV extrinsics; applies source-episode timing offsets only when provenance matches, or accepts an explicit offset file. Does not modify source videos or perform pairwise stereo rectification.
- `make_undistorted_preview.py` — Labeled all-camera H.264 grid from manifest-validated undistorted exports, with shared frame sampling, reference-camera audio, a JPEG poster, and JSON provenance. Infers camera count from the export manifest; no `--cams` or timing-offset argument is needed.
- `calibration_geometry.py` — Shared extrinsics conventions, fundamental matrices, RMS, checked point undistortion, and distortion diagnostics.
- `calibration_fit.py` — Intrinsic candidate fits and selection using distortion validity and explicitly withheld observations.
- `calibration_observations.py` — Portable `.npz` corner cache, source/configuration checks, and a shared frame split across all cameras.
- `calibration_frame_offsets.py` — Validated per-camera integer offsets for reading logical frames without modifying video files.
- `calibration_validation.py` — Reuse gate for complete geometry, lens validity, and saved independent final-test evidence; copies via staging with a backup of the destination.
- `migrate_calibration.py` — Backup-first, idempotent legacy-to-OpenCV extrinsics migration; preserves intrinsic files and rig geometry.
- `sync_vis.py` — Local Flask web server + HTML5 `<video>` grid for ±2 frame per-camera offset adjustments after auto-sync. Lazy 720p H.264 proxy generation for fast browser decode; per-video seek queue prevents pile-ups on rapid scrub; clap-peak markers above the timeline; **Apply** button re-trims the synced videos in place using current offsets (reuses `sync_pipeline.ffmpeg_input_args` so chapter-merging works the same way) and resets adjustments to 0 with `previous_offsets` recorded.
- `viz_calibration.py` — 3D camera-pose viewer (Plotly self-contained HTML), reading legacy or world-to-camera extrinsics with reference/convention validation. Converts to camera centers/axes for frusta; displays K, RMS, baseline, method, and bridging paths. Reads `output/<episode>/calibration/calibration_all_cameras.json`.
- `organize_episodes.sh` — KITTI-style orchestrator. Runs sync, validates existing per-group source calibration before reuse, recalibrates invalid sources unless `--skip-calib` requires an immediate failure, and copies only validated calibration with provenance. After all calibration/copy steps, exports each selected episode exactly once to `synced_undistorted/` plus `calibration_undistorted/`, immediately followed by its all-camera undistorted preview. `--skip-export` skips both for calibration-only work. An export or preview failure stops the pipeline. Destination calibration replacement uses staging and preserves the previous directory as a backup. Migrates legacy `session_NN/` episode folder names automatically; that naming migration is distinct from extrinsics migration.
- `pyproject.toml` + `uv.lock` — uv project metadata and locked dependencies (numpy, scipy, opencv-python, matplotlib, flask, plotly); install with `uv sync`
- `gopro_hero10_3d_rig_config.txt` — The full camera config with QR URLs, all params, per-camera naming URLs, time sync links, and operational notes
- `gopro_3d_vision_project_context.md` — This file (full context dump for agent handoff)
- `README.md` — User-facing documentation: prerequisites, shooting workflow, run commands for each script, output layout, validation checks

### Genericity boundary (what's GoPro-specific vs camera-agnostic)

**GoPro-specific** (would need editing for other camera vendors):
- `sync_pipeline.py:288` — file glob pattern `*GX*.MP4` (GoPro uses GX/GH prefix). Other cameras would need a different pattern.
- LRV proxy lookup (`replace("GX","GL").replace(".MP4",".LRV")`) — GoPro chapter format. Falls back to scaled MP4 if absent, so non-GoPro sources work but with slower preview generation.
- THM thumbnail copy is best-effort; absent thumbnails are silently skipped.
- README/docstring banners and the camera-config file are written for Hero 10.

**Camera-agnostic** (works for any video source feeding the same on-disk layout):
- All calibration math (`cv2.calibrateCamera`, `cv2.stereoCalibrate`, bridging BFS, transform composition) — pure OpenCV and numpy, no GoPro assumptions.
- The N-camera preview grid heuristic, the `--ref-cam` parameter, the audio sync algorithm.
- `sync_vis.py` (HTTP Range, HTML5 video, proxy generation), `run_eval_epipolar.py`, `viz_calibration.py`.
- The `cv2.VideoCapture` pipeline is codec-agnostic via `-c copy` for trims.
