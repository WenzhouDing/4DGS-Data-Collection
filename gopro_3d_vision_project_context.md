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

Play back synced videos and confirm the clap frame aligns visually across all cameras. Do clap at end of session too to check for drift.

### Accuracy Notes

- Audio sync at 48kHz sample rate → ~0.02ms per sample, well within one frame at 120fps (~8.3ms)
- Cross-correlation (`scipy.signal.correlate`) is more robust than peak detection in noisy environments
- GoPro Hero 10 has NO genlock — over long recordings (10+ min) cameras can drift by 1-2 frames. Clap at both ends to detect drift. If significant, resample/interpolate.
- Calibration pipeline reads synced videos in lockstep (all cameras on same frame number) — this guarantees stereo calibration pairs see the board in the same pose. Detection is parallelised across cameras within each frame using `findChessboardCornersSB` on downscaled images (~960px), with `cornerSubPix` refinement at full 4K resolution.

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
- Always do a clap at start AND end of each recording session to verify sync and detect drift

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

- `sync_pipeline.py` — Multi-session audio sync pipeline (cross-correlation clap sync, stream-copy trim, preview grid)
- `run_calibration.py` — Intrinsic + extrinsic calibration from synced checkerboard video (lockstep multi-cam, parallel detection, early stopping)
- `gopro_hero10_3d_rig_config.txt` — The full camera config with QR URLs, all params, per-camera naming URLs, time sync links, and operational notes
- `gopro_3d_vision_project_context.md` — This file (full context dump for agent handoff)
