#!/usr/bin/env python3
"""
Sync fine-tune visualizer — browser UI
======================================
Browser-based ±2 frame per-camera offset adjustment for synced GoPro video.

Architecture
------------
Flask backend + HTML5 <video> frontend.
  * Browser uses hardware-accelerated HEVC decode and GPU compositing —
    seeking and scrubbing across the camera grid is far smoother than the
    cv2.VideoCapture + matplotlib approach.
  * Backend is just Flask + send_file with HTTP Range support (Werkzeug
    handles the Range header automatically when conditional=True).
  * Frontend is a single embedded HTML page (no React, no build step) —
    plain CSS Grid for layout, range inputs for sliders, keyboard
    shortcuts for fast iteration.

Endpoints
---------
GET  /            single-page UI
GET  /metadata    JSON: cams, fps, total_frames, ref_cam, grid, offsets
GET  /video/<N>   serves cam{N}_synced.mp4 (range-supported)
POST /save        writes offsets to output/<episode>/sync_adjustments.json

Usage
-----
    uv run python sync_vis.py [--base DIR] [--episode EPISODE]
                              [--cams N] [--ref-cam N] [--port PORT]

The browser opens automatically. Ctrl+C to stop the server.
"""

import argparse
import json
import math
import os
import subprocess
import threading
import webbrowser
from concurrent.futures import ThreadPoolExecutor

import cv2
from flask import Flask, jsonify, request, send_file, Response

# Re-use chapter-merge / concat-demuxer helpers from the sync pipeline so
# the Apply endpoint goes through the same ffmpeg input plumbing as the
# original sync (no logic duplication).
from sync_pipeline import ffmpeg_input_args


PROXY_HEIGHT = 720           # H.264 proxy height; faster decode than 4K HEVC
PROXY_BITRATE = "2500k"      # plenty for visual sync verification
PROXY_PRESET = "veryfast"    # fast encode; this is one-shot


def proxy_path_for(synced_mp4):
    """Sibling proxy file path for a given synced MP4."""
    base, _ = os.path.splitext(synced_mp4)
    return f"{base}_proxy.mp4"


def generate_proxy(synced_mp4):
    """Re-encode synced video to a 720p H.264 proxy (idempotent: skips if newer than source)."""
    out = proxy_path_for(synced_mp4)
    if (os.path.exists(out)
            and os.path.getmtime(out) >= os.path.getmtime(synced_mp4)
            and os.path.getsize(out) > 0):
        return out, False  # cached
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", synced_mp4,
        "-vf", f"scale=-2:{PROXY_HEIGHT}",
        "-c:v", "libx264", "-preset", PROXY_PRESET, "-b:v", PROXY_BITRATE,
        "-pix_fmt", "yuv420p",  # broadest browser compat
        "-an",                  # vis tool doesn't need audio in the proxy
        "-movflags", "+faststart",
        out,
    ]
    subprocess.run(cmd, check=True)
    return out, True


def load_episode_metadata(base, episode):
    """Load output/<episode>/metadata/<episode>_metadata.json. Returns dict or None."""
    path = os.path.join(base, "output", episode, "metadata",
                        f"{episode}_metadata.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError):
        return None


def compute_peak_frames(meta, fps):
    """Convert each camera's raw clap_peak sample number to a synced-frame index.

      peak_frame_synced = (peak_sample / sample_rate - trim_sec) * fps

    Returns {cam_int: peak_frame_int}; cams whose peak fell before the trim
    boundary (i.e. would be negative) are omitted.
    """
    if not meta:
        return {}
    sample_rate = meta.get("sample_rate", 48000)
    out = {}
    for cam_key, cam_meta in meta.get("cameras", {}).items():
        if not cam_key.startswith("cam"):
            continue
        try:
            cam_n = int(cam_key[3:])
        except ValueError:
            continue
        peak_sample = cam_meta.get("clap_peak", {}).get("sample")
        trim_sec = cam_meta.get("trim_sec", 0)
        if peak_sample is None:
            continue
        peak_synced_sec = (peak_sample / sample_rate) - trim_sec
        if peak_synced_sec < 0:
            continue
        out[cam_n] = int(round(peak_synced_sec * fps))
    return out


def ensure_proxies(synced_mp4s):
    """Generate proxies for all synced videos in parallel; returns {original: proxy}."""
    proxies = {}
    if not synced_mp4s:
        return proxies
    print(f"  Checking proxies for {len(synced_mp4s)} cam(s) (720p H.264 for fast browser decode)...")
    n_workers = min(len(synced_mp4s), max(1, (os.cpu_count() or 4) // 2))
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(generate_proxy, src): src for src in synced_mp4s}
        for fut in futures:
            src = futures[fut]
            out, generated = fut.result()
            proxies[src] = out
            tag = "generated" if generated else "cached"
            print(f"    {os.path.basename(src)} -> {os.path.basename(out)}  [{tag}]")
    return proxies


# ─── CLI ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Sync fine-tune visualizer (web UI)")
    p.add_argument("--base", default=".", help="Project root")
    p.add_argument("--episode", default="episode_0001", help="Synced episode")
    p.add_argument("--cams", type=int, default=12, help="Number of cameras")
    p.add_argument("--ref-cam", type=int, default=1,
                   help="Reference camera (its offset stays at 0)")
    p.add_argument("--port", type=int, default=8765,
                   help="Local port for the web UI")
    p.add_argument("--no-browser", action="store_true",
                   help="Don't auto-open a browser tab")
    p.add_argument("--no-proxy", action="store_true",
                   help="Serve the original 4K HEVC instead of generating "
                        "a 720p H.264 proxy. Slower seeking; useful for "
                        "verifying the proxy isn't masking issues.")
    return p.parse_args()


def grid_layout(n):
    """Wider-than-tall heuristic, matches the sync preview grid."""
    if n <= 3:
        return n, 1
    cols = math.ceil(n / 2)
    rows = math.ceil(n / cols)
    return cols, rows


# ─── Embedded frontend ────────────────────────────────────────────
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Sync Visualizer</title>
<style>
  :root {
    --bg: #0e1116;
    --panel: #161b22;
    --border: #30363d;
    --text: #e6edf3;
    --muted: #7d8590;
    --accent: #2ea043;
    --accent-hover: #3fb950;
    --ref: #f85149;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 16px;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", sans-serif;
    background: var(--bg); color: var(--text);
    font-size: 13px;
  }
  header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 12px;
  }
  h1 { font-size: 15px; margin: 0; font-weight: 500; letter-spacing: 0.2px; }
  h1 .episode { color: var(--muted); }
  button {
    background: var(--accent); color: #fff; border: 0;
    padding: 7px 14px; border-radius: 6px; cursor: pointer;
    font-size: 13px; font-weight: 500;
    transition: background 0.15s;
  }
  button:hover { background: var(--accent-hover); }
  button:active { transform: scale(0.98); }
  button:disabled { background: #4a5160; cursor: wait; opacity: 0.6; }
  button.danger { background: #b85d00; }
  button.danger:hover { background: #d97706; }
  .header-buttons { display: flex; gap: 8px; }
  .help-text { color: var(--muted); font-size: 11px; }
  .frame-bar {
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 12px;
    padding: 10px 14px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 6px;
  }
  .frame-bar label { font-weight: 500; min-width: 50px; }
  .slider-wrap { position: relative; flex: 1; }
  .slider-wrap input[type=range] {
    width: 100%; height: 6px; accent-color: var(--accent); display: block;
  }
  .peak-track {
    position: absolute; top: -12px; left: 8px; right: 8px; height: 10px;
    pointer-events: none;
  }
  .peak-marker {
    position: absolute; top: 0; bottom: 0; width: 2px;
    background: #fbbf24; opacity: 0.85;
    pointer-events: auto; cursor: pointer;
    transform: translateX(-1px);
    transition: width 0.1s, background 0.1s;
  }
  .peak-marker.ref { background: var(--ref); }
  .peak-marker:hover { width: 4px; transform: translateX(-2px); opacity: 1; }
  .peak-marker .tooltip {
    position: absolute; bottom: 100%; left: 50%;
    transform: translateX(-50%); margin-bottom: 4px;
    background: #161b22; color: var(--text);
    padding: 3px 7px; border-radius: 3px; font-size: 11px;
    white-space: nowrap; pointer-events: none; opacity: 0;
    transition: opacity 0.1s;
    font-family: "SF Mono", Menlo, monospace;
    border: 1px solid var(--border);
  }
  .peak-marker:hover .tooltip { opacity: 1; }
  .frame-bar .display {
    font-family: "SF Mono", Menlo, monospace; min-width: 200px;
    text-align: right; color: var(--muted);
  }
  .video-grid {
    display: grid; gap: 6px; margin-bottom: 14px;
  }
  .video-cell {
    position: relative; background: #000;
    border: 2px solid var(--border); border-radius: 4px;
    overflow: hidden; aspect-ratio: 16 / 9;
  }
  .video-cell.ref { border-color: var(--ref); }
  .video-cell video {
    width: 100%; height: 100%; object-fit: contain; display: block;
  }
  .video-cell .badge {
    position: absolute; top: 4px; left: 4px;
    background: rgba(0, 0, 0, 0.78); padding: 3px 7px;
    border-radius: 3px; font-size: 11px;
    font-family: "SF Mono", Menlo, monospace;
    pointer-events: none;
  }
  .video-cell.ref .badge { color: var(--ref); }
  .video-cell .offset-tag {
    position: absolute; top: 4px; right: 4px;
    background: rgba(0, 0, 0, 0.78); padding: 3px 7px;
    border-radius: 3px; font-size: 11px;
    font-family: "SF Mono", Menlo, monospace;
    pointer-events: none;
  }
  .offsets {
    display: grid; gap: 8px; padding: 12px;
    background: var(--panel);
    border: 1px solid var(--border); border-radius: 6px;
  }
  .offset-row {
    display: flex; align-items: center; gap: 12px;
  }
  .offset-row .name {
    font-family: "SF Mono", Menlo, monospace;
    min-width: 60px; font-weight: 500;
  }
  .offset-row.ref .name { color: var(--ref); }
  .offset-row .ticks {
    display: flex; gap: 4px;
  }
  .offset-row .tick {
    width: 32px; height: 28px;
    background: #21262d; border: 1px solid var(--border);
    color: var(--text); border-radius: 4px; cursor: pointer;
    font-family: "SF Mono", Menlo, monospace; font-size: 12px;
    transition: all 0.1s;
  }
  .offset-row .tick:hover { background: #30363d; }
  .offset-row .tick.active {
    background: var(--accent); border-color: var(--accent); color: #fff;
  }
  .offset-row.ref .tick.active {
    background: var(--ref); border-color: var(--ref);
  }
  .offset-row .value {
    font-family: "SF Mono", Menlo, monospace;
    color: var(--muted); min-width: 30px;
  }
  .status {
    margin-top: 10px; padding: 8px 12px; min-height: 20px;
    color: var(--muted); font-family: "SF Mono", Menlo, monospace;
    font-size: 12px; text-align: right;
  }
  .status.ok { color: var(--accent); }
  .shortcuts {
    margin-top: 12px; padding: 10px 14px;
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--muted); font-size: 11px;
    display: flex; gap: 18px; flex-wrap: wrap;
  }
  kbd {
    background: #21262d; border: 1px solid var(--border);
    border-radius: 3px; padding: 1px 6px;
    font-family: "SF Mono", Menlo, monospace; font-size: 11px;
    color: var(--text);
  }
</style>
</head>
<body>
<header>
  <h1>Sync Visualizer — <span class="episode" id="episode-name">…</span></h1>
  <div class="header-buttons">
    <button id="save-btn">Save adjustments</button>
    <button id="apply-btn" class="danger">Apply to videos</button>
  </div>
</header>

<div class="frame-bar">
  <label>Frame</label>
  <div class="slider-wrap">
    <div class="peak-track" id="peak-track"></div>
    <input type="range" id="frame-slider" min="0" max="0" value="0" step="1">
  </div>
  <div class="display" id="frame-display">—</div>
</div>

<div class="video-grid" id="video-grid"></div>

<div class="offsets" id="offsets"></div>

<div class="shortcuts">
  <span><kbd>←</kbd> <kbd>→</kbd> step ±1 frame</span>
  <span><kbd>shift</kbd>+<kbd>←</kbd>/<kbd>→</kbd> step ±10</span>
  <span><kbd>home</kbd>/<kbd>end</kbd> jump to start/end</span>
  <span><kbd>1</kbd>–<kbd>9</kbd> focus cam offset</span>
  <span><kbd>s</kbd> save</span>
  <span>Click a peak marker (above the slider) to jump to that camera's clap moment</span>
</div>

<div class="status" id="status"></div>

<script>
"use strict";
let META, FPS, TOTAL, REF, CAMS, GRID;
const offsets = {};
const videos = {};
const offsetTagEls = {};
const tickEls = {};
let currentFrame = 0;

const $ = (id) => document.getElementById(id);

async function init() {
  const r = await fetch("/metadata");
  META = await r.json();
  FPS = META.fps;
  TOTAL = META.total_frames;
  REF = META.ref_cam;
  CAMS = META.cams;
  GRID = META.grid;

  CAMS.forEach(c => {
    const k = `cam${c}`;
    offsets[k] = (META.offsets && META.offsets[k] !== undefined)
      ? META.offsets[k] : 0;
  });

  $("episode-name").textContent = META.episode;

  // Video grid
  const grid = $("video-grid");
  grid.style.gridTemplateColumns = `repeat(${GRID.cols}, 1fr)`;
  CAMS.forEach(c => {
    const cell = document.createElement("div");
    cell.className = "video-cell" + (c === REF ? " ref" : "");
    const v = document.createElement("video");
    v.muted = true; v.preload = "auto"; v.playsInline = true;
    v.src = `/video/${c}`;
    cell.appendChild(v);
    const badge = document.createElement("div");
    badge.className = "badge";
    badge.textContent = `cam${c}${c === REF ? ' (ref)' : ''}`;
    cell.appendChild(badge);
    const tag = document.createElement("div");
    tag.className = "offset-tag";
    tag.id = `tag-${c}`;
    cell.appendChild(tag);
    grid.appendChild(cell);
    videos[c] = v;
    offsetTagEls[c] = tag;
    bindSeekQueue(c);
  });

  // Offset rows
  const offsetsEl = $("offsets");
  CAMS.forEach(c => {
    const row = document.createElement("div");
    row.className = "offset-row" + (c === REF ? " ref" : "");
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = `cam${c}${c === REF ? ' (ref)' : ''}`;
    row.appendChild(name);
    const ticks = document.createElement("div");
    ticks.className = "ticks";
    tickEls[c] = {};
    [-2, -1, 0, 1, 2].forEach(v => {
      const btn = document.createElement("button");
      btn.className = "tick";
      btn.textContent = v >= 0 ? `+${v}` : `${v}`;
      btn.addEventListener("click", () => setOffset(c, v));
      tickEls[c][v] = btn;
      ticks.appendChild(btn);
    });
    row.appendChild(ticks);
    const valEl = document.createElement("span");
    valEl.className = "value";
    valEl.id = `val-${c}`;
    row.appendChild(valEl);
    offsetsEl.appendChild(row);
    refreshTicks(c);
  });

  // Frame slider
  const fs = $("frame-slider");
  fs.max = TOTAL - 1;
  fs.addEventListener("input", e => {
    currentFrame = parseInt(e.target.value);
    seekAll();
    updateFrameDisplay();
  });

  // Save / Apply
  $("save-btn").addEventListener("click", save);
  $("apply-btn").addEventListener("click", apply);

  // Keyboard shortcuts
  document.addEventListener("keydown", onKey);

  // Wait for all videos to report duration. cv2's CAP_PROP_FRAME_COUNT is
  // unreliable for HEVC — use the reference cam's actual decoded duration
  // (via HTML5 video) as truth.
  await Promise.all(CAMS.map(c => new Promise(res => {
    if (videos[c].readyState >= 1) res();
    else videos[c].addEventListener("loadedmetadata", () => res(), { once: true });
  })));
  const refDur = videos[REF].duration;
  if (Number.isFinite(refDur) && refDur > 0) {
    TOTAL = Math.floor(refDur * FPS);
    fs.max = TOTAL - 1;
  }

  renderPeakMarkers();

  currentFrame = Math.floor(TOTAL * 0.1);
  fs.value = currentFrame;
  seekAll();
  updateFrameDisplay();
  setStatus(`Loaded ${CAMS.length} cameras · ${TOTAL} frames @ ${FPS.toFixed(2)} fps · ref=cam${REF}`);
}

function renderPeakMarkers() {
  const peaks = (META && META.peak_frames) || {};
  const track = $("peak-track");
  track.innerHTML = "";
  CAMS.forEach(c => {
    const f = peaks[`cam${c}`];
    if (f === undefined || f === null) return;
    const pct = TOTAL > 0 ? (f / (TOTAL - 1)) * 100 : 0;
    if (pct < 0 || pct > 100) return;
    const m = document.createElement("div");
    m.className = "peak-marker" + (c === REF ? " ref" : "");
    m.style.left = `${pct}%`;
    const tip = document.createElement("div");
    tip.className = "tooltip";
    tip.textContent = `cam${c}${c === REF ? ' (ref)' : ''} clap @ f=${f}`;
    m.appendChild(tip);
    m.addEventListener("click", () => {
      currentFrame = f;
      $("frame-slider").value = f;
      seekAll();
      updateFrameDisplay();
    });
    track.appendChild(m);
  });
}

function setOffset(c, v) {
  offsets[`cam${c}`] = v;
  refreshTicks(c);
  $(`val-${c}`).textContent = v >= 0 ? `+${v}` : `${v}`;
  seekCam(c);
}

function refreshTicks(c) {
  const cur = offsets[`cam${c}`];
  [-2, -1, 0, 1, 2].forEach(v => {
    tickEls[c][v].classList.toggle("active", v === cur);
  });
  $(`val-${c}`).textContent = cur >= 0 ? `+${cur}` : `${cur}`;
}

function seekAll() {
  CAMS.forEach(seekCam);
}

// Per-cam seek queue: only one outstanding seek per video at a time.
// Rapid slider drags would otherwise pile up cancellable seeks faster
// than the decoder can complete them, leaving the video frame stale.
const pending = {};

function seekCam(c) {
  const v = videos[c];
  const o = offsets[`cam${c}`];
  let target = currentFrame + o;
  if (target < 0) target = 0;
  if (target > TOTAL - 1) target = TOTAL - 1;
  // Seek to mid-frame timestamp to avoid landing on a boundary.
  const t = (target + 0.5) / FPS;
  offsetTagEls[c].textContent = `f=${target} ${o >= 0 ? '+' : ''}${o}`;
  if (v.seeking) {
    // Coalesce: remember the latest target; flush on the next 'seeked'.
    pending[c] = t;
  } else {
    v.currentTime = t;
  }
}

function bindSeekQueue(c) {
  videos[c].addEventListener("seeked", () => {
    if (pending[c] !== undefined) {
      const t = pending[c];
      delete pending[c];
      videos[c].currentTime = t;
    }
  });
}

function updateFrameDisplay() {
  $("frame-display").textContent =
    `${currentFrame} / ${TOTAL - 1}  ·  ${(currentFrame / FPS).toFixed(3)}s`;
}

function onKey(e) {
  // Don't hijack if user is typing in an input.
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
  let handled = true;
  const step = e.shiftKey ? 10 : 1;
  if (e.key === "ArrowRight") {
    currentFrame = Math.min(TOTAL - 1, currentFrame + step);
  } else if (e.key === "ArrowLeft") {
    currentFrame = Math.max(0, currentFrame - step);
  } else if (e.key === "Home") {
    currentFrame = 0;
  } else if (e.key === "End") {
    currentFrame = TOTAL - 1;
  } else if (e.key === "s" || e.key === "S") {
    save();
    return;
  } else if (/^[1-9]$/.test(e.key)) {
    const c = parseInt(e.key);
    if (CAMS.includes(c)) {
      const el = tickEls[c][offsets[`cam${c}`]];
      if (el) { el.scrollIntoView({behavior: "smooth", block: "nearest"}); el.focus(); }
    }
    handled = false;
  } else {
    handled = false;
  }
  if (handled) {
    $("frame-slider").value = currentFrame;
    seekAll();
    updateFrameDisplay();
    e.preventDefault();
  }
}

async function save() {
  setStatus("Saving…");
  try {
    const r = await fetch("/save", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({frame_offsets: offsets}),
    });
    const data = await r.json();
    if (data.status === "ok") {
      const summary = CAMS.map(c => `cam${c}=${fmt(offsets[`cam${c}`])}`).join(" ");
      setStatus(`Saved ${new Date().toLocaleTimeString()} → ${data.path}  ·  ${summary}`, true);
    } else {
      setStatus(`Save failed: ${data.error || 'unknown'}`);
    }
  } catch (e) {
    setStatus(`Save error: ${e.message}`);
  }
}

async function apply() {
  const nonZero = CAMS.filter(c => offsets[`cam${c}`] !== 0);
  if (nonZero.length === 0) {
    setStatus("All offsets are 0 — nothing to apply.");
    return;
  }
  const summary = nonZero.map(c => `cam${c}=${fmt(offsets[`cam${c}`])}`).join(", ");
  if (!confirm(
    `Re-trim cam*_synced.mp4 with these offsets?\n\n  ${summary}\n\n` +
    `This rewrites the synced videos in place (and regenerates proxies). ` +
    `Offsets reset to 0 afterward; the previous values are recorded in ` +
    `sync_adjustments.json.previous_offsets.\n\n` +
    `Takes ~30-60s for 5 cams, longer for 12.`
  )) return;
  setStatus("Applying… re-trimming videos and regenerating proxies (30-60s)…");
  $("apply-btn").disabled = true;
  $("save-btn").disabled = true;
  document.querySelectorAll(".tick").forEach(b => b.disabled = true);
  try {
    const r = await fetch("/apply", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({frame_offsets: offsets}),
    });
    const data = await r.json();
    if (data.status === "ok") {
      const w = (data.warnings && data.warnings.length)
        ? ` · warnings: ${data.warnings.join("; ")}` : "";
      setStatus(
        `Applied ${new Date().toLocaleTimeString()}: new common_dur=` +
        `${data.new_common_dur.toFixed(3)}s (${data.new_total_frames} frames)` +
        w + " · reloading…", true);
      setTimeout(() => location.reload(), 1500);
    } else {
      setStatus(`Apply failed: ${data.error || 'unknown'}`);
      $("apply-btn").disabled = false;
      $("save-btn").disabled = false;
      document.querySelectorAll(".tick").forEach(b => b.disabled = false);
    }
  } catch (e) {
    setStatus(`Apply error: ${e.message}`);
    $("apply-btn").disabled = false;
    $("save-btn").disabled = false;
    document.querySelectorAll(".tick").forEach(b => b.disabled = false);
  }
}

function fmt(v) { return v >= 0 ? `+${v}` : `${v}`; }

function setStatus(msg, ok) {
  const el = $("status");
  el.textContent = msg;
  el.className = "status" + (ok ? " ok" : "");
}

init().catch(e => {
  setStatus(`Init error: ${e.message}`);
  console.error(e);
});
</script>
</body>
</html>
"""


# ─── Backend ──────────────────────────────────────────────────────
app = Flask(__name__)
VIDEO_PATHS = {}
META = {}
SAVE_PATH = ""
USE_PROXY = True


@app.route("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


@app.route("/metadata")
def metadata():
    # _base is server-internal (used by /apply to find raw camera files);
    # don't leak the project path to the browser.
    public = {k: v for k, v in META.items() if not k.startswith("_")}
    return jsonify(public)


@app.route("/video/<int:cam>")
def video(cam):
    path = VIDEO_PATHS.get(cam)
    if not path:
        return ("not found", 404)
    return send_file(path, mimetype="video/mp4", conditional=True)


def _validate_offsets(incoming, cams):
    """Validate that incoming has an int in [-2,2] for every cam.

    Returns (cleaned_dict, None) on success or (None, error_response) on
    failure. Refuses partial bodies — every cam must be specified — so a
    malformed POST can't silently zero out the user's saved offsets.
    """
    if not isinstance(incoming, dict) or not incoming:
        return None, (jsonify({"status": "error",
                               "error": "missing or empty frame_offsets"}), 400)
    cleaned = {}
    for cam in cams:
        key = f"cam{cam}"
        if key not in incoming:
            return None, (jsonify({"status": "error",
                                   "error": f"frame_offsets missing {key}"}), 400)
        try:
            v = int(incoming[key])
        except (TypeError, ValueError):
            return None, (jsonify({"status": "error",
                                   "error": f"non-int offset for {key}"}), 400)
        if v < -2 or v > 2:
            return None, (jsonify({"status": "error",
                                   "error": f"{key} offset out of range"}), 400)
        cleaned[key] = v
    return cleaned, None


@app.route("/save", methods=["POST"])
def save_offsets():
    data = request.get_json(silent=True) or {}
    cleaned, err = _validate_offsets(data.get("frame_offsets"), META["cams"])
    if err is not None:
        return err
    out = {
        "episode": META["episode"],
        "ref_cam": META["ref_cam"],
        "fps": META["fps"],
        "total_frames": META["total_frames"],
        "frame_offsets": cleaned,
    }
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    with open(SAVE_PATH, "w") as f:
        json.dump(out, f, indent=2)
    META["offsets"] = cleaned
    print(f"  Saved offsets -> {SAVE_PATH}: {cleaned}")
    return jsonify({"status": "ok", "path": SAVE_PATH})


@app.route("/apply", methods=["POST"])
def apply_offsets():
    """Re-trim original raw videos with the offsets baked in.

    Reads incoming offsets (POST body or saved sync_adjustments.json), looks
    up each cam's source files + original trim_sec from metadata.json, and
    re-runs ffmpeg trim+concat to replace cam{N}_synced.mp4 in place.
    Stale proxies are deleted and regenerated.  metadata.json is updated
    with the new trim values; sync_adjustments.json offsets are reset to 0
    (with the previously-applied values stashed under previous_offsets).
    """
    base = META["_base"]
    episode = META["episode"]
    fps = META["fps"]
    cams = META["cams"]

    # Require body. Earlier design fell back to sync_adjustments.json when
    # body was missing, but that meant an empty/malformed POST silently
    # triggered re-trim against the last-saved values — a footgun. Apply is
    # destructive; require the caller to spell out exactly what to apply.
    body = request.get_json(silent=True) or {}
    cleaned, err = _validate_offsets(body.get("frame_offsets"), cams)
    if err is not None:
        return err

    if all(v == 0 for v in cleaned.values()):
        return jsonify({"status": "error",
                        "error": "all offsets are 0; nothing to apply"}), 400

    # 2. Load metadata.json for source paths + original trim values.
    meta_path = os.path.join(base, "output", episode, "metadata",
                             f"{episode}_metadata.json")
    if not os.path.exists(meta_path):
        return jsonify({"status": "error",
                        "error": f"metadata not found: {meta_path}"}), 400
    try:
        with open(meta_path) as f:
            episode_meta = json.load(f)
    except (json.JSONDecodeError, ValueError) as e:
        return jsonify({"status": "error",
                        "error": f"could not read metadata: {e}"}), 400

    cams_meta = episode_meta.get("cameras", {})

    # 3. Compute new trim per cam (clamped to >= 0).
    new_trims = {}
    raw_durs = {}
    source_paths = {}
    warnings = []
    for c in cams:
        key = f"cam{c}"
        cm = cams_meta.get(key, {})
        original_trim = float(cm.get("trim_sec", 0))
        offset = cleaned[key]
        new_trim = original_trim + offset / fps
        if new_trim < 0:
            warnings.append(
                f"{key}: offset {offset:+d} would push trim below 0 "
                f"(original_trim={original_trim:.4f}s); clamped to 0"
            )
            new_trim = 0
        new_trims[c] = new_trim
        raw_durs[c] = float(cm.get("duration", 0))
        names = cm.get("source_files") or [cm.get("source_file")]
        source_paths[c] = [os.path.join(base, str(c), n) for n in names if n]
        for p in source_paths[c]:
            if not os.path.exists(p):
                return jsonify({"status": "error",
                                "error": f"source file missing: {p}"}), 400

    new_common_dur = min(raw_durs[c] - new_trims[c] for c in cams)

    # 4. Re-trim each cam (concat demuxer for multi-chapter).
    synced_dir = os.path.join(base, "output", episode, "synced_raw")
    work_dir = os.path.join(base, ".sync_work")
    os.makedirs(work_dir, exist_ok=True)

    print(f"  Apply: re-trimming {len(cams)} cam(s) with offsets {cleaned} "
          f"(new common_dur={new_common_dur:.4f}s)")

    new_synced = {}
    for c in cams:
        out = os.path.join(synced_dir, f"cam{c}_synced.mp4")
        inp_args = ffmpeg_input_args(
            source_paths[c], work_dir, f"apply_cam{c}",
        )
        cmd = ["ffmpeg", "-y", "-v", "error"]
        if new_trims[c] > 0.0001:
            cmd += ["-ss", f"{new_trims[c]:.6f}"]
        cmd += inp_args + ["-t", f"{new_common_dur:.6f}", "-c", "copy", out]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            return jsonify({"status": "error",
                            "error": f"ffmpeg failed for cam{c}: {e}"}), 500
        proxy = proxy_path_for(out)
        if os.path.exists(proxy):
            os.remove(proxy)
        new_synced[c] = out
        print(f"    cam{c}: new_trim={new_trims[c]:.4f}s "
              f"(offset {cleaned[f'cam{c}']:+d}) -> {os.path.basename(out)}")

    # 5. Regenerate proxies for the new synced videos (if proxy mode).
    if USE_PROXY:
        new_proxies = ensure_proxies([new_synced[c] for c in cams])
        for c in cams:
            VIDEO_PATHS[c] = new_proxies[new_synced[c]]
    else:
        for c in cams:
            VIDEO_PATHS[c] = new_synced[c]

    # 6. Update metadata.json with new trim values.
    for c in cams:
        key = f"cam{c}"
        if key in cams_meta:
            cams_meta[key]["trim_sec"] = new_trims[c]
            cams_meta[key]["trim_frames"] = new_trims[c] * fps
    episode_meta["common_duration_sec"] = new_common_dur
    episode_meta["common_duration_frames"] = int(new_common_dur * fps)
    with open(meta_path, "w") as f:
        json.dump(episode_meta, f, indent=2, default=str)

    # 7. Reset adjustments JSON; recompute peak frames for the new timeline
    #    and update server-side META so the next /metadata reflects reality.
    META["total_frames"] = int(new_common_dur * fps)
    META["offsets"] = {f"cam{c}": 0 for c in cams}
    new_peaks = compute_peak_frames(episode_meta, fps)
    META["peak_frames"] = {f"cam{c}": pf for c, pf in new_peaks.items()}

    out_adj = {
        "episode": episode,
        "ref_cam": META["ref_cam"],
        "fps": fps,
        "total_frames": META["total_frames"],
        "frame_offsets": META["offsets"],
        "previous_offsets": cleaned,
    }
    with open(SAVE_PATH, "w") as f:
        json.dump(out_adj, f, indent=2)

    print(f"  Apply: done. common_dur={new_common_dur:.4f}s, "
          f"total_frames={META['total_frames']}")

    return jsonify({
        "status": "ok",
        "new_common_dur": new_common_dur,
        "new_total_frames": META["total_frames"],
        "new_trims": {f"cam{c}": new_trims[c] for c in cams},
        "warnings": warnings,
        "applied_offsets": cleaned,
    })


# ─── Main ─────────────────────────────────────────────────────────
def main():
    args = parse_args()
    BASE = os.path.abspath(args.base)
    EPISODE = args.episode
    NUM_CAMS = args.cams
    REF_CAM = args.ref_cam

    if not (1 <= REF_CAM <= NUM_CAMS):
        print(f"ERROR: --ref-cam {REF_CAM} must be in 1..{NUM_CAMS}")
        raise SystemExit(1)

    SYNCED_DIR = os.path.join(BASE, "output", EPISODE, "synced_raw")
    save_path = os.path.join(BASE, "output", EPISODE, "sync_adjustments.json")

    # Probe every available synced video for fps + frame count via cv2.
    fps = None
    total_frames = None
    cams_present = []
    originals = {}
    for cam in range(1, NUM_CAMS + 1):
        vid = os.path.join(SYNCED_DIR, f"cam{cam}_synced.mp4")
        if not os.path.exists(vid):
            print(f"  WARNING: {vid} not found, skipping cam {cam}")
            continue
        cap = cv2.VideoCapture(vid)
        if not cap.isOpened():
            print(f"  WARNING: cannot open {vid}")
            cap.release()
            continue
        if fps is None:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        originals[cam] = vid
        cams_present.append(cam)

    # Generate (or reuse) 720p H.264 proxies for fast browser decoding.
    # Browser decodes 720p H.264 ~10x faster than 4K HEVC; offsets stay
    # accurate because the proxy preserves the original duration.
    if not args.no_proxy:
        proxies = ensure_proxies([originals[c] for c in cams_present])
        for c in cams_present:
            VIDEO_PATHS[c] = proxies[originals[c]]
    else:
        for c in cams_present:
            VIDEO_PATHS[c] = originals[c]

    if not cams_present:
        print(f"ERROR: no synced videos found in {SYNCED_DIR}")
        raise SystemExit(1)
    if REF_CAM not in cams_present:
        print(f"ERROR: synced video for reference cam {REF_CAM} not found")
        raise SystemExit(1)

    # Resume from previous adjustments if present.
    initial_offsets = {f"cam{c}": 0 for c in cams_present}
    if os.path.exists(save_path):
        try:
            with open(save_path) as f:
                saved = json.load(f)
            for c in cams_present:
                v = saved.get("frame_offsets", {}).get(f"cam{c}")
                if v is not None:
                    initial_offsets[f"cam{c}"] = int(v)
            print(f"  Resumed from {save_path}: {initial_offsets}")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  WARNING: could not load {save_path}: {e}")

    grid_cols, grid_rows = grid_layout(len(cams_present))

    # Audio clap peak per cam (frame index in the synced timeline) — shown
    # as clickable markers on the frame slider so the user can jump straight
    # to the clap moment for visual sync verification.
    episode_meta = load_episode_metadata(BASE, EPISODE)
    peak_frames = compute_peak_frames(episode_meta, fps)

    META.update({
        "episode": EPISODE,
        "ref_cam": REF_CAM,
        "fps": fps,
        "total_frames": total_frames,
        "cams": cams_present,
        "grid": {"cols": grid_cols, "rows": grid_rows},
        "offsets": initial_offsets,
        "peak_frames": {f"cam{c}": pf for c, pf in peak_frames.items()},
        "_base": BASE,
    })

    global USE_PROXY
    USE_PROXY = not args.no_proxy

    global SAVE_PATH
    SAVE_PATH = save_path

    url = f"http://127.0.0.1:{args.port}/"
    print("=" * 60)
    print(f"  Sync visualizer ready")
    print(f"  Cameras: {cams_present} (ref=cam{REF_CAM})")
    print(f"  Frames:  {total_frames} @ {fps:.2f} fps "
          f"({total_frames/fps:.1f}s)")
    print(f"  URL:     {url}")
    print(f"  Save to: {save_path}")
    print(f"  Press Ctrl+C to stop")
    print("=" * 60)

    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    app.run(host="127.0.0.1", port=args.port,
            debug=False, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
