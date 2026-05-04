#!/usr/bin/env python3
"""
3D camera-pose visualizer
=========================
Reads output/<episode>/calibration/calibration_all_cameras.json and shows each camera
as a frustum in the reference camera's coordinate frame. Hovering a camera
displays its intrinsics, extrinsic method, and quality metrics.

Architecture: Plotly 3D scatter, self-contained HTML, no server.

Color code:
  red     reference camera (origin)
  green   direct stereoCalibrate succeeded with rms ≤ quality bar
  amber   bridged via intermediate cameras
  gray    direct used despite low quality (no bridge available)

Usage:
    uv run python viz_calibration.py [--base DIR] [--out HTML_PATH] [--show]
"""

import argparse
import json
import os
import webbrowser

import numpy as np
import plotly.graph_objects as go


# ─── CLI ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="3D camera-pose visualizer")
    p.add_argument("--base", default=".", help="Project root")
    p.add_argument("--episode", default="episode_0001",
                   help="Episode whose calibration to visualize")
    p.add_argument("--out", default=None,
                   help="Output HTML path (default: <episode>/calibration/camera_poses.html)")
    p.add_argument("--show", action="store_true",
                   help="Open in browser after generating")
    p.add_argument("--frustum-depth", type=float, default=0.05,
                   help="Frustum depth in metres (visualization scale)")
    return p.parse_args()


# ─── GEOMETRY HELPERS ─────────────────────────────────────────────
def frustum_in_ref(R_inv, T_inv, K, image_size, depth):
    """Build the 5 vertices of a camera frustum in the reference frame.

    R_inv, T_inv: cam pose in ref's frame (i.e. T_inv = cam optical center
                  in ref coords, R_inv columns = cam axes in ref coords).
    Returns: (center, [tl, tr, br, bl]) in ref frame.
    """
    w, h = image_size
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z = depth
    # Image-plane corners projected forward to depth z, in cam coords.
    corners_cam = np.array([
        [(0 - cx) * z / fx, (0 - cy) * z / fy, z],     # top-left
        [(w - cx) * z / fx, (0 - cy) * z / fy, z],     # top-right
        [(w - cx) * z / fx, (h - cy) * z / fy, z],     # bottom-right
        [(0 - cx) * z / fx, (h - cy) * z / fy, z],     # bottom-left
    ]).T  # (3, 4)
    corners_ref = (R_inv @ corners_cam + T_inv).T  # (4, 3)
    center_ref = T_inv.flatten()
    return center_ref, corners_ref


# ─── TRACE BUILDERS ───────────────────────────────────────────────
COLOR_REF = "#f85149"          # red
COLOR_DIRECT = "#3fb950"       # green
COLOR_BRIDGED = "#fbbf24"      # amber
COLOR_LOWQ = "#7d8590"         # gray


def color_for(method, is_ref):
    if is_ref:
        return COLOR_REF
    if not method:
        return COLOR_LOWQ
    if method.startswith("direct (low quality"):
        return COLOR_LOWQ
    if method.startswith("direct"):
        return COLOR_DIRECT
    if method.startswith("bridged"):
        return COLOR_BRIDGED
    return COLOR_LOWQ


def hover_html(cam_id, center, cd, is_ref):
    """Hover tooltip for a camera."""
    K = np.array(cd["K"]) if cd.get("K") else None
    rows = [f"<b>cam{cam_id}</b>" + (" (REFERENCE)" if is_ref else "")]
    rows.append(f"position (m, ref frame): "
                f"({center[0]:+.4f}, {center[1]:+.4f}, {center[2]:+.4f})")
    if cd.get("baseline_m") is not None:
        rows.append(f"baseline: {cd['baseline_m']:.4f} m")
    if cd.get("intrinsic_rms") is not None:
        rows.append(f"intrinsic RMS: {cd['intrinsic_rms']:.4f} px")
    if cd.get("stereo_rms_px") is not None:
        rows.append(f"stereo RMS: {cd['stereo_rms_px']:.4f} px")
    if K is not None:
        rows.append(f"K: fx={K[0,0]:.1f}  fy={K[1,1]:.1f}")
        rows.append(f"   cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")
    if cd.get("method"):
        rows.append(f"method: {cd['method']}")
    if cd.get("path") and len(cd["path"]) > 2:
        rows.append("path: " + " → ".join(map(str, cd["path"])))
    return "<br>".join(rows)


def build_traces(camera_data, ref_cam, frustum_depth):
    traces = []
    centers = []  # for legend coloring + reach checking
    for cam_id in sorted(camera_data.keys()):
        cd = camera_data[cam_id]
        is_ref = (cam_id == ref_cam)

        if is_ref:
            R_inv = np.eye(3)
            T_inv = np.zeros((3, 1))
        elif cd.get("R") is None:
            # No extrinsics for this cam — skip
            continue
        else:
            R_inv = np.array(cd["R"])
            T_inv = np.array(cd["T"]).reshape(3, 1)

        if not cd.get("K"):
            continue
        K = np.array(cd["K"])
        image_size = cd.get("image_size", [3840, 2160])

        center, corners = frustum_in_ref(R_inv, T_inv, K, image_size, frustum_depth)
        centers.append((cam_id, center))
        color = color_for(cd.get("method"), is_ref)
        hover = hover_html(cam_id, center, cd, is_ref)

        # Camera marker + label
        traces.append(go.Scatter3d(
            x=[center[0]], y=[center[1]], z=[center[2]],
            mode="markers+text",
            marker=dict(size=8, color=color,
                        line=dict(color="rgba(255,255,255,0.4)", width=1)),
            text=[f"cam{cam_id}"],
            textposition="top center",
            textfont=dict(color="#e6edf3", size=11),
            hovertext=hover,
            hoverinfo="text",
            name=f"cam{cam_id}",
            showlegend=False,
        ))

        # Frustum lines: 4 from center to corners + 4 around image plane
        line_x, line_y, line_z = [], [], []
        for corner in corners:
            line_x.extend([center[0], corner[0], None])
            line_y.extend([center[1], corner[1], None])
            line_z.extend([center[2], corner[2], None])
        for i in range(4):
            a, b = corners[i], corners[(i + 1) % 4]
            line_x.extend([a[0], b[0], None])
            line_y.extend([a[1], b[1], None])
            line_z.extend([a[2], b[2], None])
        traces.append(go.Scatter3d(
            x=line_x, y=line_y, z=line_z,
            mode="lines",
            line=dict(color=color, width=3),
            hoverinfo="skip",
            showlegend=False,
        ))
    return traces, centers


def add_axes_at_origin(traces, length=0.10):
    """Add small XYZ axes at the reference origin for orientation."""
    for axis_idx, color, label in [(0, "#ff5555", "X"),
                                   (1, "#55ff55", "Y"),
                                   (2, "#5599ff", "Z")]:
        end = [0.0, 0.0, 0.0]
        end[axis_idx] = length
        traces.append(go.Scatter3d(
            x=[0, end[0]], y=[0, end[1]], z=[0, end[2]],
            mode="lines+text",
            line=dict(color=color, width=4),
            text=["", label],
            textposition="middle right",
            textfont=dict(color=color, size=10),
            hoverinfo="skip",
            showlegend=False,
        ))


def add_legend_dummies(traces):
    """Show color key as a tiny invisible-positioned legend."""
    for color, label in [(COLOR_REF, "reference"),
                         (COLOR_DIRECT, "direct (good)"),
                         (COLOR_BRIDGED, "bridged"),
                         (COLOR_LOWQ, "direct (low quality)")]:
        traces.append(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode="markers",
            marker=dict(size=8, color=color),
            name=label,
            showlegend=True,
        ))


# ─── MAIN ─────────────────────────────────────────────────────────
def main():
    args = parse_args()
    base = os.path.abspath(args.base)
    episode = args.episode
    calib_path = os.path.join(base, "output", episode, "calibration",
                              "calibration_all_cameras.json")
    if not os.path.exists(calib_path):
        print(f"ERROR: {calib_path} not found.")
        print(f"  Run run_calibration.py --episode {episode} first.")
        raise SystemExit(1)

    with open(calib_path) as f:
        calib = json.load(f)

    # Load per-cam data
    camera_data = {}
    ref_cam = None
    for cam_key, cam_meta in calib.get("cameras", {}).items():
        if not cam_key.startswith("cam"):
            continue
        try:
            cam_id = int(cam_key[3:])
        except ValueError:
            continue
        cd = {
            "K": cam_meta.get("K"),
            "image_size": cam_meta.get("image_size"),
            "intrinsic_rms": cam_meta.get("rms_error_px"),
        }
        ext = cam_meta.get("extrinsics")
        if ext:
            cd.update({
                "R": ext.get("R"),
                "T": ext.get("T"),
                "stereo_rms_px": ext.get("stereo_rms_px"),
                "baseline_m": ext.get("baseline_m"),
                "method": ext.get("method"),
                "path": ext.get("path"),
            })
            if ref_cam is None and ext.get("reference"):
                ref_cam = int(ext["reference"][3:])
        else:
            cd.update({"R": None, "T": None})
        camera_data[cam_id] = cd

    if ref_cam is None:
        # No extrinsics anywhere — fall back to cam1
        ref_cam = min(camera_data.keys()) if camera_data else 1

    # Build figure
    traces, centers = build_traces(camera_data, ref_cam, args.frustum_depth)
    add_axes_at_origin(traces, length=args.frustum_depth * 1.5)
    add_legend_dummies(traces)

    n_total = len(camera_data)
    n_with_pose = sum(1 for c in camera_data.values()
                      if c.get("R") is not None) + (1 if ref_cam in camera_data else 0)

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=(f"Camera Poses — {n_with_pose}/{n_total} cams placed,"
                  f" reference = cam{ref_cam}"),
            font=dict(size=14, color="#e6edf3"),
        ),
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
            xaxis=dict(gridcolor="#30363d", color="#7d8590",
                       backgroundcolor="#0e1116"),
            yaxis=dict(gridcolor="#30363d", color="#7d8590",
                       backgroundcolor="#0e1116"),
            zaxis=dict(gridcolor="#30363d", color="#7d8590",
                       backgroundcolor="#0e1116"),
            aspectmode="data",
            bgcolor="#0e1116",
        ),
        paper_bgcolor="#0e1116",
        plot_bgcolor="#0e1116",
        font=dict(color="#e6edf3"),
        margin=dict(l=0, r=0, t=44, b=0),
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(22, 27, 34, 0.85)",
            bordercolor="#30363d", borderwidth=1,
            font=dict(color="#e6edf3", size=11),
        ),
    )

    out_path = args.out or os.path.join(
        base, "output", episode, "calibration", "camera_poses.html")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Embed plotly.js so the file works fully offline
    fig.write_html(out_path, include_plotlyjs=True)
    print(f"Wrote {out_path}")
    print(f"  {n_with_pose}/{n_total} cameras placed (ref = cam{ref_cam})")

    # Print a small text summary of what's placed where
    print("\n  Camera positions (x, y, z) in metres, ref frame:")
    for cam_id, center in sorted(centers, key=lambda x: x[0]):
        cd = camera_data[cam_id]
        method = cd.get("method", "—") if cam_id != ref_cam else "reference"
        print(f"    cam{cam_id:>2}: ({center[0]:+.4f}, {center[1]:+.4f}, "
              f"{center[2]:+.4f})  [{method}]")

    if args.show:
        webbrowser.open(f"file://{out_path}")


if __name__ == "__main__":
    main()
