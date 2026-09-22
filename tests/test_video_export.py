"""Real tiny-video export checks plus publication failure/rollback tests."""
from contextlib import redirect_stdout
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import export_calibrated_videos as exporter
from calibration_geometry import fundamental_from_KRT
from test_calibration_validation import valid_fixture, snapshot, write_json


WIDTH, HEIGHT = 160, 120
FPS = Fraction(30000, 1001)
K = np.array([[115., 0., 80.], [0., 117., 60.], [0., 0., 1.]])
DIST = np.array([.24, .015, .002, -.003, 0.])


def ffmpeg(*args, data=None):
    return subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                           "-y", *map(str, args)], input=data, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=True).stdout


def make_video(path, count=10, fps=FPS, pixel_format="yuv420p", gop=30):
    """Monochrome spatial texture plus an unambiguous per-frame intensity marker."""
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    frames = []
    for index in range(count):
        gray = np.clip(120 + 65*np.sin((xx+index*3)*.21)
                       + 45*np.cos((yy-index*2)*.24), 0, 255).astype(np.uint8)
        gray[45:75, 65:95] = 20+index*12
        frames.append(np.repeat(gray[..., None], 3, axis=2))
    ffmpeg("-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", f"{WIDTH}x{HEIGHT}",
           "-framerate", fps, "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "1",
           "-preset", "ultrafast", "-crf", "0", "-pix_fmt", pixel_format,
           "-g", gop, "-bf", "0", "-color_range", "tv", "-colorspace", "bt709",
           "-color_primaries", "bt709", "-color_trc", "bt709",
           "-video_track_timescale", fps.numerator, path,
           data=np.stack(frames).tobytes())


def decode_video(path):
    raw = ffmpeg("-threads", "1", "-i", path, "-map", "0:v:0", "-an", "-sn", "-dn",
                 "-fps_mode", "passthrough", "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1")
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, HEIGHT, WIDTH, 3)


def presentation_times(path):
    result = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_frames",
        "-show_entries", "stream=time_base:frame=best_effort_timestamp", "-of", "json", str(path)]))
    tick = Fraction(result["streams"][0]["time_base"])
    return [int(frame["best_effort_timestamp"])*tick for frame in result["frames"]]


def make_calibration(path, episode):
    valid_fixture(path)
    combined = exporter.read_json(path/"calibration_all_cameras.json")
    combined["source_episode"] = episode
    combined["source_frame_offsets"] = {"cam1": 0, "cam2": -1}
    for name, entry in combined["cameras"].items():
        entry.update(K=K.tolist(), dist=DIST.tolist(), image_size=[WIDTH, HEIGHT])
        ext = entry["extrinsics"]
        if name != "cam1":
            ext["F"] = fundamental_from_KRT(K, K, ext["R"], ext["T"]).tolist()
        write_json(path/f"{name}_intrinsics.json", {k: v for k, v in entry.items() if k != "extrinsics"})
        write_json(path/f"{name}_extrinsics.json", ext)
    write_json(path/"calibration_all_cameras.json", combined)
    split = exporter.read_json(path/"observation_split.json")
    split["source_frame_offsets"] = combined["source_frame_offsets"]
    write_json(path/"observation_split.json", split)
    return combined


class VideoRangeAndPublicationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_common_range_handles_positive_negative_offsets_and_shortest_camera(self):
        probes = {1: {"fps": str(FPS), "frame_count": 12},
                  2: {"fps": str(FPS), "frame_count": 11},
                  3: {"fps": str(FPS), "frame_count": 13}}
        self.assertEqual(exporter.common_frame_range(probes, {1: 2, 2: -1, 3: 0}), (1, 10))
        self.assertEqual(exporter.common_frame_range(probes, {1: 2, 2: -1, 3: 0}, 4), (1, 5))
        with self.assertRaisesRegex(ValueError, "positive"):
            exporter.common_frame_range(probes, {1: 0, 2: 0, 3: 0}, 0)
        with self.assertRaisesRegex(ValueError, "No common"):
            exporter.common_frame_range(probes, {1: 20, 2: 0, 3: 0})
        probes[2]["fps"] = "30"
        with self.assertRaisesRegex(ValueError, "frame rates differ"):
            exporter.common_frame_range(probes, {1: 0, 2: 0, 3: 0})

    def directories(self):
        paths = [self.root/name for name in ("stage_video", "stage_calibration", "videos", "calibration")]
        for directory in paths:
            directory.mkdir()
            (directory/"asset.bin").write_bytes(directory.name.encode())
        (paths[2]/"export_manifest.json").write_text("{}")
        (paths[3]/"calibration_all_cameras.json").write_text("{}")
        return paths

    def test_publication_preserves_entire_previous_outputs_in_backups(self):
        staged_video, staged_cal, videos, calibration = self.directories()
        before = {videos: snapshot(videos), calibration: snapshot(calibration)}
        staged = {videos: snapshot(staged_video), calibration: snapshot(staged_cal)}
        backups = exporter.publish_export(staged_video, staged_cal, videos, calibration)
        self.assertEqual(len(backups), 2)
        for target, backup in zip((videos, calibration), map(Path, backups)):
            self.assertEqual(snapshot(target), staged[target])
            self.assertEqual(snapshot(backup), before[target])

    def test_second_publication_failure_restores_both_previous_outputs(self):
        staged_video, staged_cal, videos, calibration = self.directories()
        before = {videos: snapshot(videos), calibration: snapshot(calibration)}
        original_rename = Path.rename

        def failing_rename(path, target):
            if path == staged_cal:
                raise OSError("injected second publication failure")
            return original_rename(path, target)

        with patch.object(Path, "rename", failing_rename):
            with self.assertRaisesRegex(OSError, "injected"):
                exporter.publish_export(staged_video, staged_cal, videos, calibration)
        for target in (videos, calibration):
            self.assertEqual(snapshot(target), before[target])
        self.assertEqual(list(self.root.glob("*.backup-*")), [])
        self.assertTrue(staged_cal.is_dir())

    def test_unrelated_destination_is_rejected_before_any_rename(self):
        paths = self.directories()
        (paths[3]/"calibration_all_cameras.json").unlink()
        before = snapshot(self.root)
        with self.assertRaisesRegex(ValueError, "unrelated"):
            exporter.publish_export(*paths)
        self.assertEqual(snapshot(self.root), before)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg tools unavailable")
class ActualVideoExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        encoders = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"], stderr=subprocess.DEVNULL)
        if b"libx264 " not in encoders:
            raise unittest.SkipTest("FFmpeg libx264 encoder unavailable")

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_main_exports_correct_frames_remaps_pixels_and_preserves_sources(self):
        episode_name = "episode_0007"
        episode = self.root/"output"/episode_name
        source_dir = episode/"synced_raw"
        source_dir.mkdir(parents=True)
        original_calibration = make_calibration(episode/"calibration", episode_name)
        calibration_before = snapshot(episode/"calibration")
        sources = {cam: source_dir/f"cam{cam}_synced.mp4" for cam in (1, 2)}
        make_video(sources[1], 9)
        make_video(sources[2], 10)
        source_hashes = {cam: hashlib.sha256(path.read_bytes()).hexdigest() for cam, path in sources.items()}
        arguments = ["--base", str(self.root), "--episode", episode_name, "--cams", "2",
                     "--encoder", "libx264", "--decode-accel", "none", "--workers", "1"]
        with redirect_stdout(io.StringIO()):
            self.assertEqual(exporter.main(arguments), 0)
        output_dir = episode/"synced_undistorted"
        manifest = exporter.read_json(output_dir/"export_manifest.json")
        self.assertEqual((manifest["logical_start_frame"], manifest["frame_count"]), (1, 8))
        self.assertFalse(manifest["partial_export"])
        self.assertEqual(manifest["frame_offsets"], {"cam1": 0, "cam2": -1})
        self.assertEqual(manifest["fps"], str(FPS))
        for cam, source in sources.items():
            with self.subTest(camera=cam):
                output = output_dir/f"cam{cam}_synced_undistorted.mp4"
                start = 1 if cam == 1 else 0
                self.assertEqual(manifest["cameras"][f"cam{cam}"]["source_start_frame"], start)
                before = decode_video(source)[start:start+8]
                expected = np.stack([cv2.undistort(frame, K, DIST, None, K) for frame in before])
                actual = decode_video(output)
                self.assertEqual(actual.shape, expected.shape)
                remapped_error = np.mean(np.abs(actual.astype(float)-expected.astype(float)))
                untouched_error = np.mean(np.abs(actual.astype(float)-before.astype(float)))
                self.assertLess(remapped_error, 3., "Export differs from independently undistorted source pixels")
                self.assertGreater(untouched_error, remapped_error*5,
                                   "Synthetic lens must expose an export that merely copies the source")
                np.testing.assert_allclose(actual[:, 58:62, 78:82].mean(axis=(1, 2, 3)),
                                           before[:, 58:62, 78:82].mean(axis=(1, 2, 3)), atol=3.)
                self.assertEqual(presentation_times(output), [i/FPS for i in range(8)])
                probe = exporter.probe_video(output)
                self.assertEqual((probe["frame_count"], probe["fps"]), (8, str(FPS)))
                self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), source_hashes[cam])
        self.assertEqual(snapshot(episode/"calibration"), calibration_before)
        processed = exporter.read_json(episode/"calibration_undistorted/calibration_all_cameras.json")
        self.assertEqual(processed["image_space"], "undistorted")
        self.assertEqual(processed["source_episode"], episode_name)
        self.assertEqual(processed["source_frame_offsets"], {"cam1": 0, "cam2": 0})
        self.assertEqual(processed["export_timeline"]["frame_offsets"], manifest["frame_offsets"])
        self.assertEqual(processed["extrinsics_convention"], "world_to_camera")
        for name, entry in processed["cameras"].items():
            self.assertEqual(entry["K"], original_calibration["cameras"][name]["K"])
            self.assertEqual(entry["dist"], [0.]*5)
            self.assertEqual(entry["extrinsics"], original_calibration["cameras"][name]["extrinsics"])
            sidecar = exporter.read_json(episode/f"calibration_undistorted/{name}_intrinsics.json")
            self.assertEqual(sidecar["K"], entry["K"])
            self.assertEqual(sidecar["dist"], entry["dist"])
        # Exact reuse is a no-op; a damaged processed sidecar must force regeneration.
        previous = snapshot(output_dir)
        with patch.object(exporter, "export_camera", side_effect=AssertionError("Unexpected re-export")):
            with redirect_stdout(io.StringIO()):
                self.assertEqual(exporter.main(arguments), 0)
        self.assertEqual(snapshot(output_dir), previous)
        damaged = episode/"calibration_undistorted/cam1_intrinsics.json"
        damaged.write_text("{}")
        with patch.object(exporter, "export_camera", wraps=exporter.export_camera) as export_call:
            with redirect_stdout(io.StringIO()):
                self.assertEqual(exporter.main(arguments), 0)
        self.assertEqual(export_call.call_count, 2)
        self.assertEqual(exporter.read_json(damaged)["K"], K.tolist())
        self.assertEqual(len(list(episode.glob("synced_undistorted.backup-*"))), 1)
        self.assertEqual(len(list(episode.glob("calibration_undistorted.backup-*"))), 1)

    def test_probe_refuses_real_ten_bit_video(self):
        source = self.root/"ten_bit.mp4"
        make_video(source, 2, pixel_format="yuv420p10le")
        with self.assertRaisesRegex(ValueError, "Unsupported pixel format.*10"):
            exporter.probe_video(source)

    def test_audio_trim_keeps_a_zero_based_common_duration(self):
        silent = self.root/"silent.mp4"
        source, output = self.root/"audio.mp4", self.root/"corrected.mp4"
        make_video(silent, 18)
        ffmpeg("-i", silent, "-f", "lavfi", "-i",
               f"sine=frequency=660:sample_rate=48000:duration={float(18/FPS)}",
               "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", source)
        source_before = source.read_bytes()
        probe = exporter.probe_video(source)
        self.assertTrue(probe["has_audio"])
        with redirect_stdout(io.StringIO()):
            exported = exporter.export_camera(
                source, output, {"K": K.tolist(), "dist": DIST.tolist(), "image_size": [WIDTH, HEIGHT]},
                probe, source_start=5, frame_count=8, encoder="libx264")
        self.assertTrue(exported["has_audio"])
        self.assertEqual(presentation_times(output), [i/FPS for i in range(8)])
        streams = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-show_streams", "-of", "json", str(output)]))["streams"]
        audio = next(stream for stream in streams if stream["codec_type"] == "audio")
        self.assertAlmostEqual(float(audio["start_time"]), 0., places=6)
        # AAC encodes 1024-sample blocks; its final block may pad the trimmed interval.
        self.assertLess(abs(float(audio["duration"])-float(8/FPS)), 1024/48000)
        self.assertEqual(source.read_bytes(), source_before)

    def test_main_invalid_calibration_fails_before_creating_output(self):
        episode = self.root/"output/episode_0007"
        make_calibration(episode/"calibration", "episode_0007")
        proof = episode/"calibration/heldout_epipolar.json"
        heldout = exporter.read_json(proof)
        heldout["passed"] = False
        write_json(proof, heldout)
        with self.assertRaises(ValueError):
            exporter.main(["--base", str(self.root), "--episode", "episode_0007", "--cams", "2",
                           "--encoder", "libx264", "--decode-accel", "none"])
        self.assertFalse((episode/"synced_undistorted").exists())
        self.assertFalse((episode/"calibration_undistorted").exists())
        self.assertEqual(list(episode.glob(".undistort-staging-*")), [])

    def test_probe_excludes_discarded_preroll_samples(self):
        original, trimmed = self.root/"original.mp4", self.root/"stream_copied.mp4"
        make_video(original, 18, gop=30)
        ffmpeg("-ss", float(5/FPS), "-i", original, "-t", float(8/FPS), "-map", "0:v:0",
               "-c", "copy", trimmed)
        decoded = decode_video(trimmed)
        probe = exporter.probe_video(trimmed)
        self.assertEqual(probe["frame_count"], len(decoded))
        self.assertEqual(probe["frame_count"], 8)
        self.assertGreater(int(probe["header_frame_count"]), probe["frame_count"])
        self.assertEqual(presentation_times(trimmed), [i/FPS for i in range(8)])


if __name__ == "__main__":
    unittest.main()
