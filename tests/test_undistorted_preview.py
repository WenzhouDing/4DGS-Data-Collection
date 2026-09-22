"""Verify grid timing and camera correspondence using actual tiny encoded videos."""
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

import make_undistorted_preview as preview
from test_video_export import ffmpeg, presentation_times


SOURCE_FPS = Fraction(120000, 1001)
PREVIEW_FPS = Fraction(30000, 1001)
SOURCE_WIDTH, SOURCE_HEIGHT = 128, 72


def make_camera_video(path, camera_index, count=9, fps=SOURCE_FPS, audio=False):
    # Each camera/frame combination has a distinct intensity away from the labels.
    frames = np.stack([np.full((SOURCE_HEIGHT, SOURCE_WIDTH, 3),
                               20+camera_index*18+index*12, np.uint8) for index in range(count)])
    args = ["-f", "rawvideo", "-pixel_format", "bgr24", "-video_size",
            f"{SOURCE_WIDTH}x{SOURCE_HEIGHT}", "-framerate", fps, "-i", "pipe:0"]
    if audio:
        args += ["-f", "lavfi", "-i",
                 f"sine=frequency={440+camera_index*220}:sample_rate=48000:duration={float(count/fps)}",
                 "-c:a", "aac"]
    args += ["-c:v", "libx264", "-threads", "1", "-preset", "ultrafast", "-crf", "0",
             "-pix_fmt", "yuv420p", "-bf", "0", "-color_range", "tv", "-colorspace", "bt709",
             "-color_primaries", "bt709", "-color_trc", "bt709",
             "-r", fps, "-fps_mode", "cfr",
             "-video_track_timescale", fps.numerator, path]
    ffmpeg(*args, data=frames.tobytes())


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg tools unavailable")
class UndistortedPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        encoders = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"], stderr=subprocess.DEVNULL)
        if b"libx264 " not in encoders:
            raise unittest.SkipTest("FFmpeg libx264 encoder unavailable")

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.episode = self.base/"output/episode_0004"
        self.videos = self.episode/"synced_undistorted"
        self.videos.mkdir(parents=True)
        self.output = self.episode/"episode_0004_undistorted_preview.mp4"
        self.manifest_path = self.videos/"export_manifest.json"
        self.calibration = self.episode/"calibration_undistorted/calibration_all_cameras.json"
        self.calibration.parent.mkdir()
        self.args = ["--base", str(self.base), "--episode", "episode_0004", "--decode-accel", "none"]

    def fixture(self, names=("cam1", "cam2"), counts=None, rates=None, audio=False):
        counts, rates = counts or {}, rates or {}
        paths = {}
        for index, name in enumerate(names):
            path = self.videos/f"{name}_synced_undistorted.mp4"
            make_camera_video(path, index, counts.get(name, 9), rates.get(name, SOURCE_FPS), audio)
            paths[name] = path
        # Nonzero historical source offsets have already been applied to these videos.
        # The preview must sample their displayed frames directly, without applying them again.
        manifest = {"image_space": "undistorted", "partial_export": False,
                    "frame_count": 9, "fps": str(SOURCE_FPS), "logical_start_frame": 11,
                    "frame_offsets": {name: (-1 if name == "cam2" else 0) for name in names},
                    "cameras": {name: {} for name in reversed(names)},
                    "output_signatures": {path.name: preview.signature(path) for path in paths.values()}}
        self.manifest_path.write_text(json.dumps(manifest))
        self.calibration.write_text(json.dumps({"reference_camera": names[0]}))
        return paths

    def run_preview(self):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(preview.main(self.args), 0)

    def test_actual_multirow_grid_samples_frames_zero_four_eight_in_every_camera(self):
        names = ("cam1", "cam2", "cam3", "cam4", "cam10")
        paths = self.fixture(names)
        source_hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
        self.run_preview()
        report = preview.read_json(self.output.with_suffix(".json"))
        self.assertEqual(report["cameras_row_major"], list(names))
        self.assertEqual(report["grid"], [3, 2])
        self.assertEqual(report["source_frame_stride"], 4)
        self.assertEqual(report["source_frame_count"], 9)
        self.assertEqual(report["output_probe"]["frame_count"], 3)
        self.assertEqual(report["output_probe"]["fps"], str(PREVIEW_FPS))
        self.assertEqual(presentation_times(self.output), [i/PREVIEW_FPS for i in range(3)])
        raw = ffmpeg("-i", self.output, "-map", "0:v:0", "-an", "-fps_mode", "passthrough",
                     "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1")
        frames = np.frombuffer(raw, np.uint8).reshape(3, 720, 1920, 3)
        for index, name in enumerate(names):
            x, y = index % 3*640, index // 3*360
            means = frames[:, y+150:y+210, x+280:x+360].mean(axis=(1, 2, 3))
            np.testing.assert_allclose(means, [20+index*18+frame*12 for frame in (0, 4, 8)], atol=4.,
                                       err_msg=f"Wrong camera or source-frame selection in {name}")
            self.assertEqual(hashlib.sha256(paths[name].read_bytes()).hexdigest(), source_hashes[name])
        self.assertLess(frames[:, 510:570, 1560:1640].mean(), 3., "Unused grid tile must be black")
        poster = cv2.imread(str(self.output.with_suffix(".jpg")))
        self.assertEqual(poster.shape[:2], (720, 1920))
        # Exact reuse keeps the bytes and timestamps of all published preview artifacts.
        artifacts = [self.output, self.output.with_suffix(".jpg"), self.output.with_suffix(".json")]
        before = {path: (path.read_bytes(), preview.signature(path)) for path in artifacts}
        with patch.object(preview, "ffmpeg", side_effect=AssertionError("Unexpected preview rebuild")):
            self.run_preview()
        self.assertEqual({path: (path.read_bytes(), preview.signature(path)) for path in artifacts}, before)

    def test_changed_source_signature_is_rejected_before_decoding(self):
        paths = self.fixture()
        with paths["cam2"].open("ab") as stream:
            stream.write(b"changed")
        with patch.object(preview, "probe_video") as probe:
            with self.assertRaisesRegex(ValueError, "changed since validation"):
                preview.main(self.args)
        probe.assert_not_called()
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.episode.glob(".preview-staging-*")), [])

    def test_reference_camera_change_rebuilds_with_the_correct_audio_track(self):
        paths = self.fixture(audio=True)
        before_sources = {name: path.read_bytes() for name, path in paths.items()}

        def audio_peak_hz():
            raw = ffmpeg("-i", self.output, "-map", "0:a:0", "-vn", "-ac", "1",
                         "-ar", "48000", "-f", "f32le", "pipe:1")
            samples = np.frombuffer(raw, np.float32)
            self.assertGreater(len(samples), 1024)
            spectrum = np.abs(np.fft.rfft(samples*np.hanning(len(samples))))
            return np.fft.rfftfreq(len(samples), 1/48000)[np.argmax(spectrum)]

        self.run_preview()
        report = preview.read_json(self.output.with_suffix(".json"))
        self.assertEqual(report["audio_camera"], "cam1")
        self.assertTrue(report["output_probe"]["has_audio"])
        self.assertAlmostEqual(audio_peak_hz(), 440., delta=30.)
        original_preview = self.output.read_bytes()
        self.calibration.write_text(json.dumps({"reference_camera": "cam2"}))
        self.run_preview()
        updated = preview.read_json(self.output.with_suffix(".json"))
        self.assertEqual(updated["audio_camera"], "cam2")
        self.assertNotEqual(updated["fingerprint"], report["fingerprint"])
        self.assertNotEqual(self.output.read_bytes(), original_preview)
        self.assertAlmostEqual(audio_peak_hz(), 660., delta=30.)
        self.assertEqual(presentation_times(self.output), [i/PREVIEW_FPS for i in range(3)])
        self.assertEqual({name: path.read_bytes() for name, path in paths.items()}, before_sources)

    def test_mismatched_counts_and_fps_are_rejected_before_staging(self):
        for mismatch in ("count", "fps"):
            with self.subTest(mismatch=mismatch):
                self.fixture(counts={"cam2": 8} if mismatch == "count" else None,
                             rates={"cam2": Fraction(60000, 1001)} if mismatch == "fps" else None)
                with self.assertRaisesRegex(ValueError, "does not share the export timeline"):
                    preview.main(self.args)
                self.assertFalse(self.output.exists())
                self.assertEqual(list(self.episode.glob(".preview-staging-*")), [])

    def test_partial_or_distorted_input_manifest_is_rejected(self):
        self.fixture()
        original = preview.read_json(self.manifest_path)
        for updates in ({"partial_export": True}, {"image_space": "distorted"}):
            with self.subTest(updates=updates):
                self.manifest_path.write_text(json.dumps({**original, **updates}))
                with self.assertRaisesRegex(ValueError, "complete undistorted export"):
                    preview.main(self.args)
                self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
