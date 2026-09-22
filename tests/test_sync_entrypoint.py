"""Exercise sync selection and fail-closed publication without expensive decoding."""
from contextlib import ExitStack, redirect_stdout
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import sync_pipeline as sync


class SyncEntrypointTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.commands = []
        self.audio_requests = []
        for cam in (1, 2):
            folder = self.base/str(cam)
            folder.mkdir()
            for episode in (1, 2):
                (folder/f"GX01{episode:04d}.MP4").write_bytes(b"untouched raw recording")

    def run_sync(self, *extra, confidence=.9, video_failure=False):
        def command(args, **kwargs):
            self.commands.append(list(map(str, args)))
            if args[0] == "ffprobe":
                return subprocess.CompletedProcess(args, 0, stdout="3.0\n", stderr="")
            if video_failure:
                raise subprocess.CalledProcessError(7, args)
            Path(args[-1]).write_bytes(b"synthetic ffmpeg output")
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        def extract_audio(paths, wav, work_dir, tag):
            self.audio_requests.append(tag)

        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(patch.object(sync.subprocess, "run", side_effect=command))
            stack.enter_context(patch.object(sync, "probe_creation_time", side_effect=lambda path:
                datetime(2026, 1, 1, tzinfo=timezone.utc)+timedelta(seconds=int(Path(path).stem[-4:])*10)))
            stack.enter_context(patch.object(sync, "probe_video_fps", return_value=120000/1001))
            stack.enter_context(patch.object(sync, "extract_audio", side_effect=extract_audio))
            stack.enter_context(patch.object(sync, "load_audio", return_value=(48000, np.ones(1000))))
            stack.enter_context(patch.object(sync, "cross_correlate_offset", return_value=(0, confidence)))
            stack.enter_context(patch.object(sync, "find_clap_peak", return_value=(100, 1.)))
            stack.enter_context(patch.object(sync, "get_video_metadata", return_value={"width": 128, "height": 72}))
            return sync.main(["--base", str(self.base), "--cams", "2", *extra])

    def metadata(self, episode):
        name = f"episode_{episode:04d}"
        return json.loads((self.base/"output"/name/"metadata"/f"{name}_metadata.json").read_text())

    def test_only_selected_episode_is_processed_without_renumbering(self):
        self.assertEqual(self.run_sync("--episodes", "2"), 0)
        self.assertFalse((self.base/"output/episode_0001").exists())
        self.assertEqual(self.audio_requests, ["episode_0002_cam1", "episode_0002_cam2"])
        episode = self.base/"output/episode_0002"
        self.assertTrue((episode/"synced_raw/cam1_synced.mp4").is_file())
        self.assertTrue((episode/"synced_raw/cam2_synced.mp4").is_file())
        data = self.metadata(2)
        self.assertEqual(data["episode_name"], "episode_0002")
        self.assertEqual(data["cameras"]["cam1"]["source_files"], ["GX010002.MP4"])
        self.assertEqual(data["sync_validation"], {
            "passed": True, "confidence_threshold": .3,
            "camera_confidences": {"cam1": 1., "cam2": .9}, "failed_cameras": []})
        for path in self.base.glob("[12]/*.MP4"):
            self.assertEqual(path.read_bytes(), b"untouched raw recording")

    def test_default_processes_all_episodes_and_selected_reference(self):
        self.assertEqual(self.run_sync("--ref-cam", "2"), 0)
        self.assertEqual(len(self.audio_requests), 4)
        for episode in (1, 2):
            data = self.metadata(episode)
            self.assertEqual(data["reference_camera"], 2)
            self.assertEqual(data["sync_validation"]["camera_confidences"], {"cam1": .9, "cam2": 1.})
            self.assertTrue(data["sync_validation"]["passed"])

    def test_invalid_duplicate_nonpositive_or_missing_selection_does_not_process_video(self):
        for values in (("1", "1"), ("0",), ("-1",), ("3",)):
            with self.subTest(indices=values):
                with self.assertRaises(SystemExit) as error:
                    self.run_sync("--episodes", *values)
                self.assertEqual(error.exception.code, 1)
                self.assertEqual(self.audio_requests, [])
                self.assertEqual(list(self.base.glob("output/episode_*")), [])
        self.assertFalse(any(command[0] == "ffmpeg" for command in self.commands))

    def test_different_recording_counts_fail_even_when_first_episode_is_requested(self):
        (self.base/"2/GX010002.MP4").unlink()
        with self.assertRaises(SystemExit) as error:
            self.run_sync("--episodes", "1")
        self.assertEqual(error.exception.code, 1)
        self.assertEqual(self.audio_requests, [])
        self.assertEqual(list(self.base.glob("output/episode_*")), [])

    def test_missing_camera_recordings_fail_before_any_episode_is_written(self):
        for path in (self.base/"2").glob("*.MP4"):
            path.unlink()
        with self.assertRaises(SystemExit) as error:
            self.run_sync()
        self.assertEqual(error.exception.code, 1)
        self.assertEqual(self.audio_requests, [])
        self.assertEqual(list(self.base.glob("output/episode_*")), [])

    def test_low_boundary_and_nonfinite_confidence_write_failed_diagnostics_without_video(self):
        for confidence in (0., .3, float("nan"), float("inf")):
            with self.subTest(confidence=confidence):
                with self.assertRaises(SystemExit) as error:
                    self.run_sync("--episodes", "1", confidence=confidence)
                self.assertEqual(error.exception.code, 1)
                data = self.metadata(1)["sync_validation"]
                self.assertFalse(data["passed"])
                self.assertEqual(data["failed_cameras"], ["cam2"])
                self.assertEqual(data["confidence_threshold"], .3)
                expected = confidence if np.isfinite(confidence) else None
                self.assertEqual(data["camera_confidences"]["cam2"], expected)
                self.assertFalse((self.base/"output/episode_0001/synced_raw").exists())
                self.assertFalse((self.base/"output/episode_0002").exists())
        self.assertFalse(any(command[0] == "ffmpeg" for command in self.commands))

    def test_confidence_above_boundary_passes(self):
        self.assertEqual(self.run_sync("--episodes", "1", confidence=.300001), 0)
        self.assertTrue(self.metadata(1)["sync_validation"]["passed"])

    def test_ffmpeg_failure_is_not_reported_as_success(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_sync("--episodes", "1", video_failure=True)
        self.assertFalse((self.base/"output/episode_0001/metadata/episode_0001_metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
