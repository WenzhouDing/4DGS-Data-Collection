"""Saved-settings CLI regressions; heavy video/calibration work is mocked."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import pipeline


SETTINGS = """[project]
data_dir = "../data with spaces"
cameras = 2
reference_camera = 1

[calibration]
episode = 2
board = "9x12"
square_size_m = 0.03
sample_every = 90
max_frames = 0

[run]
episodes = [4]

[frame_offsets.episode_0002]
cam2 = -1
"""


def tree_snapshot(root):
    """Include new/removed directories and file bytes, not only visible outputs."""
    return {str(p.relative_to(root)): ("symlink", str(p.readlink())) if p.is_symlink()
            else ("directory",) if p.is_dir() else ("file", p.read_bytes())
            for p in root.rglob("*")}


class PipelineCliTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="pipeline test ")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.config_dir = self.root / "settings with spaces"
        self.data = self.root / "data with spaces"
        self.config_dir.mkdir()
        self.data.mkdir()
        self.config = self.config_dir / "my pipeline.toml"
        self.config.write_text(SETTINGS)
        # A dry run may inspect discovery inputs, but must not read these as media.
        for cam in (1, 2):
            folder = self.data / str(cam)
            folder.mkdir()
            (folder / "GX010001.MP4").write_bytes(b"raw capture untouched")

    def invoke(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                code = pipeline.main(["--config", str(self.config), *args])
            except SystemExit as exc:
                code = exc.code
        return code or 0, stdout.getvalue() + stderr.getvalue()

    @contextlib.contextmanager
    def forbid_commands(self):
        with patch.object(subprocess, "run", side_effect=AssertionError("dry run launched a subprocess")), \
             patch.object(subprocess, "Popen", side_effect=AssertionError("dry run launched a subprocess")), \
             patch.object(subprocess, "check_output", side_effect=AssertionError("dry run launched a subprocess")):
            yield

    def test_default_command_is_run_and_dry_run_does_not_mutate_or_execute(self):
        before = tree_snapshot(self.root)
        with self.forbid_commands():
            default_code, default_text = self.invoke("--dry-run")
            explicit_code, explicit_text = self.invoke("run", "--dry-run")
        self.assertEqual(default_code, 0, default_text)
        self.assertEqual(explicit_code, 0, explicit_text)
        self.assertEqual(default_text, explicit_text)
        self.assertEqual(tree_snapshot(self.root), before)
        self.assertIn(str(self.data), default_text)
        self.assertIn("episode_0004", default_text)

    def test_numeric_and_named_episode_overrides_are_equivalent(self):
        with self.forbid_commands():
            numeric_code, numeric = self.invoke("--dry-run", "--episode", "3", "4")
            named_code, named = self.invoke("--dry-run", "--episode", "episode_0003", "episode_0004")
        self.assertEqual(numeric_code, 0, numeric)
        self.assertEqual(named_code, 0, named)
        self.assertEqual(numeric, named)
        self.assertIn("episode_0003", numeric)
        self.assertIn("episode_0004", numeric)

    def test_invalid_episode_overrides_fail_before_work(self):
        for episodes in (("0",), ("-1",), ("episode_0000",), ("banana",)):
            with self.subTest(episodes=episodes), self.forbid_commands():
                before = tree_snapshot(self.root)
                code, output = self.invoke("--dry-run", "--episode", *episodes)
                self.assertNotEqual(code, 0, output)
                self.assertEqual(tree_snapshot(self.root), before)

    def test_invalid_saved_settings_are_rejected_before_any_work(self):
        cases = {
            "unknown top-level table": SETTINGS + "\n[typo]\nvalue = 1\n",
            "unknown project key": SETTINGS.replace("cameras = 2", "cameras = 2\ncamera_count = 2"),
            "unknown calibration key": SETTINGS.replace("max_frames = 0", "max_frames = 0\nmax_frame = 12"),
            "unknown run key": SETTINGS.replace("episodes = [4]", "episodes = [4]\nepisode = 4"),
            "camera count below minimum": SETTINGS.replace("cameras = 2", "cameras = 1"),
            "boolean camera count": SETTINGS.replace("cameras = 2", "cameras = true"),
            "reference outside camera range": SETTINGS.replace("reference_camera = 1", "reference_camera = 3"),
            "boolean reference": SETTINGS.replace("reference_camera = 1", "reference_camera = true"),
            "zero calibration episode": SETTINGS.replace("episode = 2", "episode = 0"),
            "malformed board": SETTINGS.replace('board = "9x12"', 'board = "9by12"'),
            "zero board dimension": SETTINGS.replace('board = "9x12"', 'board = "0x12"'),
            "zero square size": SETTINGS.replace("square_size_m = 0.03", "square_size_m = 0"),
            "nonfinite square size": SETTINGS.replace("square_size_m = 0.03", "square_size_m = nan"),
            "boolean square size": SETTINGS.replace("square_size_m = 0.03", "square_size_m = true"),
            "zero sample interval": SETTINGS.replace("sample_every = 90", "sample_every = 0"),
            "fractional sample interval": SETTINGS.replace("sample_every = 90", "sample_every = 2.5"),
            "negative maximum frames": SETTINGS.replace("max_frames = 0", "max_frames = -1"),
            "boolean maximum frames": SETTINGS.replace("max_frames = 0", "max_frames = false"),
            "empty episode selection": SETTINGS.replace("episodes = [4]", "episodes = []"),
            "zero selected episode": SETTINGS.replace("episodes = [4]", "episodes = [0]"),
            "boolean selected episode": SETTINGS.replace("episodes = [4]", "episodes = [true]"),
            "offset for nonexistent camera": SETTINGS.replace("cam2 = -1", "cam3 = -1"),
            "malformed offset camera name": SETTINGS.replace("cam2 = -1", "camera2 = -1"),
            "fractional offset": SETTINGS.replace("cam2 = -1", "cam2 = -0.5"),
            "boolean offset": SETTINGS.replace("cam2 = -1", "cam2 = false"),
            "zero offset episode": SETTINGS.replace("frame_offsets.episode_0002", "frame_offsets.episode_0000"),
        }
        for label, contents in cases.items():
            with self.subTest(setting=label), self.forbid_commands():
                self.config.write_text(contents)
                before = tree_snapshot(self.root)
                code, output = self.invoke("--dry-run")
                self.assertNotEqual(code, 0, output)
                self.assertTrue(output.strip(), "Invalid settings need a diagnostic")
                self.assertEqual(tree_snapshot(self.root), before)

    def test_duplicate_episode_spellings_are_deduplicated(self):
        settings = pipeline.load_settings(self.config, ["4", "episode_0004", "2", "episode_0002"])
        self.assertEqual(settings.episodes, ["episode_0004", "episode_0002"])
        self.assertEqual(pipeline.required_episodes(settings), ["episode_0002", "episode_0004"])

    def populate_existing(self, episodes=(2, 4)):
        combined = {
            "checkerboard": {"board_squares": [9, 12], "square_size_m": .03},
            "source_episode": "episode_0002",
            "source_frame_offsets": {"cam1": 0, "cam2": -1},
        }
        for number in episodes:
            root = self.data / "output" / f"episode_{number:04d}"
            raw = root / "synced_raw"
            raw.mkdir(parents=True)
            for cam in (1, 2):
                (raw / f"cam{cam}_synced.mp4").write_bytes(f"unchanged raw {number}/{cam}".encode())
            calibration = root / "calibration"
            calibration.mkdir()
            (calibration / "calibration_all_cameras.json").write_text(json.dumps(combined))

    def test_settings_resolve_relative_to_config_and_scope_offsets_by_episode(self):
        settings = pipeline.load_settings(self.config)
        self.assertEqual(settings.data_dir, self.data.resolve())
        self.assertEqual(settings.frame_offsets, {"episode_0002": {"cam1": 0, "cam2": -1}})
        work = self.root / "scratch"
        work.mkdir()
        target_arguments = pipeline.offset_arguments(settings, "episode_0004", work)
        self.assertEqual(target_arguments[0], "--frame-offsets")
        self.assertEqual(json.loads(Path(target_arguments[1]).read_text()), {"cam1": 0, "cam2": 0})
        arguments = pipeline.offset_arguments(settings, "episode_0002", work)
        self.assertEqual(arguments[0], "--frame-offsets")
        self.assertEqual(json.loads(Path(arguments[1]).read_text()), {"cam1": 0, "cam2": -1})

    def test_resume_skips_completed_sync_and_calibration_and_preserves_path_arguments(self):
        self.populate_existing()
        before_raw = {p: p.read_bytes() for p in self.data.glob("output/*/synced_raw/*.mp4")}
        with patch.object(pipeline, "check_environment"), \
             patch.object(pipeline, "verify_inputs") as verify, \
             patch.object(pipeline, "calibration_matches", return_value=True), \
             patch.object(pipeline.Runner, "run", autospec=True) as run, self.forbid_commands():
            code, text = self.invoke("--cpu")
        self.assertEqual(code, 0, text)
        self.assertEqual([call.args[1] for call in run.call_args_list],
                         ["export_calibrated_videos.py", "make_undistorted_preview.py", "viz_calibration.py"])
        self.assertEqual([call.args[1] for call in verify.call_args_list], ["episode_0002", "episode_0004"])
        export = run.call_args_list[0].args
        self.assertEqual(export[export.index("--base") + 1], self.data)
        self.assertEqual(export[export.index("--episode") + 1], "episode_0004")
        self.assertEqual(export[export.index("--encoder") + 1], "libx264")
        self.assertEqual(export[export.index("--decode-accel") + 1], "none")
        self.assertEqual(json.loads(Path(export[export.index("--frame-offsets") + 1]).read_text()),
                         {"cam1": 0, "cam2": 0}, "Episode2 corrections must not leak into episode4")
        self.assertEqual(before_raw, {p: p.read_bytes() for p in before_raw})

    def test_failed_export_stops_before_preview_and_viewer(self):
        self.populate_existing()
        with patch.object(pipeline, "check_environment"), \
             patch.object(pipeline, "verify_inputs"), \
             patch.object(pipeline, "calibration_matches", return_value=True), \
             patch.object(pipeline.Runner, "run", autospec=True, side_effect=RuntimeError("export failed")) as run, \
             self.forbid_commands():
            code, text = self.invoke()
        self.assertEqual(code, 1)
        self.assertIn("export failed", text)
        self.assertEqual([call.args[1] for call in run.call_args_list], ["export_calibrated_videos.py"])

    def test_recalibrate_uses_only_source_offsets_and_checks_the_result(self):
        self.populate_existing()
        with patch.object(pipeline, "check_environment"), \
             patch.object(pipeline, "verify_inputs"), \
             patch.object(pipeline, "calibration_matches", return_value=True), \
             patch.object(pipeline.Runner, "run", autospec=True) as run, self.forbid_commands():
            code, text = self.invoke("--recalibrate")
        self.assertEqual(code, 0, text)
        fit = run.call_args_list[0].args
        self.assertEqual(fit[1], "run_calibration.py")
        self.assertEqual(fit[fit.index("--episode") + 1], "episode_0002")
        self.assertEqual(json.loads(Path(fit[fit.index("--frame-offsets") + 1]).read_text()),
                         {"cam1": 0, "cam2": -1})
        export = next(call.args for call in run.call_args_list if call.args[1] == "export_calibrated_videos.py")
        self.assertEqual(json.loads(Path(export[export.index("--frame-offsets") + 1]).read_text()),
                         {"cam1": 0, "cam2": 0})

    def test_invalid_fitted_calibration_stops_before_copy_or_export(self):
        self.populate_existing()
        with patch.object(pipeline, "check_environment"), \
             patch.object(pipeline, "verify_inputs"), \
             patch.object(pipeline, "calibration_matches", return_value=False), \
             patch.object(pipeline.Runner, "run", autospec=True) as run, self.forbid_commands():
            code, text = self.invoke("--recalibrate")
        self.assertEqual(code, 1)
        self.assertIn("Calibration did not pass", text)
        self.assertEqual([call.args[1] for call in run.call_args_list], ["run_calibration.py"])

    def test_incomplete_existing_episode_is_preserved_and_blocks_work(self):
        target = self.data / "output/episode_0004/synced_raw"
        target.mkdir(parents=True)
        (target / "cam1_synced.mp4").write_bytes(b"irreplaceable existing partial input")
        before = tree_snapshot(self.root)
        with self.forbid_commands():
            code, text = self.invoke("--dry-run")
        self.assertEqual(code, 1)
        self.assertIn("incomplete", text.lower())
        self.assertEqual(tree_snapshot(self.root), before)

    def staged_sync_runner(self, sync_passed=True):
        """Represent the sync child's outputs, while leaving promotion to the CLI."""
        def run(script, *arguments):
            self.assertEqual(script, "sync_pipeline.py")
            stage = Path(arguments[arguments.index("--base") + 1])
            for cam in (1, 2):
                self.assertTrue((stage / str(cam)).is_symlink())
                self.assertEqual((stage / str(cam)).resolve(), self.data / str(cam))
            requested = arguments[arguments.index("--episodes") + 1:]
            for number in requested:
                name = f"episode_{number:04d}"
                directory = stage / "output" / name
                (directory / "metadata").mkdir(parents=True)
                (directory / "metadata" / f"{name}_metadata.json").write_text(
                    json.dumps({"sync_validation": {"passed": sync_passed}}))
                (directory / "synced_raw").mkdir()
                for cam in (1, 2):
                    (directory / "synced_raw" / f"cam{cam}_synced.mp4").write_bytes(b"newly synchronized")
            return stage
        return Mock(side_effect=run)

    def test_fresh_sync_stages_only_requested_episodes_and_validates_all_before_promotion(self):
        settings = pipeline.load_settings(self.config)
        output = self.data / "output"
        output.mkdir()
        raw_before = {p: p.read_bytes() for p in self.data.glob("*/*.MP4")}
        run = self.staged_sync_runner()
        def verify(_settings, _name, stage):
            self.assertTrue(stage.name.startswith(".pipeline-sync-"))
            self.assertFalse((output / "episode_0002").exists())
            self.assertFalse((output / "episode_0004").exists())
        with patch.object(pipeline, "verify_inputs", side_effect=verify) as checked, self.forbid_commands():
            pipeline.synchronize_missing(settings, ["episode_0002", "episode_0004"], Mock(run=run))
        args = run.call_args.args
        self.assertEqual(args[args.index("--episodes") + 1:], (2, 4))
        self.assertEqual([call.args[1] for call in checked.call_args_list], ["episode_0002", "episode_0004"])
        for episode in (2, 4):
            self.assertEqual((output / f"episode_{episode:04d}/synced_raw/cam2_synced.mp4").read_bytes(), b"newly synchronized")
        self.assertEqual(list(output.glob(".pipeline-sync-*")), [])
        self.assertEqual(raw_before, {p: p.read_bytes() for p in raw_before})

    def test_failed_fresh_sync_confidence_preserves_diagnostics_without_promotion(self):
        settings = pipeline.load_settings(self.config)
        output = self.data / "output"
        output.mkdir()
        with patch.object(pipeline, "verify_inputs") as checked, self.forbid_commands(), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "confidence"):
                pipeline.synchronize_missing(settings, ["episode_0002", "episode_0004"],
                                             Mock(run=self.staged_sync_runner(sync_passed=False)))
        checked.assert_not_called()
        self.assertFalse((output / "episode_0002").exists())
        self.assertFalse((output / "episode_0004").exists())
        self.assertEqual(len(list(output.glob(".pipeline-sync-*"))), 1)

    def test_second_staged_episode_probe_failure_prevents_any_promotion(self):
        settings = pipeline.load_settings(self.config)
        output = self.data / "output"
        output.mkdir()
        with patch.object(pipeline, "verify_inputs", side_effect=[None, ValueError("bad presentation timestamps")]), \
             self.forbid_commands(), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "bad presentation"):
                pipeline.synchronize_missing(settings, ["episode_0002", "episode_0004"],
                                             Mock(run=self.staged_sync_runner()))
        self.assertFalse((output / "episode_0002").exists())
        self.assertFalse((output / "episode_0004").exists())

    def test_sync_promotion_does_not_replace_a_newly_created_nonempty_target(self):
        settings = pipeline.load_settings(self.config)
        output = self.data / "output"
        output.mkdir()
        def create_competing_target(_settings, name, _stage):
            target = output / name
            target.mkdir()
            (target / "valuable.txt").write_text("preserve me")
        with patch.object(pipeline, "verify_inputs", side_effect=create_competing_target), \
             self.forbid_commands(), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "Refusing to replace"):
                pipeline.synchronize_missing(settings, ["episode_0002"], Mock(run=self.staged_sync_runner()))
        self.assertEqual((output / "episode_0002/valuable.txt").read_text(), "preserve me")
        self.assertFalse((output / "episode_0002/synced_raw").exists())

    def test_runner_uses_argument_list_logs_child_output_and_raises_on_failure(self):
        log = self.root / "run log.txt"
        child = Mock()
        child.stdout = iter(["stage started\n", "specific diagnostic\n"])
        child.wait.return_value = 7
        manager = Mock()
        manager.__enter__ = Mock(return_value=child)
        manager.__exit__ = Mock(return_value=False)
        with patch.object(pipeline.subprocess, "Popen", return_value=manager) as popen, \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "exit 7") as raised:
                pipeline.Runner(log).run("run_calibration.py", "--base", self.data)
        self.assertIn(str(log), str(raised.exception))
        command = popen.call_args.args[0]
        self.assertIsInstance(command, list)
        self.assertEqual(command[-1], str(self.data))
        self.assertEqual(command[1], str(pipeline.ROOT / "run_calibration.py"))
        self.assertNotIn("shell", popen.call_args.kwargs)
        self.assertIn("specific diagnostic", log.read_text())

    def test_verify_inputs_rejects_mixed_fps_or_frame_size(self):
        settings = pipeline.load_settings(self.config)
        valid = {"fps": "120000/1001", "width": 3840, "height": 2160}
        for invalid, message in (({**valid, "fps": "120"}, "frame rates"),
                                 ({**valid, "width": 1920}, "frame sizes")):
            with self.subTest(invalid=invalid), \
                 patch("export_calibrated_videos.probe_video", side_effect=[valid, invalid]):
                with self.assertRaisesRegex(ValueError, message):
                    pipeline.verify_inputs(settings, "episode_0004")

    def test_calibration_reuse_requires_matching_board_scale_and_source_offsets(self):
        self.populate_existing()
        settings = pipeline.load_settings(self.config)
        directory = self.data / "output/episode_0002/calibration"
        path = directory / "calibration_all_cameras.json"
        original = json.loads(path.read_text())
        with patch("calibration_validation.validate_calibration"):
            self.assertTrue(pipeline.calibration_matches(settings, directory))
            for field, value in (("board_squares", [8, 12]), ("square_size_m", .031)):
                bad = json.loads(json.dumps(original))
                bad["checkerboard"][field] = value
                path.write_text(json.dumps(bad))
                self.assertFalse(pipeline.calibration_matches(settings, directory))
            for field, value in (("source_episode", "episode_0001"),
                                 ("source_frame_offsets", {"cam1": 0, "cam2": 0})):
                bad = {**original, field: value}
                path.write_text(json.dumps(bad))
                self.assertFalse(pipeline.calibration_matches(settings, directory))

    def test_removing_offsets_table_rejects_prior_nonzero_calibration(self):
        self.populate_existing()
        self.config.write_text(SETTINGS.split("[frame_offsets.")[0])
        settings = pipeline.load_settings(self.config)
        directory = self.data / "output/episode_0002/calibration"
        with patch("calibration_validation.validate_calibration"):
            self.assertFalse(pipeline.calibration_matches(settings, directory))

    def test_saved_sync_failure_blocks_reuse_but_legacy_absence_is_allowed(self):
        self.populate_existing()
        settings = pipeline.load_settings(self.config)
        path = self.data / "output/episode_0004/metadata/episode_0004_metadata.json"
        path.parent.mkdir()
        probe = {"fps": "120000/1001", "width": 3840, "height": 2160}
        with patch("export_calibrated_videos.probe_video", return_value=probe) as reader:
            for saved in ({"passed": False}, {"passed": 1}, {"passed": "true"}, []):
                with self.subTest(saved=saved):
                    path.write_text(json.dumps({"sync_validation": saved}))
                    with self.assertRaisesRegex(ValueError, "synchronization validation failed"):
                        pipeline.verify_inputs(settings, "episode_0004")
            reader.assert_not_called()
            path.write_text(json.dumps({"legacy_metadata": True}))
            pipeline.verify_inputs(settings, "episode_0004")
            self.assertEqual(reader.call_count, 2)

    def test_changed_calibration_source_video_forces_refit_on_resume(self):
        self.populate_existing()
        with patch.object(pipeline, "check_environment"), \
             patch.object(pipeline, "verify_inputs"), \
             patch.object(pipeline, "calibration_matches", return_value=True), \
             patch.object(pipeline.Runner, "run", autospec=True) as run, self.forbid_commands():
            first_code, first_text = self.invoke()
            self.assertEqual(first_code, 0, first_text)
            self.assertNotIn("run_calibration.py", [call.args[1] for call in run.call_args_list])
            run.reset_mock()
            unchanged_code, unchanged_text = self.invoke()
            self.assertEqual(unchanged_code, 0, unchanged_text)
            self.assertNotIn("run_calibration.py", [call.args[1] for call in run.call_args_list])
            changed = self.data / "output/episode_0002/synced_raw/cam2_synced.mp4"
            changed.write_bytes(changed.read_bytes() + b"different capture")
            run.reset_mock()
            changed_code, changed_text = self.invoke()
        self.assertEqual(changed_code, 0, changed_text)
        self.assertEqual(run.call_args_list[0].args[1], "run_calibration.py")
        state = json.loads((self.data / "output/pipeline_state.json").read_text())
        self.assertEqual(state["episode_0002"]["inputs"][changed.name]["size"], changed.stat().st_size)

    def test_input_change_during_fit_prevents_publication_or_state_update(self):
        self.populate_existing()
        with patch.object(pipeline, "check_environment"), \
             patch.object(pipeline, "verify_inputs"), \
             patch.object(pipeline, "calibration_matches", return_value=True), \
             patch.object(pipeline, "input_signatures", side_effect=[{"cam1": "before"}, {"cam1": "after"}]), \
             patch.object(pipeline.Runner, "run", autospec=True) as run, self.forbid_commands():
            code, text = self.invoke("--recalibrate")
        self.assertEqual(code, 1)
        self.assertIn("changed during this run", text)
        self.assertEqual([call.args[1] for call in run.call_args_list], ["run_calibration.py"])
        self.assertFalse((self.data / "output/pipeline_state.json").exists())


if __name__ == "__main__":
    unittest.main()
