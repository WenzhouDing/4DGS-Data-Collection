"""Reuse/copy safety and orchestrator integration without processing videos."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from calibration_geometry import fundamental_from_KRT, distortion_diagnostics
from calibration_validation import copy_calibration, validate_calibration


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def snapshot(path):
    return {str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()}


def valid_fixture(directory):
    directory.mkdir(parents=True, exist_ok=True)
    K = np.array([[900., 0., 640.], [0., 900., 360.], [0., 0., 1.]])
    combined = {"schema_version": 2, "extrinsics_convention": "world_to_camera",
                "reference_camera": "cam1", "cameras": {}}
    for cam in (1, 2):
        name = f"cam{cam}"
        intr = {"K": K.tolist(), "dist": [0.] * 5, "image_size": [1280, 720]}
        t = np.array([-.2 * (cam-1), 0., 0.])
        ext = {"reference": "cam1", "target": name, "convention": "world_to_camera",
               "R": np.eye(3).tolist(), "T": t.tolist(),
               "F": None if cam == 1 else fundamental_from_KRT(K, K, np.eye(3), t).tolist()}
        write_json(directory / f"{name}_intrinsics.json", intr)
        write_json(directory / f"{name}_extrinsics.json", ext)
        combined["cameras"][name] = {**intr, "extrinsics": ext}
    write_json(directory / "calibration_all_cameras.json", combined)
    write_json(directory / "observation_split.json", {
        "training": [f"frame_{i:06d}.jpg" for i in range(5)],
        "model_selection": [f"frame_{i:06d}.jpg" for i in range(5, 8)],
        "test": [f"frame_{i:06d}.jpg" for i in range(8, 11)]})
    write_json(directory / "heldout_epipolar.json", {
        "passed": True, "reference": "cam1", "pairs": {
            "1-2": {"status": "PASS", "invalid_points": 0, "mean_px": .2, "p95_px": .5,
                    "frames": [{"source_frame": i, "mean_px": .2, "max_px": .7} for i in range(8, 11)]}}})
    (directory / "validation").mkdir(exist_ok=True)
    (directory / "validation" / "asset.bin").write_bytes(b"validation-image-asset")


class CalibrationReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        valid_fixture(self.source)

    def test_valid_copy_preserves_source_and_backs_up_entire_destination(self):
        destination = self.root / "destination"
        valid_fixture(destination)
        (destination / "previous.bin").write_bytes(b"previous calibration")
        before = snapshot(destination)
        source_before = snapshot(self.source)
        backup = copy_calibration(self.source, destination, 2, 1, "episode_0001")
        self.assertEqual(snapshot(Path(backup)), before)
        self.assertEqual(snapshot(self.source), source_before)
        self.assertEqual(validate_calibration(destination, 2, 1)["camera_count"], 2)
        self.assertFalse((destination / "previous.bin").exists())
        provenance = json.loads((destination / "calibration_source.json").read_text())
        self.assertEqual(provenance["source_episode"], "episode_0001")

    def test_invalid_source_cases_do_not_mutate_destination(self):
        cases = ("legacy", "lens", "wrong_f", "missing_pair", "failed_test", "high_error", "leaked_test")
        original = snapshot(self.source)
        destination = self.root / "destination"
        valid_fixture(destination)
        before = snapshot(destination)
        for case in cases:
            with self.subTest(case=case):
                for name, data in original.items():
                    (self.source / name).write_bytes(data)
                if case == "legacy":
                    path = self.source / "calibration_all_cameras.json"
                    data = json.loads(path.read_text()); data.pop("schema_version")
                elif case in {"lens", "wrong_f"}:
                    combined_path = self.source / "calibration_all_cameras.json"
                    combined = json.loads(combined_path.read_text())
                    if case == "lens":
                        path = self.source / "cam2_intrinsics.json"
                        data = json.loads(path.read_text()); data["dist"] = [-1., 0., 0., 0., 0.]
                        combined["cameras"]["cam2"]["dist"] = data["dist"]
                    else:
                        path = self.source / "cam2_extrinsics.json"
                        data = json.loads(path.read_text()); data["F"] = np.eye(3).tolist()
                        combined["cameras"]["cam2"]["extrinsics"] = data
                    write_json(combined_path, combined)
                elif case == "leaked_test":
                    path = self.source / "observation_split.json"
                    data = json.loads(path.read_text()); data["test"][0] = data["training"][0]
                else:
                    path = self.source / "heldout_epipolar.json"
                    data = json.loads(path.read_text())
                    if case == "missing_pair": data["pairs"] = {}
                    elif case == "failed_test": data["passed"] = False
                    else: data["pairs"]["1-2"]["p95_px"] = 6.
                write_json(path, data)
                with self.assertRaises(ValueError):
                    copy_calibration(self.source, destination, 2, 1)
                self.assertEqual(snapshot(destination), before)
                self.assertEqual(list(self.root.glob("destination.backup-*")), [])

    def test_missing_test_evidence_and_wrong_reference_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_calibration(self.source, 2, 2)
        (self.source / "heldout_epipolar.json").unlink()
        with self.assertRaises(OSError):
            validate_calibration(self.source, 2, 1)

    def test_rational_pole_is_rejected_even_outside_the_sampled_sensor_domain(self):
        path = self.source / "cam2_intrinsics.json"
        intr = json.loads(path.read_text())
        intr["dist"] = [0., 0., 0., 0., 0., -.01, 0., 0.]
        self.assertTrue(distortion_diagnostics(intr["K"], intr["dist"], intr["image_size"])["valid"])
        write_json(path, intr)
        combined_path = self.source / "calibration_all_cameras.json"
        combined = json.loads(combined_path.read_text())
        combined["cameras"]["cam2"]["dist"] = intr["dist"]
        write_json(combined_path, combined)
        with self.assertRaisesRegex(ValueError, "positive-radius pole"):
            validate_calibration(self.source, 2, 1)

    def test_offsets_require_integer_camera_mapping_and_matching_provenance(self):
        combined_path = self.source / "calibration_all_cameras.json"
        split_path = self.source / "observation_split.json"
        combined = json.loads(combined_path.read_text())
        split = json.loads(split_path.read_text())
        combined["source_episode"] = "episode_0001"
        combined["source_frame_offsets"] = {"cam1": 0, "cam2": -1}
        write_json(combined_path, combined)
        with self.assertRaisesRegex(ValueError, "different source-frame offsets"):
            validate_calibration(self.source, 2, 1)
        split["source_frame_offsets"] = combined["source_frame_offsets"]
        write_json(split_path, split)
        validate_calibration(self.source, 2, 1)
        target = self.root / "copy"
        copy_calibration(self.source, target, 2, 1, "episode_0001")
        copied = json.loads((target / "calibration_all_cameras.json").read_text())
        self.assertEqual(copied["source_frame_offsets"], combined["source_frame_offsets"])
        self.assertEqual(copied["source_episode"], "episode_0001")
        for bad in ({"cam2": True}, {"cam2": 1.0}, {"cam3": 0}):
            combined["source_frame_offsets"] = bad
            write_json(combined_path, combined)
            with self.assertRaises(ValueError):
                validate_calibration(self.source, 2, 1)


class OrganizeEpisodesIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source = self.base / "output/episode_0001/calibration"
        self.target = self.base / "output/episode_0002/calibration"
        valid_fixture(self.source)
        valid_fixture(self.target)
        (self.target / "old.bin").write_bytes(b"old destination")
        self.template = self.base / "template"
        valid_fixture(self.template)
        self.log = self.base / "commands.jsonl"
        bindir = self.base / "bin"; bindir.mkdir()
        uv = bindir / "uv"
        uv.write_text(f"#!{sys.executable}\n" + '''import json, os, pathlib, shutil, subprocess, sys, tempfile
args = sys.argv[1:]
with open(os.environ['TEST_UV_LOG'], 'a') as stream: stream.write(json.dumps(args)+'\\n')
assert args[:2] == ['run', 'python']
command = args[2]
if command == 'run_calibration.py':
    if os.environ.get('TEST_FIT_FAIL') == '1': sys.exit(7)
    root = pathlib.Path(args[args.index('--base')+1])
    episode = args[args.index('--episode')+1]
    target = root/'output'/episode/'calibration'
    staging = pathlib.Path(tempfile.mkdtemp(dir=target.parent))
    shutil.copytree(os.environ['TEST_VALID_TEMPLATE'], staging, dirs_exist_ok=True)
    sys.path.insert(0, os.getcwd())
    from run_calibration import publish_calibration
    publish_calibration(staging, target)
    sys.exit(0)
if command == 'viz_calibration.py': sys.exit(0)
if command == 'export_calibrated_videos.py':
    if os.environ.get('TEST_EXPORT_FAIL') == '1': sys.exit(9)
    sys.exit(0)
if command == 'make_undistorted_preview.py':
    if os.environ.get('TEST_PREVIEW_FAIL') == '1': sys.exit(10)
    sys.exit(0)
sys.exit(subprocess.call([sys.executable] + args[2:]))
''')
        uv.chmod(0o755)
        self.env = {**os.environ, "PATH": str(bindir)+os.pathsep+os.environ["PATH"],
                    "TEST_UV_LOG": str(self.log), "TEST_VALID_TEMPLATE": str(self.template)}

    def run_pipeline(self, skip_calib=False, skip_export=False, groups=None):
        command = ["bash", str(ROOT / "organize_episodes.sh"), "--base", str(self.base),
                   "--cams", "2", "--board", "9x12", "--square-size", ".03",
                   "--ref-cam", "1", "--groups", *(groups or ["1,2"]), "--skip-sync"]
        if skip_calib: command.append("--skip-calib")
        if skip_export: command.append("--skip-export")
        result = subprocess.run(command, cwd=ROOT, env=self.env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        calls = [json.loads(line) for line in self.log.read_text().splitlines()]
        return result, calls

    def invalidate_source(self):
        path = self.source / "heldout_epipolar.json"
        report = json.loads(path.read_text()); report["passed"] = False
        write_json(path, report)

    def test_valid_existing_source_is_reused_and_target_is_backed_up(self):
        before = snapshot(self.target)
        result, calls = self.run_pipeline()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertFalse(any(c[2] == "run_calibration.py" for c in calls))
        backups = list(self.target.parent.glob("calibration.backup-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(snapshot(backups[0]), before)

    def test_default_exports_each_selected_episode_once_after_all_copies(self):
        (self.base / "output/episode_0003").mkdir()
        result, calls = self.run_pipeline(groups=["1,2", "1,2"])
        self.assertEqual(result.returncode, 0, result.stdout)
        exports = [call for call in calls if call[2] == "export_calibrated_videos.py"]
        self.assertEqual(len(exports), 2)
        self.assertEqual([call[call.index("--episode") + 1] for call in exports],
                         ["episode_0001", "episode_0002"])
        for call in exports:
            self.assertEqual(call[call.index("--base") + 1], str(self.base))
            self.assertEqual(call[call.index("--cams") + 1], "2")
            self.assertNotIn("--ref-cam", call)
        first_export = next(i for i, call in enumerate(calls)
                            if call[2] == "export_calibrated_videos.py")
        self.assertTrue(all(i < first_export for i, call in enumerate(calls) if "--copy-to" in call))
        previews = [call for call in calls if call[2] == "make_undistorted_preview.py"]
        self.assertEqual([call[call.index("--episode") + 1] for call in previews],
                         ["episode_0001", "episode_0002"])
        for preview, export in zip(previews, exports):
            self.assertEqual(calls.index(preview), calls.index(export) + 1)

    def test_skip_export_preserves_calibration_only_workflow(self):
        result, calls = self.run_pipeline(skip_export=True)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertFalse(any(call[2] == "export_calibrated_videos.py" for call in calls))
        self.assertFalse(any(call[2] == "make_undistorted_preview.py" for call in calls))
        self.assertTrue(any("--copy-to" in call for call in calls))
        self.assertTrue(any(call[2] == "viz_calibration.py" for call in calls))
        validate_calibration(self.target, 2, 1)

    def test_export_failure_stops_pipeline_with_nonzero_exit(self):
        self.env["TEST_EXPORT_FAIL"] = "1"
        result, calls = self.run_pipeline()
        self.assertEqual(result.returncode, 9, result.stdout)
        self.assertEqual(sum(call[2] == "export_calibrated_videos.py" for call in calls), 1)
        self.assertFalse(any(call[2] == "make_undistorted_preview.py" for call in calls))
        self.assertFalse(any(call[2] == "viz_calibration.py" for call in calls))

    def test_preview_failure_stops_pipeline_with_nonzero_exit(self):
        self.env["TEST_PREVIEW_FAIL"] = "1"
        result, calls = self.run_pipeline()
        self.assertEqual(result.returncode, 10, result.stdout)
        self.assertEqual(sum(call[2] == "make_undistorted_preview.py" for call in calls), 1)
        self.assertFalse(any(call[2] == "viz_calibration.py" for call in calls))

    def test_invalid_source_refits_before_copying(self):
        self.invalidate_source()
        result, calls = self.run_pipeline()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(sum(c[2] == "run_calibration.py" for c in calls), 1)
        validate_calibration(self.target, 2, 1)

    def test_skip_calib_rejects_invalid_source_before_copying(self):
        self.invalidate_source()
        before = snapshot(self.target)
        result, calls = self.run_pipeline(skip_calib=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(c[2] == "run_calibration.py" or "--copy-to" in c for c in calls))
        self.assertEqual(snapshot(self.target), before)

    def test_failed_refit_stops_before_copying_and_preserves_target(self):
        self.invalidate_source()
        self.env["TEST_FIT_FAIL"] = "1"
        before = snapshot(self.target)
        result, calls = self.run_pipeline()
        self.assertEqual(result.returncode, 7, result.stdout)
        self.assertFalse(any("--copy-to" in c for c in calls))
        self.assertEqual(snapshot(self.target), before)


if __name__ == "__main__":
    unittest.main()
