"""Geometry and file-safety regression tests for the extrinsics migration."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from calibration_geometry import world_to_camera
from migrate_calibration import migrate_calibration
from viz_calibration import build_traces, camera_data_from_calibration


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def snapshot(directory):
    return {str(path.relative_to(directory)): path.read_bytes()
            for path in directory.rglob("*") if path.is_file()}


def legacy_fixture(directory):
    directory.mkdir(parents=True)
    combined = {"source_episode": "episode_fixture", "cameras": {}}
    for cam in range(1, 11):
        name = f"cam{cam}"
        K = [[900 + cam, 0., 640.], [0., 902 + cam, 360.], [0., 0., 1.]]
        intr = {"K": K, "dist": [-.2, .05, 0., 0., -.001],
                "image_size": [1280, 720], "rms_error_px": .3}
        write_json(directory / f"{name}_intrinsics.json", intr)
        combined["cameras"][name] = deepcopy(intr)
        if cam == 3:
            continue
        R = cv2.Rodrigues(np.array([.01 * cam, .02 * (cam-3), -.005 * cam]))[0]
        C = np.array([.08 * (cam-3), .01 * cam, .002 * cam])
        ext = {"reference": "cam3", "target": name,
               "R": R.tolist(), "T": C.tolist(),
               "F": np.eye(3).tolist(),  # Deliberately wrong; migration must rebuild it.
               "baseline_m": float(np.linalg.norm(C)), "method": "direct",
               "stereo_rms_px": .4, "path": [3, cam],
               "euler_deg": {"rx": 100., "ry": 100., "rz": 100.}}
        write_json(directory / f"{name}_extrinsics.json", ext)
        combined["cameras"][name]["extrinsics"] = {
            key: value for key, value in ext.items() if key not in ("target", "F", "euler_deg")}
    write_json(directory / "calibration_all_cameras.json", combined)
    (directory / "validation").mkdir()
    (directory / "validation" / "image.bin").write_bytes(b"unchanged validation asset\0")
    return combined


class CalibrationMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / "calibration"
        self.legacy = legacy_fixture(self.directory)

    def test_geometry_frusta_intrinsics_and_backup_preserved(self):
        original = snapshot(self.directory)
        old_data, old_ref = camera_data_from_calibration(self.legacy)
        old_traces, old_centers = build_traces(old_data, old_ref, .05)
        result = migrate_calibration(self.directory)
        self.assertTrue(result["changed"])
        self.assertEqual(snapshot(Path(result["backup_dir"])), original)
        migrated = json.loads((self.directory / "calibration_all_cameras.json").read_text())
        self.assertEqual(migrated["schema_version"], 2)
        self.assertEqual(migrated["extrinsics_convention"], "world_to_camera")
        self.assertEqual(migrated["reference_camera"], "cam3")
        new_data, new_ref = camera_data_from_calibration(migrated)
        new_traces, new_centers = build_traces(new_data, new_ref, .05)
        for (old_id, old_center), (new_id, new_center) in zip(old_centers, new_centers):
            self.assertEqual(old_id, new_id)
            np.testing.assert_allclose(old_center, new_center, atol=1e-12)
        for old_trace, new_trace in zip(old_traces, new_traces):
            for axis in ("x", "y", "z"):
                a = np.array(getattr(old_trace, axis), dtype=float)
                b = np.array(getattr(new_trace, axis), dtype=float)
                np.testing.assert_allclose(a, b, atol=1e-12, equal_nan=True)

        points = np.array([[-.3, -.2, 2.], [.1, .3, 3.], [.4, -.1, 4.], [-.1, .1, 1.]])
        Kref = np.array(migrated["cameras"]["cam3"]["K"])
        p1 = cv2.projectPoints(points, np.zeros(3), np.zeros(3), Kref, np.zeros(5))[0].reshape(-1, 2)
        for name, entry in migrated["cameras"].items():
            with self.subTest(camera=name):
                filename = f"{name}_intrinsics.json"
                self.assertEqual((self.directory / filename).read_bytes(), original[filename])
                ext = json.loads((self.directory / f"{name}_extrinsics.json").read_text())
                self.assertEqual(entry["extrinsics"], ext)
                self.assertEqual(ext["convention"], "world_to_camera")
                R, t = world_to_camera(ext, expected_reference="cam3")
                self.assertEqual(ext["euler_rotation_order"], "Rz @ Ry @ Rx")
                if name == "cam3":
                    np.testing.assert_array_equal(R, np.eye(3))
                    np.testing.assert_array_equal(t, np.zeros((3, 1)))
                    self.assertIsNone(ext["F"])
                    continue
                old = self.legacy["cameras"][name]["extrinsics"]
                Rold, Cold = np.array(old["R"]), np.array(old["T"])
                before = (Rold.T @ (points-Cold).T).T
                after = (R @ points.T + t).T
                np.testing.assert_allclose(before, after, atol=1e-12)
                p2 = cv2.projectPoints(points, cv2.Rodrigues(R)[0], t,
                                       np.array(entry["K"]), np.zeros(5))[0].reshape(-1, 2)
                lines = cv2.computeCorrespondEpilines(p1.reshape(-1, 1, 2), 1,
                                                      np.array(ext["F"])).reshape(-1, 3)
                distance = np.abs(np.sum(lines[:, :2]*p2, axis=1) + lines[:, 2])
                self.assertLess(float(distance.max()), 1e-8)
                rx, ry, rz = np.radians([ext["euler_deg"][key] for key in ("rx", "ry", "rz")])
                Rx = cv2.Rodrigues(np.array([rx, 0., 0.]))[0]
                Ry = cv2.Rodrigues(np.array([0., ry, 0.]))[0]
                Rz = cv2.Rodrigues(np.array([0., 0., rz]))[0]
                np.testing.assert_allclose(Rz @ Ry @ Rx, R, atol=3e-8)

    def test_second_migration_is_byte_identical_no_op(self):
        first = migrate_calibration(self.directory)
        before = snapshot(self.directory)
        second = migrate_calibration(self.directory)
        self.assertFalse(second["changed"])
        self.assertIsNone(second["backup_dir"])
        self.assertEqual(snapshot(self.directory), before)
        self.assertEqual(len(list(self.directory.parent.glob("calibration_backup_*"))), 1)
        self.assertTrue(Path(first["backup_dir"]).is_dir())

    def test_invalid_inputs_fail_before_mutation_or_backup(self):
        cases = ("unknown_convention", "different_reference", "different_pose", "global_convention")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp) / "calibration"
                combined = legacy_fixture(directory)
                path = directory / "cam1_extrinsics.json"
                ext = json.loads(path.read_text())
                if case == "unknown_convention":
                    ext["convention"] = "guess"
                elif case == "different_reference":
                    ext["reference"] = "cam2"
                elif case == "different_pose":
                    ext["T"][0] += .1
                else:
                    combined["extrinsics_convention"] = "world_to_camera"
                    write_json(directory / "calibration_all_cameras.json", combined)
                write_json(path, ext)
                before = snapshot(directory)
                with self.assertRaises(ValueError):
                    migrate_calibration(directory)
                self.assertEqual(snapshot(directory), before)
                self.assertEqual(list(directory.parent.glob("calibration_backup_*")), [])

    def test_visualizer_rejects_conflicting_metadata(self):
        for case in ("reference", "convention", "global"):
            with self.subTest(case=case):
                combined = deepcopy(self.legacy)
                ext = combined["cameras"]["cam1"]["extrinsics"]
                if case == "reference":
                    ext["reference"] = "cam2"
                elif case == "convention":
                    ext["convention"] = "bad"
                else:
                    combined["extrinsics_convention"] = "world_to_camera"
                with self.assertRaises(ValueError):
                    camera_data_from_calibration(combined)

    def test_missing_combined_file_is_created(self):
        (self.directory / "calibration_all_cameras.json").unlink()
        migrate_calibration(self.directory)
        combined = json.loads((self.directory / "calibration_all_cameras.json").read_text())
        self.assertEqual(len(combined["cameras"]), 10)
        self.assertEqual(combined["reference_camera"], "cam3")

    def test_replace_failure_rolls_back_completed_replacements(self):
        before = snapshot(self.directory)
        import os
        real_replace = os.replace
        calls = 0
        def failing_replace(source, destination):
            nonlocal calls
            calls += 1
            if calls == 4:
                raise OSError("injected replacement failure")
            return real_replace(source, destination)
        with patch("migrate_calibration.os.replace", side_effect=failing_replace):
            with self.assertRaises(OSError):
                migrate_calibration(self.directory)
        self.assertEqual(snapshot(self.directory), before)
        self.assertEqual(len(list(self.directory.parent.glob("calibration_backup_*"))), 1)


if __name__ == "__main__":
    unittest.main()
