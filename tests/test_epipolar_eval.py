"""Synthetic evaluator runs exercise geometry, coverage, and the emitted report."""

import contextlib
import io
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import run_eval_epipolar as evaluator


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.calib = self.base / "output" / "episode_fixture" / "calibration"
        self.videos = self.calib.parent / "synced_raw"
        self.out = self.base / "evaluation"
        self.calib.mkdir(parents=True)
        self.videos.mkdir()
        self.K = np.array([[100., 0, 80], [0, 100., 60], [0, 0, 1.]])
        self.points = np.array([(20. + 5*x, 15. + 5*y)
                                for y in range(11) for x in range(8)], dtype=np.float32)
        self.x_offsets = {}
        self.per_camera_points = {}
        self.no_detection = set()
        self.capture_instances = []
        self.read_events = []
        self.video_frame_counts = {}
        for cam in (1, 2, 3):
            self.write_intrinsics(cam)
            (self.videos / f"cam{cam}_synced.mp4").touch()
        self.write_extrinsics()

    def write_intrinsics(self, cam, dist=None, image_size=(160, 120), K=None):
        data = {"K": (self.K if K is None else K).tolist(),
                "dist": [0.] * 5 if dist is None else dist,
                "image_size": image_size, "rms_error_px": 0.0}
        (self.calib / f"cam{cam}_intrinsics.json").write_text(json.dumps(data))

    def write_extrinsics(self, convention=None):
        for cam, position in ((1, [.1, 0., 0.]), (2, [0., .1, 0.])):
            translation = np.array(position)
            if convention == "world_to_camera":
                translation = -translation
            data = {"reference": "cam3", "target": f"cam{cam}",
                    "R": np.eye(3).tolist(), "T": translation.tolist(),
                    # Deliberately invalid saved F: evaluation must recompute it.
                    "F": np.ones((3, 3)).tolist()}
            if convention is not None:
                data["convention"] = convention
            (self.calib / f"cam{cam}_extrinsics.json").write_text(json.dumps(data))

    def capture(self, path):
        cam = int(re.search(r"cam(\d+)_synced", path).group(1))
        owner = self
        class Capture:
            released = False
            frame = 0

            def isOpened(self):
                return True

            def get(self, prop):
                return owner.video_frame_counts.get(cam, 100)

            def set(self, prop, value):
                self.frame = int(value)
                return True

            def read(self):
                owner.read_events.append((cam, self.frame))
                image = np.full((120, 160, 3), cam, np.uint8)
                image[0, 1, :] = self.frame
                return True, image

            def release(self):
                self.released = True

        cap = Capture()
        self.capture_instances.append(cap)
        return cap

    def detect(self, gray, board_size):
        cam, frame = int(gray[0, 0]), int(gray[0, 1])
        if cam in self.no_detection:
            return None
        points = self.per_camera_points.get(cam, self.points).copy()
        offset = self.x_offsets.get(cam, 0)
        points[:, 0] += offset(frame) if callable(offset) else offset
        return points.reshape(-1, 1, 2)

    def run_evaluation(self, *extra):
        argv = ["--base", str(self.base), "--episode", "episode_fixture",
                "--cams", "3", "--num-frames", "3", "--out", str(self.out), *extra]
        output = io.StringIO()
        with patch.object(evaluator.cv2, "VideoCapture", side_effect=self.capture), \
                patch.object(evaluator, "detect_board", side_effect=self.detect), \
                contextlib.redirect_stdout(output):
            code = evaluator.main(argv)
        path = self.out / "epipolar_eval_summary.json"
        summary = json.loads(path.read_text(), parse_constant=lambda value: self.fail(value))
        self.assertTrue(all(cap.released for cap in self.capture_instances))
        return code, summary, output.getvalue()

    def test_infers_reference_and_reports_all_good_pairs(self):
        code, summary, _ = self.run_evaluation()
        self.assertEqual(code, 0)
        self.assertEqual(summary["reference"], "cam3")
        self.assertEqual(summary["verdict"], "GOOD")
        self.assertEqual(summary["expected_pair_count"], 2)
        self.assertEqual(summary["evaluated_pair_count"], 2)
        self.assertEqual(set(summary["pairs"]), {"3-1", "3-2"})
        self.assertEqual(summary["pairs"]["3-2"]["total_point_pairs"], 88 * 3)
        self.assertFalse((self.calib / "validation").exists())
        self.assertTrue(list(self.out.glob("*.jpg")))

    def test_all_high_error_frames_are_retained_and_prevent_good_verdict(self):
        self.x_offsets[2] = 50
        code, summary, output = self.run_evaluation()
        bad = summary["pairs"]["3-2"]
        self.assertEqual(code, 1)
        self.assertEqual(summary["verdict"], "POOR")
        self.assertEqual(bad["status"], "POOR")
        self.assertEqual(bad["frames_evaluated"], 3)
        self.assertEqual(bad["frames_high_error"], 3)
        self.assertAlmostEqual(bad["mean_px"], 50, places=5)
        self.assertEqual(bad["total_point_pairs"], 88 * 3)
        self.assertIn("included in statistics", output)
        self.assertNotIn("REJECTED", output)

    def test_mixed_good_and_bad_frames_do_not_filter_statistics(self):
        self.x_offsets[2] = lambda frame: 50 if frame == 50 else 0
        code, summary, _ = self.run_evaluation()
        bad = summary["pairs"]["3-2"]
        self.assertEqual(code, 1)
        self.assertEqual(bad["frames_evaluated"], 3)
        self.assertEqual(bad["frames_high_error"], 1)
        self.assertAlmostEqual(bad["mean_px"], 50 / 3, places=5)
        self.assertAlmostEqual(summary["overall_mean_px"], 50 / 6, places=5)

    def test_worst_pair_controls_quality_even_when_overall_mean_is_good(self):
        self.x_offsets[2] = 1.5
        code, summary, _ = self.run_evaluation()
        self.assertEqual(code, 0)
        self.assertLess(summary["overall_mean_px"], 1)
        self.assertEqual(summary["pairs"]["3-2"]["status"], "OK")
        self.assertEqual(summary["verdict"], "OK")

    def test_untested_pair_is_present_and_overall_is_incomplete(self):
        self.no_detection.add(2)
        code, summary, _ = self.run_evaluation()
        pair = summary["pairs"]["3-2"]
        self.assertEqual(code, 2)
        self.assertEqual(summary["verdict"], "INCOMPLETE")
        self.assertEqual(summary["evaluated_pair_count"], 1)
        self.assertEqual(pair["frames_without_detections"], 3)
        self.assertIsNone(pair["mean_px"])

    def test_missing_video_is_reported_without_key_error(self):
        (self.videos / "cam2_synced.mp4").unlink()
        code, summary, _ = self.run_evaluation()
        self.assertEqual(code, 2)
        self.assertEqual(summary["verdict"], "INCOMPLETE")
        self.assertIn("Missing video", " ".join(summary["pairs"]["3-2"]["reasons"]))

    def test_missing_intrinsics_and_extrinsics_each_preserve_expected_pair(self):
        for filename, reason in (("cam2_intrinsics.json", "Missing intrinsics"),
                                 ("cam2_extrinsics.json", "Missing extrinsics")):
            with self.subTest(filename=filename):
                path = self.calib / filename
                original = path.read_text()
                path.unlink()
                try:
                    code, summary, _ = self.run_evaluation()
                    self.assertEqual(code, 2)
                    self.assertEqual(summary["verdict"], "INCOMPLETE")
                    self.assertIn(reason, " ".join(summary["pairs"]["3-2"]["reasons"]))
                finally:
                    path.write_text(original)

    def test_explicit_wrong_reference_fails_before_opening_videos(self):
        code, summary, output = self.run_evaluation("--ref-cam", "1")
        self.assertEqual(code, 2)
        self.assertEqual(summary["verdict"], "FAILED")
        self.assertIn("does not match saved reference cam3", output)
        self.assertEqual(len(self.capture_instances), 0)

    def test_reference_camera_transform_must_be_identity(self):
        path = self.calib / "cam3_extrinsics.json"
        for rotation, translation in ((np.eye(3), [1., 0., 0.]),
                                      (cv2.Rodrigues(np.array([0., .1, 0.]))[0], [0., 0., 0.])):
            with self.subTest(translation=translation):
                path.write_text(json.dumps({
                    "reference": "cam3", "target": "cam3", "convention": "world_to_camera",
                    "R": rotation.tolist(), "T": translation, "F": None,
                }))
                code, summary, output = self.run_evaluation()
                self.assertEqual(code, 2)
                self.assertEqual(summary["verdict"], "FAILED")
                self.assertIn("identity rotation and zero translation", output)
                self.assertEqual(self.read_events, [])

    def test_extrinsics_target_must_match_filename(self):
        path = self.calib / "cam1_extrinsics.json"
        pose = json.loads(path.read_text())
        pose["target"] = "cam2"
        path.write_text(json.dumps(pose))
        code, summary, output = self.run_evaluation()
        self.assertEqual(code, 2)
        self.assertEqual(summary["pairs"]["3-1"]["status"], "FAILED")
        self.assertIn("target does not match", output)

    def test_explicit_frame_indices_override_num_frames_and_only_those_frames_are_read(self):
        code, summary, _ = self.run_evaluation(
            "--num-frames", "1", "--frame-indices", "0", "17", "99")
        self.assertEqual(code, 0)
        self.assertEqual(summary["sampling_method"], "explicit")
        self.assertEqual(summary["requested_frame_indices"], [0, 17, 99])
        self.assertEqual(summary["sample_frame_indices"], [0, 17, 99])
        self.assertEqual(summary["requested_sample_count"], 3)
        self.assertEqual(summary["num_sample_frames"], 3)
        self.assertEqual({frame for _, frame in self.read_events}, {0, 17, 99})
        for camera in (1, 2, 3):
            self.assertEqual({frame for cam, frame in self.read_events if cam == camera},
                             {0, 17, 99})
        for record in summary["pairs"].values():
            self.assertEqual(record["frames_requested"], 3)
            self.assertEqual(record["frames_sampled"], 3)

    def test_explicit_indices_reject_negative_duplicate_and_unsorted_requests(self):
        for indices in ((-1, 0), (0, 0), (17, 0)):
            with self.subTest(indices=indices):
                code, summary, output = self.run_evaluation(
                    "--frame-indices", *(str(i) for i in indices))
                self.assertEqual(code, 2)
                self.assertEqual(summary["verdict"], "FAILED")
                self.assertIn("nonnegative, unique, and ascending", output)
                self.assertEqual(self.read_events, [])

    def test_out_of_range_frame_indices_fail_without_reading_any_frames(self):
        code, summary, output = self.run_evaluation("--frame-indices", "0", "100")
        self.assertEqual(code, 2)
        self.assertEqual(summary["verdict"], "FAILED")
        self.assertEqual(summary["sample_frame_indices"], [0, 100])
        self.assertEqual(summary["requested_sample_count"], 2)
        self.assertIn("outside cam3 video range 0..99", output)
        self.assertEqual(self.read_events, [])

    def test_out_of_range_indices_on_shorter_target_video_also_fail(self):
        self.video_frame_counts[2] = 80
        code, summary, output = self.run_evaluation("--frame-indices", "0", "90")
        self.assertEqual(code, 2)
        self.assertEqual(summary["pairs"]["3-2"]["status"], "FAILED")
        self.assertIn("outside cam2 video range 0..79", output)
        self.assertEqual(self.read_events, [])

    def test_calibration_directory_override_does_not_modify_or_activate_it(self):
        candidate = self.base / "candidate_calibration"
        self.calib.rename(candidate)
        before = {p.name: p.read_bytes() for p in candidate.iterdir()}
        code, summary, _ = self.run_evaluation("--calibration-dir", str(candidate))
        self.assertEqual(code, 0)
        self.assertEqual(summary["calibration_dir"], str(candidate))
        self.assertFalse(self.calib.exists())
        self.assertEqual(before, {p.name: p.read_bytes() for p in candidate.iterdir()})

    def test_offsets_apply_to_reference_and_target_source_indices(self):
        offsets = self.base / "offsets.json"
        offsets.write_text(json.dumps({"cam2": -1, "cam3": 2}))
        code, summary, _ = self.run_evaluation(
            "--frame-offsets", str(offsets), "--frame-indices", "1", "17", "90")
        self.assertEqual(code, 0)
        self.assertEqual(summary["sample_frame_indices"], [1, 17, 90])
        self.assertEqual(summary["frame_offsets"], {"cam1": 0, "cam2": -1, "cam3": 2})
        self.assertEqual(summary["frame_offsets_source"]["kind"], "file")
        expected = {1: [1, 17, 90], 2: [0, 16, 89], 3: [3, 19, 92]}
        for camera, indices in expected.items():
            self.assertEqual(summary["source_frame_indices"][f"cam{camera}"], indices)
            self.assertEqual({frame for cam, frame in self.read_events if cam == camera}, set(indices))
        self.assertEqual(summary["pairs"]["3-2"]["source_frame_indices"],
                         {"cam3": [3, 19, 92], "cam2": [0, 16, 89]})

    def test_negative_offset_out_of_range_fails_before_reading(self):
        offsets = self.base / "offsets.json"
        offsets.write_text(json.dumps({"cam2": -1}))
        code, summary, output = self.run_evaluation(
            "--frame-offsets", str(offsets), "--frame-indices", "0", "17")
        self.assertEqual(code, 2)
        self.assertEqual(summary["pairs"]["3-2"]["status"], "FAILED")
        self.assertEqual(summary["source_frame_indices"]["cam2"], [-1, 16])
        self.assertIn("after offset -1", output)
        self.assertEqual(self.read_events, [])

    def test_positive_offset_out_of_range_also_fails(self):
        offsets = self.base / "offsets.json"
        offsets.write_text(json.dumps({"cam2": 1}))
        code, summary, output = self.run_evaluation(
            "--frame-offsets", str(offsets), "--frame-indices", "99")
        self.assertEqual(code, 2)
        self.assertEqual(summary["source_frame_indices"]["cam2"], [100])
        self.assertIn("outside cam2 video range 0..99", output)
        self.assertEqual(self.read_events, [])

    def test_matching_source_episode_infers_saved_offsets(self):
        (self.calib / "calibration_all_cameras.json").write_text(json.dumps({
            "source_episode": "episode_fixture", "source_frame_offsets": {"cam2": -1},
        }))
        code, summary, _ = self.run_evaluation("--frame-indices", "1", "17")
        self.assertEqual(code, 0)
        self.assertEqual(summary["frame_offsets_source"]["kind"], "calibration")
        self.assertEqual(summary["source_frame_indices"]["cam2"], [0, 16])
        self.assertEqual({frame for cam, frame in self.read_events if cam == 2}, {0, 16})

    def test_different_source_episode_does_not_apply_saved_offsets(self):
        (self.calib / "calibration_all_cameras.json").write_text(json.dumps({
            "source_episode": "different_episode", "source_frame_offsets": {"cam2": -1},
        }))
        code, summary, _ = self.run_evaluation("--frame-indices", "0", "17")
        self.assertEqual(code, 0)
        self.assertEqual(summary["frame_offsets_source"]["kind"], "zero_default")
        self.assertEqual(summary["frame_offsets"], {"cam1": 0, "cam2": 0, "cam3": 0})
        self.assertEqual({frame for cam, frame in self.read_events if cam == 2}, {0, 17})

    def test_explicit_offset_file_overrides_calibration_metadata(self):
        (self.calib / "calibration_all_cameras.json").write_text(json.dumps({
            "source_episode": "episode_fixture", "source_frame_offsets": {"cam2": -1},
        }))
        offsets = self.base / "offsets.json"
        offsets.write_text(json.dumps({"cam2": 1}))
        code, summary, _ = self.run_evaluation(
            "--frame-offsets", str(offsets), "--frame-indices", "1", "17")
        self.assertEqual(code, 0)
        self.assertEqual(summary["frame_offsets_source"]["kind"], "file")
        self.assertEqual({frame for cam, frame in self.read_events if cam == 2}, {2, 18})

    def test_automatic_sampling_respects_common_source_range_with_offsets(self):
        offsets = self.base / "offsets.json"
        offsets.write_text(json.dumps({"cam2": -25}))
        code, summary, _ = self.run_evaluation("--frame-offsets", str(offsets))
        self.assertEqual(code, 0)
        self.assertEqual(summary["common_logical_frame_range"], [25, 99])
        self.assertTrue(all(0 <= frame < 100 for _, frame in self.read_events))
        for logical, source in zip(summary["sample_frame_indices"],
                                   summary["source_frame_indices"]["cam2"]):
            self.assertEqual(source, logical - 25)

    def test_world_to_camera_and_legacy_give_identical_geometry(self):
        self.x_offsets[2] = 3
        _, legacy, _ = self.run_evaluation()
        old = evaluator.load_calib(str(self.calib), 3, 3)
        self.write_extrinsics("world_to_camera")
        _, standard, _ = self.run_evaluation()
        new = evaluator.load_calib(str(self.calib), 3, 3)
        self.assertEqual(legacy["pairs"], standard["pairs"])
        for cam in (1, 2):
            np.testing.assert_allclose(old[cam]["R"], new[cam]["R"])
            np.testing.assert_allclose(old[cam]["t"], new[cam]["t"])

    def test_rotated_camera_projection_matches_both_pose_formats(self):
        R, _ = cv2.Rodrigues(np.array([.03, .15, .02]))
        t = np.array([-.1, .02, .01])
        reference_pixels = np.column_stack((self.points, np.ones(len(self.points))))
        world = (np.linalg.inv(self.K) @ reference_pixels.T).T * 3
        camera = (R @ world.T).T + t
        projected = (self.K @ camera.T).T
        self.per_camera_points[1] = (projected[:, :2] / projected[:, 2:]).astype(np.float32)
        path = self.calib / "cam1_extrinsics.json"
        pose = json.loads(path.read_text())
        pose.update(convention="world_to_camera", R=R.tolist(), T=t.tolist())
        path.write_text(json.dumps(pose))
        code, standard, _ = self.run_evaluation()
        self.assertEqual(code, 0)
        self.assertLess(standard["pairs"]["3-1"]["max_px"], 1e-4)
        pose.pop("convention")
        pose.update(R=R.T.tolist(), T=(-R.T @ t).tolist())
        path.write_text(json.dumps(pose))
        code, legacy, _ = self.run_evaluation()
        self.assertEqual(code, 0)
        self.assertLess(legacy["pairs"]["3-1"]["max_px"], 1e-4)
        self.assertAlmostEqual(legacy["pairs"]["3-1"]["mean_px"],
                               standard["pairs"]["3-1"]["mean_px"], places=6)

    def test_invalid_lens_stops_with_report_and_diagnostic_override_still_fails(self):
        self.write_intrinsics(2, dist=[-1., 0., 0., 0., 0.])
        code, summary, _ = self.run_evaluation()
        self.assertEqual(code, 2)
        self.assertEqual(summary["verdict"], "FAILED")
        self.assertEqual(summary["invalid_cameras"], ["cam2"])
        self.assertEqual(summary["pairs"]["3-2"]["status"], "FAILED")
        self.assertEqual(len(self.capture_instances), 0)
        code, summary, _ = self.run_evaluation("--allow-invalid-intrinsics")
        self.assertEqual(code, 2)
        self.assertTrue(summary["diagnostic_mode"])
        self.assertEqual(summary["verdict"], "FAILED")
        self.assertEqual(summary["pairs"]["3-1"]["status"], "GOOD")

    def test_malformed_intrinsics_fail_instead_of_looking_incomplete(self):
        K = self.K.copy()
        K[0, 0] = -100
        self.write_intrinsics(2, K=K)
        code, summary, _ = self.run_evaluation()
        self.assertEqual(code, 2)
        self.assertEqual(summary["verdict"], "FAILED")
        self.assertEqual(summary["invalid_cameras"], ["cam2"])

    def test_invalid_points_are_counted_excluded_and_prevent_good(self):
        original = evaluator.undistort_points_checked

        def invalidate_one(points, K, dist):
            undistorted, mask = original(points, K, dist)
            undistorted[0] = np.nan
            mask[0] = False
            return undistorted, mask

        with patch.object(evaluator, "undistort_points_checked", side_effect=invalidate_one):
            code, summary, _ = self.run_evaluation()
        self.assertEqual(code, 2)
        self.assertEqual(summary["verdict"], "FAILED")
        for pair in summary["pairs"].values():
            self.assertEqual(pair["invalid_point_pairs"], 3)
            self.assertEqual(pair["total_point_pairs"], 87 * 3)
            self.assertEqual(pair["status"], "FAILED")
            self.assertTrue(np.isfinite(pair["mean_px"]))

    def test_wrong_video_resolution_cannot_produce_good_result(self):
        self.write_intrinsics(2, image_size=(320, 240))
        code, summary, _ = self.run_evaluation()
        self.assertEqual(code, 2)
        self.assertEqual(summary["verdict"], "FAILED")
        self.assertIn("dimensions", " ".join(summary["pairs"]["3-2"]["reasons"]))

    def test_all_pairs_without_detections_are_incomplete(self):
        self.no_detection.add(3)
        code, summary, _ = self.run_evaluation()
        self.assertEqual(code, 2)
        self.assertEqual(summary["verdict"], "INCOMPLETE")
        self.assertEqual(summary["evaluated_pair_count"], 0)
        self.assertEqual(len(summary["pairs"]), 2)

    def test_vertical_epipolar_line_is_drawn_vertically(self):
        image = np.zeros((20, 30, 3), np.uint8)
        evaluator.draw_epipolar_line(image, [1, 0, -10], (0, 255, 0))
        self.assertTrue(np.all(image[:, 10, 1] == 255))
        self.assertTrue(np.all(image[:, 20, :] == 0))


if __name__ == "__main__":
    unittest.main()
