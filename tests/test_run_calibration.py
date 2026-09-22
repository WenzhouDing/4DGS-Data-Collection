"""Pipeline regressions for directional geometry, withheld frames, and publishing."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from calibration_geometry import fundamental_from_KRT
from calibration_observations import load_corner_cache, save_corner_cache
import calibration_fit
import run_calibration as pipeline


SIZE = (640, 480)
K = np.array([[550., 0., 320.], [0., 552., 240.], [0., 0., 1.]])
RIG_DISTORTION = np.array([4 / 15, 0., .0002, -.00015, 0., 3 / 5, 0., 0.])


def normalized_matrix(matrix):
    matrix = np.asarray(matrix, dtype=float)
    return matrix / np.linalg.norm(matrix)


def synthetic_rig():
    board = np.zeros((8 * 6, 3), dtype=np.float32)
    board[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2) * .04
    rotations = {1: cv2.Rodrigues(np.array([.012, .055, -.01]))[0],
                 2: np.eye(3),
                 3: cv2.Rodrigues(np.array([-.018, -.065, .015]))[0]}
    centers = {1: np.array([-.15, .015, .005]), 2: np.zeros(3),
               3: np.array([.18, -.025, .01])}
    translations = {cam: -rotations[cam] @ centers[cam] for cam in centers}
    observations = {cam: {} for cam in centers}
    world_points = {}
    rng = np.random.default_rng(832)
    for index in range(24):
        board_r = cv2.Rodrigues(rng.uniform([-.55, -.5, -.3], [.55, .5, .3]))[0]
        board_t = rng.uniform([-.2, -.15, .9], [0., .03, 1.4])
        points = (board_r @ board.T + board_t[:, None]).T
        filename = f'frame_{30 * (index + 1):06d}.jpg'
        world_points[filename] = points
        for cam in centers:
            pixels, _ = cv2.projectPoints(points, cv2.Rodrigues(rotations[cam])[0],
                                         translations[cam], K, RIG_DISTORTION)
            observations[cam][filename] = (pixels.astype(np.float32), None)
    return observations, rotations, translations, centers, world_points


class PipelineGeometryTests(unittest.TestCase):
    def test_validation_video_reads_apply_positive_and_negative_source_offsets(self):
        for offset, source_frame in ((-1, 29), (2, 32)):
            with self.subTest(offset=offset):
                capture = Mock()
                expected = np.full((12, 16, 3), source_frame, dtype=np.uint8)
                capture.read.return_value = (True, expected)
                with patch.object(pipeline.cv2, 'VideoCapture', return_value=capture):
                    actual = pipeline.read_validation_image(None, '/videos', 2,
                                                            'frame_000030.jpg', offset)
                capture.set.assert_called_once_with(cv2.CAP_PROP_POS_FRAMES, source_frame)
                capture.release.assert_called_once()
                np.testing.assert_array_equal(actual, expected)

    def test_validation_images_already_extracted_do_not_apply_offset_twice(self):
        expected = np.zeros((12, 16, 3), dtype=np.uint8)
        with patch.object(pipeline.cv2, 'imread', return_value=expected), \
             patch.object(pipeline.cv2, 'VideoCapture') as capture:
            actual = pipeline.read_validation_image('/extracted.jpg', '/videos', 2,
                                                    'frame_000030.jpg', -1)
        self.assertIs(actual, expected)
        capture.assert_not_called()

    def test_lookup_pair_reverses_every_directional_quantity_without_mutation(self):
        rotation = cv2.Rodrigues(np.array([.07, -.21, .035]))[0]
        translation = np.array([[.5], [-.03], [.08]])
        fundamental = fundamental_from_KRT(K, K, rotation, translation)
        original = {'R': rotation.copy(), 'T': translation.copy(),
                    'F': fundamental.copy(), 'rms': .2, 'shared': ['frame_000030.jpg']}
        pairs = {(1, 2): original}
        forward_r, forward_t, forward_info = pipeline.lookup_pair(pairs, 1, 2)
        reverse_r, reverse_t, reverse_info = pipeline.lookup_pair(pairs, 2, 1)
        np.testing.assert_allclose(forward_r, rotation)
        np.testing.assert_allclose(forward_t, translation)
        np.testing.assert_allclose(reverse_r @ rotation, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(reverse_r @ translation + reverse_t, np.zeros((3, 1)), atol=1e-12)
        np.testing.assert_allclose(reverse_info['F'], fundamental.T)
        np.testing.assert_allclose(reverse_info['R'], reverse_r)
        np.testing.assert_allclose(reverse_info['T'], reverse_t)
        np.testing.assert_array_equal(original['F'], fundamental)
        self.assertEqual(reverse_info['shared'], original['shared'])
        self.assertIs(forward_info, original)
        self.assertIsNone(pipeline.lookup_pair(pairs, 1, 3))

    def test_heldout_keeps_high_errors_and_expected_missing_pairs(self):
        intrinsics = {cam: {'K': K, 'dist': np.zeros(5)} for cam in (1, 2, 3)}
        fundamental = fundamental_from_KRT(K, K, np.eye(3), np.array([.2, 0., 0.]))
        extrinsics = {1: {'F': fundamental.tolist()}}
        observations = {cam: {} for cam in (1, 2, 3)}
        for index in range(3):
            filename = f'frame_{index * 30:06d}.jpg'
            points = np.array([[100., 100.], [200., 150.], [300., 250.]], dtype=np.float32)
            observations[2][filename] = (points.reshape(-1, 1, 2), None)
            observations[1][filename] = ((points + [10., 40. if index == 2 else 0.]).reshape(-1, 1, 2), None)
        result = pipeline.evaluate_heldout(intrinsics, extrinsics, observations, reference=2)
        self.assertEqual(set(result['pairs']), {'2-1', '2-3'})
        self.assertFalse(result['passed'])
        tested, missing = result['pairs']['2-1'], result['pairs']['2-3']
        self.assertEqual(tested['status'], 'FAIL')
        self.assertEqual(len(tested['frames']), 3)
        self.assertAlmostEqual(tested['mean_px'], 40 / 3, places=7)
        self.assertAlmostEqual(tested['max_px'], 40., places=7)
        self.assertEqual(missing['status'], 'INCOMPLETE')
        self.assertEqual(missing['reason'], 'No extrinsics')

    def test_heldout_reports_no_shared_frames_instead_of_dropping_pair(self):
        intrinsics = {cam: {'K': K, 'dist': np.zeros(5)} for cam in (1, 2)}
        ext = {1: {'F': fundamental_from_KRT(K, K, np.eye(3), [.2, 0., 0.]).tolist()}}
        result = pipeline.evaluate_heldout(intrinsics, ext, {1: {}, 2: {}}, 2)
        self.assertFalse(result['passed'])
        self.assertEqual(result['pairs']['2-1']['status'], 'INCOMPLETE')
        self.assertEqual(result['pairs']['2-1']['shared_frames'], 0)


class CalibrationPublishTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.target = self.root / 'calibration'
        self.staging = self.root / 'staging'
        self.target.mkdir()
        self.staging.mkdir()
        (self.target / 'calibration_all_cameras.json').write_text('{"old": true}')
        (self.target / 'keep.bin').write_bytes(b'old bytes')
        (self.staging / 'calibration_all_cameras.json').write_text('{"new": true}')

    def test_success_backs_up_all_previous_files(self):
        backup = Path(pipeline.publish_calibration(self.staging, self.target))
        self.assertFalse(self.staging.exists())
        self.assertEqual((backup / 'keep.bin').read_bytes(), b'old bytes')
        self.assertEqual(json.loads((backup / 'calibration_all_cameras.json').read_text()), {'old': True})
        self.assertEqual(json.loads((self.target / 'calibration_all_cameras.json').read_text()), {'new': True})

    def test_failed_replacement_rolls_back_the_original_directory(self):
        real_rename = Path.rename

        def rename(path, destination):
            if path == self.staging:
                raise OSError('simulated publication failure')
            return real_rename(path, destination)

        with patch.object(Path, 'rename', autospec=True, side_effect=rename):
            with self.assertRaisesRegex(OSError, 'simulated publication failure'):
                pipeline.publish_calibration(self.staging, self.target)
        self.assertEqual((self.target / 'keep.bin').read_bytes(), b'old bytes')
        self.assertEqual(json.loads((self.target / 'calibration_all_cameras.json').read_text()), {'old': True})
        self.assertTrue(self.staging.exists())
        self.assertEqual(list(self.root.glob('calibration.backup-*')), [])


    def test_refuses_non_calibration_directory_before_any_mutation(self):
        (self.target / 'calibration_all_cameras.json').unlink()
        with self.assertRaisesRegex(ValueError, 'non-calibration directory'):
            pipeline.publish_calibration(self.staging, self.target)
        self.assertEqual((self.target / 'keep.bin').read_bytes(), b'old bytes')
        self.assertTrue(self.staging.exists())
        self.assertEqual(list(self.root.glob('calibration.backup-*')), [])


class UncachedOffsetDetectionTests(unittest.TestCase):
    def test_lockstep_detection_reads_each_logical_frame_plus_its_camera_offset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            synced = root / 'output/synthetic/synced_raw'
            synced.mkdir(parents=True)
            for cam in (1, 2, 3):
                (synced / f'cam{cam}_synced.mp4').write_bytes(b'source signature placeholder')
            offsets = {1: 0, 2: 2, 3: -1}
            offset_file = root / 'offsets.json'
            offset_file.write_text(json.dumps({f'cam{cam}': value for cam, value in offsets.items()}))
            cache = root / 'detected.npz'

            class Capture:
                def __init__(self, cam): self.cam, self.next_frame, self.current = cam, 0, -1
                def get(self, prop):
                    return 30 if prop == cv2.CAP_PROP_FPS else 60 if prop == cv2.CAP_PROP_FRAME_COUNT else 0
                def set(self, prop, value): self.next_frame = int(value); return True
                def grab(self):
                    self.current = self.next_frame; self.next_frame += 1
                    return self.current < 60
                def retrieve(self):
                    image = np.zeros((24, 32, 3), dtype=np.uint8)
                    image[0, 0, :] = self.current
                    image[0, 1, :] = self.cam
                    return True, image
                def release(self): pass

            def detect(gray, board_size):
                corners = np.zeros((int(np.prod(board_size)), 1, 2), dtype=np.float32)
                corners[:, 0, 0] = gray[0, 0]
                corners[:, 0, 1] = gray[0, 1]
                return corners

            argv = ['run_calibration.py', '--base', str(root), '--episode', 'synthetic',
                    '--cams', '3', '--ref-cam', '1', '--board', '9x7', '--square-size', '.04',
                    '--every', '3', '--max-frames', '0', '--corners-cache', str(cache),
                    '--frame-offsets', str(offset_file)]
            with patch('sys.argv', argv), \
                 patch.object(pipeline.cv2, 'VideoCapture', side_effect=lambda path: Capture(int(Path(path).name[3]))), \
                 patch.object(pipeline, '_detect_frame', side_effect=detect), \
                 patch.object(pipeline, 'fit_intrinsics', side_effect=ValueError('intentional fitting stop')), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, 'failed validated intrinsic fitting'):
                    pipeline.main()
            observations, size, info = load_corner_cache(cache, synced, (8, 6), 3, offsets)
            self.assertEqual(size, (32, 24))
            expected_frames = list(range(3, 58, 3))
            for cam in (1, 2, 3):
                self.assertEqual(sorted(observations[cam]), [f'frame_{frame:06d}.jpg' for frame in expected_frames])
                for name, (corners, _) in observations[cam].items():
                    logical_frame = int(name[6:-4])
                    np.testing.assert_array_equal(corners[:, 0, 0], logical_frame + offsets[cam])
                    np.testing.assert_array_equal(corners[:, 0, 1], cam)

class CachedCalibrationIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.episode = self.root / 'output' / 'synthetic'
        self.synced = self.episode / 'synced_raw'
        self.synced.mkdir(parents=True)
        self.output = self.episode / 'calibration'
        self.cache = self.root / 'corners.npz'
        self.observations, self.rotations, self.translations, self.centers, self.world_points = synthetic_rig()
        for cam in self.observations:
            (self.synced / f'cam{cam}_synced.mp4').write_bytes(f'camera {cam} source placeholder'.encode())
        info = {cam: {'source_fps': 30., 'total_frames': 721, 'every_n': 30,
                      'scanned': 24, 'detected': 24,
                      'frame_numbers': [30 * (i + 1) for i in range(24)], 'elapsed_sec': 0.}
                for cam in self.observations}
        save_corner_cache(self.cache, self.observations, SIZE, (8, 6), info, self.synced)
        self.original_cache_bytes = self.cache.read_bytes()
        self.argv = ['run_calibration.py', '--base', str(self.root), '--episode', 'synthetic',
                     '--cams', '3', '--ref-cam', '2', '--board', '9x7', '--square-size', '.04',
                     '--corners-cache', str(self.cache)]

    def main_context(self):
        stack = contextlib.ExitStack()
        stack.enter_context(patch('sys.argv', self.argv))
        stack.enter_context(patch.object(pipeline, 'read_validation_image',
                                        side_effect=lambda *args: np.zeros((480, 640, 3), np.uint8)))
        stack.enter_context(patch.object(pipeline, 'make_bar_chart'))
        stack.enter_context(patch.object(pipeline, 'make_error_bar_chart'))
        # Exercise the rational stereo path even when a polynomial approximation
        # is within the normal 0.05 px preference for a simpler distortion model.
        stack.enter_context(patch.object(calibration_fit, '_SIMPLER_MODEL_TOLERANCE_PX', 0.))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        return stack

    def test_cached_end_to_end_geometry_and_independent_global_frame_split(self):
        fitting_calls, stereo_calls = [], []
        real_fit, real_stereo = pipeline.fit_intrinsics, cv2.stereoCalibrate
        image_to_source = {points.tobytes(): (cam, name)
                           for cam, rows in self.observations.items()
                           for name, (points, _) in rows.items()}

        def identify(images):
            return [image_to_source[np.asarray(image, dtype=np.float32).tobytes()] for image in images]

        def fit(objects, images, size, **kwargs):
            fitting_calls.append((identify(images), identify(kwargs['validation_img_points'])))
            return real_fit(objects, images, size, **kwargs)

        def stereo(objects, first_images, second_images, *args, **kwargs):
            stereo_calls.append((identify(first_images), identify(second_images)))
            if len(np.asarray(args[1]).ravel()) == 8 or len(np.asarray(args[3]).ravel()) == 8:
                self.assertTrue(kwargs['flags'] & cv2.CALIB_RATIONAL_MODEL,
                                'OpenCV requires RATIONAL_MODEL even with FIX_INTRINSIC.')
            return real_stereo(objects, first_images, second_images, *args, **kwargs)

        with self.main_context(), patch.object(pipeline, 'fit_intrinsics', side_effect=fit), \
             patch.object(pipeline.cv2, 'stereoCalibrate', side_effect=stereo):
            pipeline.main()
        self.assertEqual(self.cache.read_bytes(), self.original_cache_bytes)
        split = json.loads((self.output / 'observation_split.json').read_text())
        train, select, test = (set(split[key]) for key in ('training', 'model_selection', 'test'))
        self.assertTrue(train.isdisjoint(select) and train.isdisjoint(test) and select.isdisjoint(test))
        self.assertEqual(train | select | test, set(self.observations[2]))
        self.assertGreaterEqual(len(select), 3)
        self.assertGreaterEqual(len(test), 3)
        self.assertEqual(len(fitting_calls), 3)
        for training_call, selection_call in fitting_calls:
            self.assertEqual({name for cam, name in training_call}, train)
            self.assertEqual({name for cam, name in selection_call}, select)
        self.assertEqual(len(stereo_calls), 3)
        for first, second in stereo_calls:
            self.assertEqual({name for cam, name in first}, train)
            self.assertEqual({name for cam, name in second}, train)
        combined = json.loads((self.output / 'calibration_all_cameras.json').read_text())
        self.assertEqual(combined['schema_version'], 2)
        self.assertEqual(combined['extrinsics_convention'], 'world_to_camera')
        self.assertEqual(combined['reference_camera'], 'cam2')
        self.assertTrue(any(len(camera['dist']) == 8 for camera in combined['cameras'].values()),
                        'The fixture must exercise rational stereo calibration.')
        heldout = json.loads((self.output / 'heldout_epipolar.json').read_text())
        self.assertTrue(heldout['passed'], heldout)
        for pair in heldout['pairs'].values():
            self.assertEqual({f"frame_{frame['source_frame']:06d}.jpg" for frame in pair['frames']}, test)
        for cam in (1, 2, 3):
            intrinsics = json.loads((self.output / f'cam{cam}_intrinsics.json').read_text())
            ext = json.loads((self.output / f'cam{cam}_extrinsics.json').read_text())
            self.assertEqual(combined['cameras'][f'cam{cam}']['extrinsics'], ext)
            self.assertEqual(ext['reference'], 'cam2')
            self.assertEqual(ext['convention'], 'world_to_camera')
            np.testing.assert_allclose(ext['R'], self.rotations[cam], atol=1e-3)
            np.testing.assert_allclose(ext['T'], self.translations[cam], atol=1e-3)
            np.testing.assert_allclose(ext['camera_center_world'], self.centers[cam], atol=1e-3)
            camera_K, distortion = np.array(intrinsics['K']), np.array(intrinsics['dist'])
            for name in sorted(test):
                projected, _ = cv2.projectPoints(self.world_points[name], cv2.Rodrigues(np.array(ext['R']))[0],
                                                 np.array(ext['T']), camera_K, distortion)
                np.testing.assert_allclose(projected, self.observations[cam][name][0], atol=.08)
            if cam == 2:
                np.testing.assert_array_equal(ext['R'], np.eye(3))
                np.testing.assert_array_equal(ext['T'], np.zeros(3))
                self.assertIsNone(ext['F'])
            else:
                reference_K = np.array(combined['cameras']['cam2']['K'])
                expected_f = fundamental_from_KRT(reference_K, camera_K, np.array(ext['R']), np.array(ext['T']))
                np.testing.assert_allclose(normalized_matrix(ext['F']), normalized_matrix(expected_f), atol=1e-10)
                for name in sorted(test):
                    a = cv2.undistortPointsIter(
                        self.observations[2][name][0], K, RIG_DISTORTION, None, K,
                        (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-10)).reshape(-1, 2)
                    b = cv2.undistortPointsIter(
                        self.observations[cam][name][0], K, RIG_DISTORTION, None, K,
                        (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-10)).reshape(-1, 2)
                    homogeneous_a = np.column_stack((a, np.ones(len(a))))
                    homogeneous_b = np.column_stack((b, np.ones(len(b))))
                    lines = homogeneous_a @ np.array(ext['F']).T
                    error = abs(np.sum(homogeneous_b * lines, axis=1)) / np.linalg.norm(lines[:, :2], axis=1)
                    self.assertLess(error.max(), .08)
        self.assertEqual(list(self.episode.glob('calibration.failed-*')), [])

    def create_previous_output(self):
        self.output.mkdir()
        (self.output / 'calibration_all_cameras.json').write_text('{"old": true}')
        (self.output / 'sentinel.bin').write_bytes(b'previous calibration')

    def assert_previous_unchanged(self):
        self.assertEqual(json.loads((self.output / 'calibration_all_cameras.json').read_text()), {'old': True})
        self.assertEqual((self.output / 'sentinel.bin').read_bytes(), b'previous calibration')
        self.assertEqual(set(p.name for p in self.output.iterdir()), {'calibration_all_cameras.json', 'sentinel.bin'})
        self.assertEqual(list(self.episode.glob('calibration.backup-*')), [])
        self.assertEqual(len(list(self.episode.glob('calibration.failed-*'))), 1)
        self.assertEqual(self.cache.read_bytes(), self.original_cache_bytes)

    def test_failed_intrinsic_fit_does_not_publish_or_replace_previous_output(self):
        self.create_previous_output()
        with self.main_context(), patch.object(pipeline, 'fit_intrinsics', side_effect=ValueError('synthetic invalid lens')), \
             patch.object(pipeline, 'publish_calibration') as publish:
            with self.assertRaisesRegex(ValueError, 'failed validated intrinsic fitting'):
                pipeline.main()
        publish.assert_not_called()
        self.assert_previous_unchanged()

    def test_failed_independent_heldout_check_does_not_publish(self):
        self.create_previous_output()
        failed = {'passed': False, 'pairs': {'2-1': {'status': 'FAIL', 'mean_px': 9., 'p95_px': 12.}}}
        with self.main_context(), patch.object(pipeline, 'evaluate_heldout', return_value=failed), \
             patch.object(pipeline, 'publish_calibration') as publish:
            with self.assertRaisesRegex(ValueError, 'Independent held-out geometry failed'):
                pipeline.main()
        publish.assert_not_called()
        self.assert_previous_unchanged()


if __name__ == '__main__':
    unittest.main()
