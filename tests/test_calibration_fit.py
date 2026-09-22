"""Regression tests for intrinsic refitting and independent model selection."""

import unittest
from unittest.mock import patch

import cv2
import numpy as np

import calibration_fit
from calibration_geometry import distortion_diagnostics


SIZE = (640, 480)
TRUE_K = np.array([[350., 0., 320.], [0., 352., 240.], [0., 0., 1.]])
TRUE_DIST = np.array([4 / 15, 0., .0002, -.00015, 0., 3 / 5, 0., 0.])
# Old camera 8 rational fit: its near-cancelled denominator pole lies beyond
# training observations, while the finite image-domain screen still passes.
FINITE_POLE_DIST = np.array([
    .6036451704915687, -.5410122354796412, .00010081730098022018,
    .0003164865172080193, -.026273167361149548,
    .9008542944646701, -.48710168200809334, -.136577079817813,
])


def synthetic_views(count=24):
    """Diverse physically projected, fully visible board poses with small noise."""
    rng = np.random.default_rng(584)
    board = np.zeros((8 * 6, 3), dtype=np.float32)
    board[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2) * .04
    board[:, :2] -= np.array([.14, .10], dtype=np.float32)
    objects, images = [], []
    while len(objects) < count:
        rvec = rng.uniform([-.6, -.6, -.5], [.6, .6, .5])
        tvec = rng.uniform([-.38, -.28, .28], [.38, .28, .75])
        points, _ = cv2.projectPoints(board, rvec, tvec, TRUE_K, TRUE_DIST)
        flat = points.reshape(-1, 2)
        if (flat.min(axis=0) >= [3, 3]).all() and (flat.max(axis=0) <= [637, 477]).all():
            objects.append(board.copy())
            images.append((points + rng.normal(0, .015, points.shape)).astype(np.float32))
    return objects, images


class CalibrationFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.objects, cls.images = synthetic_views()
        cls.training_objects = cls.objects[:18]
        cls.training_images = cls.images[:18]
        cls.validation_objects = cls.objects[18:]
        cls.validation_images = cls.images[18:]

    def fit(self, **kwargs):
        return calibration_fit.fit_intrinsics(
            self.training_objects, self.training_images, SIZE,
            kwargs.pop('validation_obj_points', self.validation_objects),
            kwargs.pop('validation_img_points', self.validation_images), **kwargs)

    def fake_calibrate(self, objects, images, size, K, dist, **kwargs):
        # OpenCV rational calibration can return fourteen entries with unused
        # thin-prism/tilt parameters even when only the eight-parameter model ran.
        coefficient_count = 14 if kwargs['flags'] & cv2.CALIB_RATIONAL_MODEL else 5
        return (.02, TRUE_K.copy(), np.zeros((1, coefficient_count)),
                [np.zeros((3, 1)) for _ in objects],
                [np.array([[0.], [0.], [1.]]) for _ in objects])

    def test_refits_real_rational_observations_without_validation_leakage(self):
        original_images = [image.copy() for image in self.images]
        actual_calibrate = cv2.calibrateCamera
        calls = []

        def checked_calibrate(objects, images, size, K, dist, **kwargs):
            self.assertEqual(len(images), len(self.training_images))
            self.assertTrue(all(image.dtype == np.float32 for image in images))
            self.assertTrue(all(obj.dtype == np.float32 for obj in objects))
            for actual, expected in zip(images, self.training_images):
                np.testing.assert_array_equal(actual, expected)
            self.assertFalse(any(np.array_equal(image, withheld)
                                 for image in images for withheld in self.validation_images))
            calls.append(kwargs['flags'])
            return actual_calibrate(objects, images, size, K, dist, **kwargs)

        with patch.object(calibration_fit.cv2, 'calibrateCamera', side_effect=checked_calibrate):
            result = self.fit()
        self.assertEqual(len(calls), 5)
        self.assertTrue(all(flags & cv2.CALIB_USE_LU for flags in calls))
        self.assertTrue(result['diagnostics']['valid'])
        self.assertTrue(distortion_diagnostics(result['K'], result['dist'], SIZE)['valid'])
        self.assertLess(result['selection']['heldout_rms_px'], .1)
        self.assertLess(result['rms'], .1)
        self.assertEqual(len(result['rvecs']), 18)
        self.assertEqual(len(result['tvecs']), 18)
        self.assertEqual(result['selection']['heldout_view_count'], 6)
        self.assertEqual(len(result['selection']['candidates']), 5)
        self.assertEqual(len(result['selection']['heldout_per_view_rms_px']), 6)
        np.testing.assert_allclose(result['K'][:2, :2], TRUE_K[:2, :2], atol=1.)
        for actual, original in zip(self.images, original_images):
            np.testing.assert_array_equal(actual, original)

    def test_requires_sufficient_explicit_heldout_and_training_views(self):
        with self.assertRaisesRegex(ValueError, 'Held-out validation'):
            calibration_fit.fit_intrinsics(self.training_objects, self.training_images, SIZE)
        with self.assertRaisesRegex(ValueError, 'at least 3'):
            self.fit(validation_obj_points=self.validation_objects[:2],
                     validation_img_points=self.validation_images[:2])
        with self.assertRaisesRegex(ValueError, 'at least 5'):
            calibration_fit.fit_intrinsics(
                self.training_objects[:4], self.training_images[:4], SIZE,
                self.validation_objects, self.validation_images)

    def test_rejects_exact_training_validation_overlap(self):
        with self.assertRaisesRegex(ValueError, 'training image'):
            self.fit(validation_obj_points=self.training_objects[:3],
                     validation_img_points=[image.copy() for image in self.training_images[:3]])

    def test_all_geometrically_invalid_candidates_fail_with_model_reasons(self):
        with patch.object(calibration_fit.cv2, 'calibrateCamera', side_effect=self.fake_calibrate), \
             patch.object(calibration_fit, 'distortion_diagnostics',
                          return_value={'valid': False, 'reasons': ['Synthetic radial fold']}), \
             patch.object(calibration_fit, '_heldout_error') as heldout:
            with self.assertRaises(ValueError) as raised:
                self.fit()
        self.assertIn('No valid intrinsic model', str(raised.exception))
        for model in ('opencv5', 'opencv4', 'rational_low', 'rational6', 'rational8'):
            self.assertIn(model, str(raised.exception))
        self.assertIn('Synthetic radial fold', str(raised.exception))
        heldout.assert_not_called()

    def test_nonfinite_fit_is_excluded_and_other_candidates_still_run(self):
        calls = 0

        def calibrate(*args, **kwargs):
            nonlocal calls
            result = self.fake_calibrate(*args, **kwargs)
            calls += 1
            if calls == 1:
                result[1][0, 0] = np.nan
            return result

        with patch.object(calibration_fit.cv2, 'calibrateCamera', side_effect=calibrate), \
             patch.object(calibration_fit, '_heldout_error', return_value=(.1, [.1] * 6)):
            result = self.fit()
        self.assertEqual(calls, 5)
        candidate = result['selection']['candidates'][0]
        self.assertFalse(candidate['valid'])
        self.assertIn('nonfinite', ' '.join(candidate['rejection_reasons']))
        self.assertNotEqual(result['model'], 'opencv5')

    def test_similar_heldout_scores_prefer_simpler_refitted_model(self):
        # Eight coefficients improve by only 0.04 px; use the four-coefficient fit.
        scores = iter((.13, .10, .12, .08, .06))
        with patch.object(calibration_fit.cv2, 'calibrateCamera', side_effect=self.fake_calibrate), \
             patch.object(calibration_fit, '_heldout_error',
                          side_effect=lambda *args: (next(scores), [.1] * 6)):
            result = self.fit()
        self.assertEqual(result['model'], 'opencv4')
        self.assertEqual(len(result['dist']), 4)
        self.assertAlmostEqual(result['selection']['heldout_rms_px'], .10)

    def test_material_heldout_gain_can_select_rational8_and_trims_unused_coefficients(self):
        scores = iter((.3, .2, .18, .15, .03))
        with patch.object(calibration_fit.cv2, 'calibrateCamera', side_effect=self.fake_calibrate), \
             patch.object(calibration_fit, '_heldout_error',
                          side_effect=lambda *args: (next(scores), [.1] * 6)):
            result = self.fit()
        self.assertEqual(result['model'], 'rational8')
        self.assertEqual(len(result['dist']), 8)

    def test_rational6_can_supply_an_intermediate_model_without_cubic_terms(self):
        scores = iter((.4, .3, .25, .10, .08))
        flags_seen = []

        def calibrate(*args, **kwargs):
            flags_seen.append(kwargs['flags'])
            return self.fake_calibrate(*args, **kwargs)

        with patch.object(calibration_fit.cv2, 'calibrateCamera', side_effect=calibrate), \
             patch.object(calibration_fit, '_heldout_error',
                          side_effect=lambda *args: (next(scores), [.1] * 6)):
            result = self.fit()
        self.assertEqual(result['model'], 'rational6')
        self.assertEqual(len(result['dist']), 8)
        candidate = next(c for c in result['selection']['candidates'] if c['model'] == 'rational6')
        self.assertEqual(candidate['free_distortion_coefficients'], 6)
        self.assertTrue(flags_seen[3] & cv2.CALIB_FIX_K3)
        self.assertTrue(flags_seen[3] & cv2.CALIB_FIX_K6)
        self.assertFalse(flags_seen[3] & cv2.CALIB_FIX_K2)
        self.assertFalse(flags_seen[3] & cv2.CALIB_FIX_K5)

    def test_new_fit_rejects_a_positive_pole_even_when_image_geometry_is_valid(self):
        self.assertTrue(distortion_diagnostics(TRUE_K, FINITE_POLE_DIST, SIZE)['valid'])
        poles = calibration_fit.rational_positive_poles(FINITE_POLE_DIST)
        self.assertEqual(len(poles), 1)
        self.assertAlmostEqual(poles[0], 1.3808435896932318, places=10)
        calls = 0

        def calibrate(*args, **kwargs):
            nonlocal calls
            result = self.fake_calibrate(*args, **kwargs)
            calls += 1
            if calls == 5:
                result[2][0, :8] = FINITE_POLE_DIST
            return result

        with patch.object(calibration_fit.cv2, 'calibrateCamera', side_effect=calibrate), \
             patch.object(calibration_fit, '_heldout_error', return_value=(.1, [.1] * 6)) as heldout:
            result = self.fit()
        rejected = result['selection']['candidates'][-1]
        self.assertEqual(rejected['model'], 'rational8')
        self.assertTrue(rejected['diagnostics']['valid'])
        self.assertFalse(rejected['valid'])
        self.assertIn('positive real denominator pole', ' '.join(rejected['rejection_reasons']))
        self.assertIsNone(rejected['heldout_rms_px'])
        self.assertEqual(heldout.call_count, 4)
        self.assertNotEqual(result['model'], 'rational8')

    def test_pole_policy_handles_far_poles_cancellation_and_no_pole_models(self):
        far_pole = np.zeros(8)
        far_pole[5] = -.0001  # Radius 100, far outside the sampled sensor.
        self.assertTrue(distortion_diagnostics(TRUE_K, far_pole, SIZE)['valid'])
        self.assertEqual(calibration_fit.rational_positive_poles(far_pole), [100.])
        cancelled = np.array([-2., 1., 0., 0., 0., -2., 1., 0.])
        self.assertTrue(calibration_fit.rational_positive_poles(cancelled))
        for distortion in (np.zeros(4), np.zeros(5), np.zeros(8), TRUE_DIST,
                           np.array([0., 0., 0., 0., 0., 0., 1., 0.])):
            with self.subTest(distortion=distortion.tolist()):
                self.assertEqual(calibration_fit.rational_positive_poles(distortion), [])

    def test_bad_heldout_fit_rejects_even_finite_geometrically_valid_models(self):
        with patch.object(calibration_fit.cv2, 'calibrateCamera', side_effect=self.fake_calibrate), \
             patch.object(calibration_fit, '_heldout_error', return_value=(3., [3.] * 6)):
            with self.assertRaisesRegex(ValueError, 'Held-out RMS 3.000 px exceeds 2.0 px'):
                self.fit()


if __name__ == '__main__':
    unittest.main()
