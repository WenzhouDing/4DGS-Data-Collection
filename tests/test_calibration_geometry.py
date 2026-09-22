"""Physical regression checks for calibration geometry, independent of downloaded data.

Run with: .venv/bin/python -m unittest discover -s tests -v
"""
import itertools
import unittest

import cv2
import numpy as np

from calibration_geometry import (
    EXTRINSICS_CONVENTION,
    distortion_diagnostics,
    fundamental_from_KRT,
    reprojection_rms,
    undistort_points_checked,
    world_to_camera,
)


IMAGE_SIZE = (3840, 2160)
K = np.array([[1800.0, 0.0, 1920.0], [0.0, 1800.0, 1080.0], [0.0, 0.0, 1.0]])
# Stable rational approximation of wide-angle barrel distortion:
# scale(r) = (1 + 4*r**2/15) / (1 + 3*r**2/5).
# Its radial derivative has positive numerator 1 + r**2/5 + 4*r**4/25.
RATIONAL_WIDE = np.array([4 / 15, 0, 0, 0, 0, 3 / 5, 0, 0], dtype=float)
# Actual old coefficients, copied here so regressions do not require episode data.
FOLDING_CAM1_K = np.array(
    [[1793.8673604741603, 0, 1910.8783900565552],
     [0, 1792.0738602133385, 1084.229109177105], [0, 0, 1]], dtype=float
)
FOLDING_CAM1_DIST = np.array(
    [-0.2960158831266753, 0.18824789425705452,
     0.00020973552170438866, -1.3282518079932076e-06,
     -0.09164318020704276]
)
FOLDING_CAM7_K = np.array(
    [[1790.0049936687788, 0, 1929.0383230161854],
     [0, 1789.59051746492, 1074.1446368127451], [0, 0, 1]], dtype=float
)
FOLDING_CAM7_DIST = np.array(
    [-0.30312232276164747, 0.2162631434804194,
     0.00011475675101609118, -0.00032629609204606983,
     -0.12970325420431292]
)


def project_pixels(points, camera_matrix, distortion=None, rotation=None, translation=None):
    """Independent reference projection through OpenCV's forward camera model."""
    if rotation is None:
        rotation = np.eye(3)
    if translation is None:
        translation = np.zeros(3)
    pixels, _ = cv2.projectPoints(
        np.asarray(points, dtype=float).reshape(-1, 3),
        cv2.Rodrigues(np.asarray(rotation, dtype=float))[0],
        np.asarray(translation, dtype=float).reshape(3),
        camera_matrix,
        np.zeros(5) if distortion is None else distortion,
    )
    return pixels.reshape(-1, 2)


def camera_rays(pixels, camera_matrix):
    homogeneous = np.column_stack((np.asarray(pixels).reshape(-1, 2),
                                   np.ones(len(pixels))))
    return (np.linalg.inv(camera_matrix) @ homogeneous.T).T


class IntrinsicsGeometryTests(unittest.TestCase):
    def assert_model_valid(self, camera_matrix, distortion, image_size=IMAGE_SIZE):
        result = distortion_diagnostics(camera_matrix, distortion, image_size)
        self.assertTrue(result['valid'], result.get('reasons'))
        self.assertIsInstance(result['reasons'], list)

    def assert_model_rejected(self, camera_matrix, distortion, image_size=IMAGE_SIZE):
        # Invalid input matrices may be rejected by ValueError before diagnostics;
        # a well-formed but physically invalid model should normally return reasons.
        try:
            result = distortion_diagnostics(camera_matrix, distortion, image_size)
        except (ValueError, np.linalg.LinAlgError):
            return
        self.assertFalse(result['valid'], result)
        self.assertTrue(result['reasons'])

    def test_identity_pincushion_and_stable_wide_angle_models_are_valid(self):
        for distortion in (
            np.zeros(5),
            np.array([0.07, 0.005, 0, 0, 0.0001]),
            RATIONAL_WIDE,
        ):
            with self.subTest(distortion=distortion.tolist()):
                self.assert_model_valid(K, distortion)

    def test_old_low_rms_models_are_rejected_for_full_field_foldover(self):
        for matrix, distortion in (
            (FOLDING_CAM1_K, FOLDING_CAM1_DIST),
            (FOLDING_CAM7_K, FOLDING_CAM7_DIST),
        ):
            with self.subTest(distortion=distortion.tolist()):
                result = distortion_diagnostics(matrix, distortion, IMAGE_SIZE)
                self.assertFalse(result['valid'], result)
                self.assertTrue(result['reasons'])
                # Confirm the fixture causes actual map foldover on the original-K
                # output canvas using finite differences of OpenCV's forward map.
                p = np.array([[3836., 2156.], [3837., 2156.], [3836., 2157.]])
                mapped = project_pixels(camera_rays(p, matrix), matrix, distortion)
                determinant = np.linalg.det(np.column_stack(
                    (mapped[1] - mapped[0], mapped[2] - mapped[0])))
                self.assertLess(determinant, 0.0)

    def test_rational_pole_on_output_canvas_is_rejected(self):
        # Denominator = 1-r**2, with a pole inside the original-K canvas.
        distortion = np.array([0., 0., 0., 0., 0., -1., 0., 0.])
        self.assert_model_rejected(K, distortion)

    def test_locally_valid_output_map_does_not_prove_full_sensor_coverage(self):
        # Old camera 2 stays monotonic on the default output canvas, but its
        # central branch cannot reach the complete raw sensor. A check of only
        # the default remap Jacobian would incorrectly approve this model.
        matrix = np.array([[1785.384387607997, 0, 1927.9177433383043],
                           [0, 1786.8529342442243, 1078.8918070752825],
                           [0, 0, 1]], dtype=float)
        distortion = np.array([-0.24876693439126701, 0.08293062255540255,
                               -0.0003666164722123743, -0.0006610118288498608,
                               -0.014493578245794147])
        x, y = np.meshgrid(np.linspace(1, 3837, 31), np.linspace(1, 2157, 19))
        points = np.column_stack((x.ravel(), y.ravel()))
        forward = project_pixels(camera_rays(points, matrix), matrix, distortion)
        dx = project_pixels(camera_rays(points + [1., 0.], matrix), matrix, distortion) - forward
        dy = project_pixels(camera_rays(points + [0., 1.], matrix), matrix, distortion) - forward
        self.assertTrue(np.all(dx[:, 0] * dy[:, 1] - dx[:, 1] * dy[:, 0] > 0))
        self.assert_model_rejected(matrix, distortion)

    def test_coincident_numerator_denominator_zero_is_not_a_safe_model(self):
        # Apparent identity away from r=1 conceals an undefined 0/0 ring.
        distortion = np.array([-2., 1., 0., 0., 0., -2., 1., 0.])
        self.assert_model_rejected(K, distortion)

    def test_singular_or_nonfinite_intrinsics_are_rejected(self):
        for invalid in (
            np.zeros((3, 3)),
            np.array([[0, 0, 1920], [0, 1800, 1080], [0, 0, 1]], dtype=float),
            np.array([[np.nan, 0, 1920], [0, 1800, 1080], [0, 0, 1]], dtype=float),
        ):
            with self.subTest(matrix=invalid.tolist()):
                self.assert_model_rejected(invalid, np.zeros(5))

    def test_identity_inverse_preserves_center_edges_and_corners(self):
        points = np.array([[0., 0.], [3839., 0.], [3839., 2159.],
                           [0., 2159.], [1920., 1080.], [100., 1700.]])
        recovered, valid = undistort_points_checked(points, K, np.zeros(5))
        self.assertTrue(np.asarray(valid).all())
        np.testing.assert_allclose(np.asarray(recovered).reshape(-1, 2), points,
                                   atol=1e-7)

    def test_inverse_matches_independent_forward_projection(self):
        # Rays cover the usable field, including strong wide-angle distortion.
        rays = np.array([[0., 0., 1.], [.2, -.1, 1.], [-.8, .4, 1.],
                         [1.3, .4, 1.], [-1.7, -.6, 1.]])
        for distortion in (np.zeros(5), np.array([.05, .003, .0002, -.0001, 0]),
                           RATIONAL_WIDE):
            with self.subTest(distortion=distortion.tolist()):
                distorted = project_pixels(rays, K, distortion)
                expected = project_pixels(rays, K)
                recovered, valid = undistort_points_checked(distorted, K, distortion)
                recovered = np.asarray(recovered).reshape(-1, 2)
                self.assertTrue(np.asarray(valid).all(), (distorted, valid))
                np.testing.assert_allclose(recovered, expected, atol=1e-3)
                redistorted = project_pixels(camera_rays(recovered, K), K, distortion)
                np.testing.assert_allclose(redistorted, distorted, atol=1e-3)

    def test_unrecoverable_folded_periphery_is_marked_invalid(self):
        points = np.array([[FOLDING_CAM7_K[0, 2], FOLDING_CAM7_K[1, 2]],
                           [3839., 2159.], [0., 0.]])
        _, valid = undistort_points_checked(points, FOLDING_CAM7_K, FOLDING_CAM7_DIST)
        valid = np.asarray(valid).reshape(-1)
        self.assertEqual(len(valid), 3)
        self.assertTrue(valid[0], 'The optical center must remain usable.')
        self.assertFalse(valid[1], 'A folded far corner must not be accepted.')
        self.assertFalse(valid[2], 'A folded far corner must not be accepted.')


class ExtrinsicsGeometryTests(unittest.TestCase):
    @staticmethod
    def fixture(rotation_vector, center, convention=EXTRINSICS_CONVENTION,
                reference='cam8', target='cam2'):
        rotation = cv2.Rodrigues(np.array(rotation_vector, dtype=float))[0]
        center = np.array(center, dtype=float)
        translation = -rotation @ center
        if convention in ('camera_to_world', None):
            stored_rotation, stored_translation = rotation.T, center
        else:
            stored_rotation, stored_translation = rotation, translation
        ext = {'reference': reference, 'target': target,
               'R': stored_rotation.tolist(), 'T': stored_translation.tolist()}
        if convention is not None:
            ext['convention'] = convention
        return ext, rotation, translation

    def test_exported_convention_is_world_to_camera(self):
        self.assertEqual(EXTRINSICS_CONVENTION, 'world_to_camera')

    def test_explicit_world_to_camera_and_legacy_camera_to_world_agree(self):
        world_point = np.array([.2, -.1, 3.])
        for convention in (EXTRINSICS_CONVENTION, 'camera_to_world', None):
            with self.subTest(convention=convention):
                ext, expected_rotation, expected_translation = self.fixture(
                    [.04, -.25, .02], [.7, -.2, .1], convention)
                rotation, translation = world_to_camera(ext, expected_reference='cam8')
                np.testing.assert_allclose(rotation, expected_rotation, atol=1e-12)
                np.testing.assert_allclose(np.asarray(translation).reshape(3),
                                           expected_translation, atol=1e-12)
                np.testing.assert_allclose(
                    rotation @ world_point + np.asarray(translation).reshape(3),
                    expected_rotation @ world_point + expected_translation, atol=1e-12)

    def test_wrong_reference_and_unknown_convention_raise(self):
        ext, _, _ = self.fixture([0., .1, 0.], [.5, 0., 0.])
        with self.assertRaises(ValueError):
            world_to_camera(ext, expected_reference='cam3')
        ext['convention'] = 'ambiguous_transform'
        with self.assertRaises(ValueError):
            world_to_camera(ext)

    def test_fundamental_geometry_for_both_directions_and_nonreference_pair(self):
        # Reference ID 8 exceeds target ID 2: sorting IDs must not reverse geometry.
        fixtures = {
            8: self.fixture([0., 0., 0.], [0., 0., 0.], target='cam8'),
            2: self.fixture([.025, -.16, .01], [.55, .08, .02], target='cam2'),
            11: self.fixture([-.035, .13, -.02], [-.45, -.1, .05], target='cam11'),
        }
        intrinsics = {
            8: K,
            2: np.array([[1780., 0, 1910.], [0, 1790., 1070.], [0, 0, 1.]]),
            11: np.array([[1820., 0, 1930.], [0, 1810., 1090.], [0, 0, 1.]]),
        }
        points_world = np.array([[x, y, z] for z in (2., 3.5, 5.)
                                 for y in (-.4, .15, .6) for x in (-.7, .2, .8)])
        recovered = {cam: world_to_camera(f[0], expected_reference='cam8')
                     for cam, f in fixtures.items()}
        for first, second in itertools.permutations(fixtures, 2):
            with self.subTest(pair=(first, second)):
                r1, t1 = recovered[first]
                r2, t2 = recovered[second]
                t1, t2 = np.asarray(t1).reshape(3), np.asarray(t2).reshape(3)
                relative_r = r2 @ r1.T
                relative_t = t2 - relative_r @ t1
                fundamental = fundamental_from_KRT(
                    intrinsics[first], intrinsics[second], relative_r, relative_t)
                self.assertTrue(np.isfinite(fundamental).all())
                self.assertEqual(np.linalg.matrix_rank(fundamental), 2)
                # Projections use independently known fixture poses, not recovered ones.
                p1 = project_pixels(points_world, intrinsics[first],
                                    rotation=fixtures[first][1], translation=fixtures[first][2])
                p2 = project_pixels(points_world, intrinsics[second],
                                    rotation=fixtures[second][1], translation=fixtures[second][2])
                h1 = np.column_stack((p1, np.ones(len(p1))))
                h2 = np.column_stack((p2, np.ones(len(p2))))
                lines = (fundamental @ h1.T).T
                distances = abs(np.sum(h2 * lines, axis=1)) / np.linalg.norm(lines[:, :2], axis=1)
                self.assertLess(float(distances.max()), 1e-7)
                reversed_f = fundamental_from_KRT(
                    intrinsics[second], intrinsics[first],
                    relative_r.T, -relative_r.T @ relative_t)
                f_unit = fundamental.T / np.linalg.norm(fundamental)
                reverse_unit = reversed_f / np.linalg.norm(reversed_f)
                if np.sum(f_unit * reverse_unit) < 0:
                    reverse_unit = -reverse_unit
                np.testing.assert_allclose(reverse_unit, f_unit, atol=1e-10)

    def test_fundamental_rejects_singular_intrinsics(self):
        with self.assertRaises((ValueError, np.linalg.LinAlgError)):
            fundamental_from_KRT(np.zeros((3, 3)), K, np.eye(3), np.array([1., 0., 0.]))


class ReprojectionErrorTests(unittest.TestCase):
    def test_one_pixel_offset_is_one_pixel_rms_for_any_corner_count(self):
        for count in (1, 4, 88):
            with self.subTest(count=count):
                observed = np.arange(count * 2, dtype=float).reshape(-1, 1, 2)
                projected = observed + np.array([1., 0.])
                self.assertAlmostEqual(reprojection_rms(observed, projected), 1., places=12)

    def test_rms_is_root_mean_square_of_euclidean_pixel_distance(self):
        observed = np.array([[0., 0.], [3., 4.]])
        projected = np.array([[1., 0.], [3., 7.]])
        self.assertAlmostEqual(reprojection_rms(observed, projected), np.sqrt(5), places=12)
        self.assertEqual(reprojection_rms(observed, observed), 0.)


if __name__ == '__main__':
    unittest.main()
