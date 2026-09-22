"""Calibration conventions and numerical checks shared by writers and readers."""

import cv2
import numpy as np
from numpy.polynomial import polynomial as poly

EXTRINSICS_CONVENTION = "world_to_camera"


def _camera_matrix(K):
    K = np.asarray(K, dtype=np.float64)
    if (K.shape != (3, 3) or not np.isfinite(K).all()
            or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose(K[2], [0, 0, 1])
            or not np.allclose([K[0, 1], K[1, 0]], 0)):
        raise ValueError("Invalid OpenCV camera matrix")
    return K


def _rotation_translation(R, t):
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    if (R.shape != (3, 3) or t.size != 3 or not np.isfinite(R).all()
            or not np.isfinite(t).all()
            or not np.allclose(R.T @ R, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(R), 1, atol=1e-6)):
        raise ValueError("Invalid rigid camera transform")
    return R, t.reshape(3, 1)


def world_to_camera(ext, expected_reference=None):
    """Read new w2c extrinsics or explicitly support unmarked legacy c2w files."""
    if expected_reference is not None:
        expected = str(expected_reference)
        if not expected.startswith("cam"):
            expected = "cam" + expected
        if ext.get("reference") != expected:
            raise ValueError(f"Extrinsics reference {ext.get('reference')!r} "
                             f"does not match {expected}")
    convention = ext.get("convention", "camera_to_world")
    if convention not in (EXTRINSICS_CONVENTION, "camera_to_world"):
        raise ValueError(f"Unknown extrinsics convention: {convention}")
    R, t = _rotation_translation(ext["R"], ext["T"])
    return (R.T, -R.T @ t) if convention == "camera_to_world" else (R, t)


def fundamental_from_KRT(K1, K2, R, t):
    """F for x2.T @ F @ x1 = 0; R,t transform camera-1 into camera-2."""
    K1, K2 = _camera_matrix(K1), _camera_matrix(K2)
    R, t = _rotation_translation(R, t)
    x, y, z = t.ravel()
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.linalg.inv(K2).T @ cross @ R @ np.linalg.inv(K1)


def reprojection_rms(observed, projected):
    observed = np.asarray(observed, dtype=np.float64).reshape(-1, 2)
    projected = np.asarray(projected, dtype=np.float64).reshape(-1, 2)
    if observed.shape != projected.shape or not len(observed):
        raise ValueError("Reprojection RMS needs corresponding nonempty points")
    return float(np.sqrt(np.mean(np.sum((observed - projected) ** 2, axis=1))))


def _coefficients(dist):
    dist = np.asarray(dist, dtype=np.float64).ravel()
    if len(dist) not in (4, 5, 8) or not np.isfinite(dist).all():
        raise ValueError("Expected 4, 5, or 8 finite OpenCV distortion coefficients")
    return np.pad(dist, (0, 8 - len(dist)))


def _radial_limit(d):
    """First positive radial fold, zero, or pole on the central physical branch."""
    n = np.array([1., d[0], d[1], d[4]])
    den = np.array([1., d[5], d[6], d[7]])
    slope = poly.polysub(poly.polymul(poly.polyadd(n, poly.polymul([0, 2], poly.polyder(n))), den),
                         poly.polymul(poly.polymul([0, 2], n), poly.polyder(den)))
    roots = []
    for coefficients in (n, den, slope):
        for r in poly.polyroots(poly.polytrim(coefficients)):
            if abs(r.imag) < 1e-8 and r.real > 0:
                roots.append(np.sqrt(r.real))
    return min(roots, default=np.inf)


def _distort(xy, d):
    """Normalized OpenCV distortion and its analytic 2x2 Jacobian."""
    x, y = xy[:, 0], xy[:, 1]
    s = x*x + y*y
    n = 1 + d[0]*s + d[1]*s*s + d[4]*s*s*s
    den = 1 + d[5]*s + d[6]*s*s + d[7]*s*s*s
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        f = n / den
        g = ((d[0] + 2*d[1]*s + 3*d[4]*s*s)*den
             - n*(d[5] + 2*d[6]*s + 3*d[7]*s*s)) / den**2
        mapped = np.column_stack((x*f + 2*d[2]*x*y + d[3]*(s+2*x*x),
                                  y*f + d[2]*(s+2*y*y) + 2*d[3]*x*y))
        a = f + 2*x*x*g + 2*d[2]*y + 6*d[3]*x
        b = 2*x*y*g + 2*d[2]*x + 2*d[3]*y
        c = f + 2*y*y*g + 6*d[2]*y + 2*d[3]*x
    return mapped, a, b, c, den


def undistort_points_checked(points, K, dist):
    """Damped Newton inversion; mask noninvertible/nonconverged input pixels.

    Unlike OpenCV's default five fixed-point iterations, convergence is checked
    by redistorting to the original pixels. Solutions cannot cross a radial fold.
    Returned points stay in pixel coordinates; invalid points are NaN.
    """
    K, d = _camera_matrix(K), _coefficients(dist)
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    scale, center = np.diag(K)[:2], K[:2, 2]
    target = (p - center) / scale
    limit = _radial_limit(d)
    xy = target.copy()
    if np.isfinite(limit):
        xy *= np.minimum(1, .8*limit / np.maximum(np.linalg.norm(xy, axis=1), 1e-15))[:, None]
    finite = np.isfinite(target).all(axis=1)
    xy[~finite] = 0
    for _ in range(60):
        mapped, a, b, c, den = _distort(xy, d)
        residual = mapped - target
        error = np.linalg.norm(residual * scale, axis=1)
        active = finite & (error > 1e-7)
        if not active.any():
            break
        det = a*c-b*b
        good = active & np.isfinite(det) & (det > 1e-12) & (den > 1e-10)
        delta = np.zeros_like(xy)
        delta[good, 0] = (c[good]*residual[good, 0] - b[good]*residual[good, 1])/det[good]
        delta[good, 1] = (a[good]*residual[good, 1] - b[good]*residual[good, 0])/det[good]
        delta /= np.maximum(1, np.linalg.norm(delta, axis=1))[:, None]
        accepted = ~good
        for power in range(14):
            trial = xy - delta * (0.5**power)
            tm, ta, tb, tc, td = _distort(trial, d)
            terr = np.linalg.norm((tm-target)*scale, axis=1)
            ok = (~accepted & (terr < error) & (ta*tc-tb*tb > 1e-12)
                  & (td > 1e-10) & (np.linalg.norm(trial, axis=1) < limit))
            xy[ok] = trial[ok]
            accepted |= ok
            if accepted.all():
                break
    mapped, a, b, c, den = _distort(xy, d)
    valid = (finite & np.isfinite(xy).all(axis=1)
             & (np.linalg.norm((mapped-target)*scale, axis=1) < .01)
             & (a*c-b*b > 1e-10) & (den > 1e-10)
             & (np.linalg.norm(xy, axis=1) < limit))
    result = xy*scale + center
    result[~valid] = np.nan
    return result, valid


def distortion_diagnostics(K, dist, image_size):
    """Conservative full-sensor validity screen, not proof of calibration accuracy."""
    K, d = _camera_matrix(K), _coefficients(dist)
    w, h = map(int, image_size)
    if w < 2 or h < 2:
        raise ValueError("Invalid calibration image size")
    xx, yy = np.meshgrid(np.linspace(0, w-1, 65), np.linspace(0, h-1, 37))
    pixels = np.column_stack((xx.ravel(), yy.ravel()))
    xy = (pixels-K[:2, 2])/np.diag(K)[:2]
    _, a, b, c, den = _distort(xy, d)
    det = a*c-b*b
    folded = ~np.isfinite(det) | (det <= 1e-8) | (den <= 1e-8)
    restored, valid = undistort_points_checked(pixels, K, d)
    reasons = []
    if folded.any():
        reasons.append("Default output map folds or has a pole")
    if not valid.all():
        reasons.append("Full sensor cannot be inverted on the central physical branch")
    limit = _radial_limit(d)
    if np.isfinite(limit) and (np.linalg.norm(xy, axis=1) >= limit).any():
        if not folded.any():
            reasons.append("A radial fold or pole crosses the output ray domain")
    return {"valid": not reasons, "reasons": reasons,
            "sample_count": int(len(pixels)),
            "default_output_invalid_fraction": float(np.mean(folded)),
            "inverse_invalid_fraction": float(np.mean(~valid)),
            "radial_branch_limit": float(limit) if np.isfinite(limit) else None,
            "scope": "full_sensor", "grid_size": [65, 37]}
