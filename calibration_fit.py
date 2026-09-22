"""Choose a physically usable lens model using explicitly held-out board views."""

import cv2
import numpy as np

from calibration_geometry import (
    distortion_diagnostics,
    reprojection_rms,
    undistort_points_checked,
)

_MIN_TRAINING_VIEWS = 5
_MIN_VALIDATION_VIEWS = 3
_MAX_VALIDATION_RMS_PX = 2.0
_SIMPLER_MODEL_TOLERANCE_PX = 0.05
_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-9)


def rational_positive_poles(dist):
    """Positive denominator-zero radii for a finite OpenCV 4/5/8-coefficient model.

    These are normalized undistorted ray radii, not pixels. Polynomial models
    return an empty list. Near-real roots use a relative numerical tolerance so
    repeated real roots perturbed by the eigensolver are conservatively retained.
    Numerator cancellation does not remove a denominator zero from this policy.
    """
    dist = np.asarray(dist, dtype=float).ravel()
    if len(dist) not in (4, 5, 8) or not np.isfinite(dist).all():
        raise ValueError("Expected 4, 5, or 8 finite OpenCV distortion coefficients")
    if len(dist) < 8:
        return []
    roots = np.polynomial.polynomial.polyroots([1., *dist[5:8]])
    if not np.isfinite(roots).all():
        raise ValueError("Rational denominator roots are nonfinite")
    return sorted(float(np.sqrt(root.real)) for root in roots
                  if root.real > 0
                  and abs(root.imag) <= 1e-7 * max(1., abs(root.real)))


def _views(obj_points, img_points, label, minimum):
    if obj_points is None or img_points is None:
        raise ValueError(f"{label} requires at least {minimum} explicit board views")
    if len(obj_points) != len(img_points) or len(obj_points) < minimum:
        raise ValueError(f"{label} requires at least {minimum} corresponding board views")
    objects, images = [], []
    for index, (obj, img) in enumerate(zip(obj_points, img_points)):
        obj = np.array(obj, dtype=np.float32, copy=True).reshape(-1, 3)
        img = np.array(img, dtype=np.float32, copy=True).reshape(-1, 1, 2)
        if (len(obj) != len(img) or len(obj) < 6
                or not np.isfinite(obj).all() or not np.isfinite(img).all()):
            raise ValueError(f"Invalid {label.lower()} board view {index}")
        objects.append(obj)
        images.append(img)
    return objects, images


def _heldout_error(objects, images, K, dist):
    """Fit only each held-out board pose, retaining the training intrinsics."""
    errors = []
    squared_error_sum = 0.0
    point_count = 0
    for obj, img in zip(objects, images):
        undistorted, valid = undistort_points_checked(img, K, dist)
        if not valid.all():
            raise ValueError("Held-out board has points outside the invertible lens domain")
        # The checked inverse avoids OpenCV's five-iteration initial undistortion.
        found, rvec, tvec = cv2.solvePnP(
            obj, undistorted.reshape(-1, 1, 2), K, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not found:
            raise ValueError("Could not solve a held-out board pose")
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec,
                                       criteria=_CRITERIA)
        camera_points = (cv2.Rodrigues(rvec)[0] @ obj.astype(float).T
                         + np.asarray(tvec).reshape(3, 1)).T
        if not np.isfinite(camera_points).all() or np.any(camera_points[:, 2] <= 0):
            raise ValueError("Held-out board pose has nonpositive camera depth")
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        error = reprojection_rms(img, projected)
        if not np.isfinite(error):
            raise ValueError("Nonfinite held-out reprojection error")
        errors.append(error)
        squared_error_sum += error * error * len(obj)
        point_count += len(obj)
    return float(np.sqrt(squared_error_sum / point_count)), errors


def fit_intrinsics(obj_points, img_points, image_size,
                   validation_obj_points=None, validation_img_points=None):
    """Refit several models and select using untouched validation board images.

    At least five training and three explicit held-out views are required. The
    caller owns the split; no validation view is passed to calibrateCamera.
    Returned rvecs/tvecs correspond to the supplied training views in input order.
    New rational fits must have no positive real denominator zeros at any radius;
    even a pole outside the sampled image can create unstable extrapolation.
    Model selection also requires full-sensor distortion validity and <=2 px
    held-out RMS, then prefers fewer free distortion coefficients when scores
    differ by no more than 0.05 px. This screen does not establish edge accuracy
    when the board observations themselves have poor spatial coverage.
    """
    objects, images = _views(obj_points, img_points, "Training", _MIN_TRAINING_VIEWS)
    validation_objects, validation_images = _views(
        validation_obj_points, validation_img_points, "Held-out validation",
        _MIN_VALIDATION_VIEWS)
    if len(image_size) != 2 or min(image_size) < 2:
        raise ValueError("Expected a positive (width, height) image size")
    image_size = tuple(int(value) for value in image_size)
    # Catch exact accidental leakage while allowing the object grid to be reused.
    training_images = {image.tobytes() for image in images}
    if any(image.tobytes() in training_images for image in validation_images):
        raise ValueError("Held-out validation contains a training image's corner coordinates")

    candidates = (
        ("opencv5", 5, 5, cv2.CALIB_USE_LU),
        ("opencv4", 4, 4, cv2.CALIB_USE_LU | cv2.CALIB_FIX_K3),
        ("rational_low", 4, 8,
         cv2.CALIB_USE_LU | cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_USE_INTRINSIC_GUESS
         | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6),
        ("rational6", 6, 8,
         cv2.CALIB_USE_LU | cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_USE_INTRINSIC_GUESS
         | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K6),
        ("rational8", 8, 8,
         cv2.CALIB_USE_LU | cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_USE_INTRINSIC_GUESS),
    )
    statistics, fitted = [], []
    basic_fit = None
    low_fit = None
    quadratic_fit = None
    for model, complexity, coefficients, flags in candidates:
        stats = {"model": model, "free_distortion_coefficients": complexity,
                 "rms_px": None, "heldout_rms_px": None,
                 "heldout_per_view_rms_px": [], "valid": False,
                 "rejection_reasons": []}
        statistics.append(stats)
        try:
            initial_K, initial_dist = None, None
            if model.startswith("rational"):
                if basic_fit is not None:
                    initial_K = basic_fit[0].copy()
                else:
                    initial_K = cv2.initCameraMatrix2D(objects, images, image_size, 0)
                initial_dist = np.zeros((8, 1), dtype=float)
                initial_dist[0, 0], initial_dist[5, 0] = .2, .5
                seed = quadratic_fit if model == "rational8" and quadratic_fit is not None else low_fit
                if model in ("rational6", "rational8") and seed is not None:
                    initial_K, initial_dist = (seed[0].copy(), seed[1].reshape(8, 1).copy())
            rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                objects, images, image_size, initial_K, initial_dist,
                flags=flags, criteria=_CRITERIA)
            K = np.asarray(K, dtype=float)
            dist = np.asarray(dist, dtype=float).ravel()[:coefficients].copy()
            stats["rms_px"] = float(rms) if np.isfinite(rms) else None
            if (not np.isfinite(rms) or not np.isfinite(K).all()
                    or not np.isfinite(dist).all()):
                raise ValueError("Calibration produced nonfinite parameters")
            if model == "opencv5":
                basic_fit = (K, dist)
            poles = rational_positive_poles(dist)
            stats["positive_denominator_pole_radii"] = poles
            diagnostics = distortion_diagnostics(K, dist, image_size)
            stats["diagnostics"] = diagnostics
            if not diagnostics["valid"]:
                stats["rejection_reasons"].extend(diagnostics["reasons"])
            if poles:
                stats["rejection_reasons"].append(
                    "New rational fit has a positive real denominator pole "
                    f"at normalized radius {poles[0]:.9g}; finite poles are rejected "
                    "even outside the sampled image or near numerator cancellation")
            if stats["rejection_reasons"]:
                continue
            if model == "rational_low":
                low_fit = (K, dist)
            if model == "rational6":
                quadratic_fit = (K, dist)
            heldout_rms, heldout_per_view = _heldout_error(
                validation_objects, validation_images, K, dist)
            if not np.isfinite(heldout_rms):
                raise ValueError("Nonfinite held-out reprojection error")
            stats["heldout_rms_px"] = heldout_rms
            stats["heldout_per_view_rms_px"] = heldout_per_view
            if heldout_rms > _MAX_VALIDATION_RMS_PX:
                stats["rejection_reasons"].append(
                    f"Held-out RMS {heldout_rms:.3f} px exceeds {_MAX_VALIDATION_RMS_PX:.1f} px")
                continue
            stats["valid"] = True
            fitted.append({"K": K, "dist": dist, "rms": float(rms),
                           "rvecs": rvecs, "tvecs": tvecs,
                           "diagnostics": diagnostics, "model": model,
                           "_statistics": stats})
        except (cv2.error, ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            stats["rejection_reasons"].append(str(exc))

    if not fitted:
        details = "; ".join(
            f"{entry['model']} (training RMS={entry['rms_px']}, "
            f"held-out RMS={entry['heldout_rms_px']}): "
            + ", ".join(entry["rejection_reasons"]) for entry in statistics)
        raise ValueError("No valid intrinsic model passed geometry and held-out checks. " + details)
    best_score = min(entry["_statistics"]["heldout_rms_px"] for entry in fitted)
    equivalent = [entry for entry in fitted if
                  entry["_statistics"]["heldout_rms_px"] <= best_score + _SIMPLER_MODEL_TOLERANCE_PX]
    selected = min(equivalent, key=lambda entry: (
        entry["_statistics"]["free_distortion_coefficients"],
        entry["_statistics"]["heldout_rms_px"], entry["rms"]))
    selected_stats = selected.pop("_statistics")
    selected["selection"] = {
        "selected_model": selected["model"],
        "training_view_count": len(objects),
        "heldout_view_count": len(validation_objects),
        "heldout_rms_px": selected_stats["heldout_rms_px"],
        "heldout_per_view_rms_px": selected_stats["heldout_per_view_rms_px"],
        "max_heldout_rms_px": _MAX_VALIDATION_RMS_PX,
        "simpler_model_tolerance_px": _SIMPLER_MODEL_TOLERANCE_PX,
        "rational_pole_policy": "reject_any_positive_real_denominator_zero",
        "candidates": statistics,
    }
    return selected
