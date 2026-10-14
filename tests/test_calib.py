"""Tests with expected values from geometry, not from this code's own output.

Ground truth comes from the synthetic camera; independent references are the
Rodrigues closed form, exact homographies built from K[r1 r2 t], and OpenCV's
calibrateCamera run on the same correspondences.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calib import (Board, Camera, GROUND_TRUTH, Renderer, calibrate, detect_corners,  # noqa: E402
                   distort_normalized, extrinsics_from_homography, homography_dlt,
                   intrinsics_from_homographies, random_poses, rodrigues, rodrigues_inv,
                   undistort_normalized)

BOARD = Board()
X = BOARD.object_points()


def _angle_between(Ra, Rb):
    return np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))


# ------------------------------------------------------------------ rotations
def test_rodrigues_quarter_turn_about_z():
    R = rodrigues([0, 0, np.pi / 2])
    np.testing.assert_allclose(R, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)


@pytest.mark.parametrize("rvec", [[0.1, -0.2, 0.3], [1e-10, 0, 0], [0, np.pi - 1e-9, 0], [2.0, 1.0, -0.5]])
def test_rodrigues_round_trip_and_orthonormal(rvec):
    R = rodrigues(rvec)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(rodrigues(rodrigues_inv(R)), R, atol=1e-8)


# ------------------------------------------------------------------ distortion
def test_distortion_round_trip():
    rng = np.random.default_rng(0)
    xy = rng.uniform(-0.6, 0.6, (500, 2))
    back = undistort_normalized(distort_normalized(xy, GROUND_TRUTH.dist), GROUND_TRUTH.dist)
    np.testing.assert_allclose(back, xy, atol=1e-10)


def test_zero_distortion_is_identity_and_radial_formula_by_hand():
    xy = np.array([[0.3, -0.4]])
    np.testing.assert_allclose(distort_normalized(xy, np.zeros(4)), xy)
    # r^2 = 0.25 -> radial = 1 - 0.2*0.25 + 0.04*0.0625 = 0.9525
    np.testing.assert_allclose(distort_normalized(xy, np.array([-0.2, 0.04, 0, 0])),
                               [[0.3 * 0.9525, -0.4 * 0.9525]], atol=1e-15)


def test_undistort_pixels_matches_ideal_pinhole():
    r, t = random_poses(BOARD, GROUND_TRUTH, 1, seed=3)[0]
    distorted = GROUND_TRUTH.project(X, r, t)
    pinhole = Camera(GROUND_TRUTH.fx, GROUND_TRUTH.fy, GROUND_TRUTH.cx, GROUND_TRUTH.cy).project(X, r, t)
    np.testing.assert_allclose(GROUND_TRUTH.undistort_pixels(distorted), pinhole, atol=1e-8)


# ------------------------------------------------------------------ homography + closed form
def _exact_H(K, r, t):
    R = rodrigues(r)
    H = K @ np.column_stack([R[:, 0], R[:, 1], t])
    return H / H[2, 2]


def test_homography_dlt_exact_on_noiseless_points():
    pin = Camera(600, 590, 310, 250)
    r, t = random_poses(BOARD, pin, 1, seed=11)[0]
    uv = pin.project(X, r, t)
    np.testing.assert_allclose(homography_dlt(X[:, :2], uv), _exact_H(pin.K, r, t), rtol=1e-8, atol=1e-8)


def test_closed_form_recovers_K_with_skew_exactly():
    K = np.array([[700.0, 1.5, 330.0], [0, 690.0, 245.0], [0, 0, 1]])
    poses = random_poses(BOARD, Camera(700, 690, 330, 245), 5, seed=2)
    Hs = [_exact_H(K, r, t) for r, t in poses]
    np.testing.assert_allclose(intrinsics_from_homographies(Hs), K, rtol=1e-9, atol=1e-7)
    for (r, t), H in zip(poses, Hs):
        re, te = extrinsics_from_homography(K, H)
        assert _angle_between(rodrigues(r), rodrigues(re)) < 1e-6
        np.testing.assert_allclose(te, t, rtol=1e-9)


def test_closed_form_needs_three_views():
    poses = random_poses(BOARD, GROUND_TRUTH, 2, seed=1)
    with pytest.raises(ValueError):
        intrinsics_from_homographies([_exact_H(GROUND_TRUTH.K, r, t) for r, t in poses])


def test_fronto_parallel_boards_are_degenerate():
    # Pure in-plane motion: every homography has the same r1, r2 up to a z-rotation,
    # so the 2N constraints span too little of B to pin down K.
    K = GROUND_TRUTH.K
    Hs = [_exact_H(K, [0, 0, a], [-100 + 10 * a, -60, 400 + 30 * a]) for a in (0.0, 0.5, 1.0, 1.5)]
    with pytest.raises(ValueError, match="degenerate"):
        intrinsics_from_homographies(Hs)


# ------------------------------------------------------------------ full calibration on points
@pytest.fixture(scope="module")
def noisy_points():
    poses = random_poses(BOARD, GROUND_TRUTH, 15, seed=21)
    rng = np.random.default_rng(5)
    sigma = 0.25
    obs = [GROUND_TRUTH.project(X, r, t) + rng.normal(0, sigma, (len(X), 2)) for r, t in poses]
    return poses, obs, sigma


def test_noiseless_points_recover_everything(noisy_points):
    poses, _, _ = noisy_points
    obs = [GROUND_TRUTH.project(X, r, t) for r, t in poses]
    res = calibrate([X] * len(obs), obs)
    c = res.camera
    np.testing.assert_allclose([c.fx, c.fy, c.cx, c.cy], [520, 515, 322.5, 236.8], atol=1e-5)
    np.testing.assert_allclose(c.dist, GROUND_TRUTH.dist, atol=1e-8)
    assert res.rms < 1e-6
    for (r, t), re, te in zip(poses, res.rvecs, res.tvecs):
        assert _angle_between(rodrigues(r), rodrigues(re)) < 1e-6
        np.testing.assert_allclose(te, t, atol=1e-5)


def test_noisy_points_within_reported_uncertainty(noisy_points):
    _, obs, sigma = noisy_points
    res = calibrate([X] * len(obs), obs)
    c = res.camera
    est = np.array([c.fx, c.fy, c.cx, c.cy, *c.dist])
    truth = np.array([GROUND_TRUTH.fx, GROUND_TRUTH.fy, GROUND_TRUTH.cx, GROUND_TRUTH.cy, *GROUND_TRUTH.dist])
    z = np.abs(est - truth) / res.std_intrinsics
    assert z.max() < 4.0, z
    # Residual RMS ~ sigma * sqrt(1 - p/m) for a least-squares fit with p params, m residuals.
    m, p = 2 * len(X) * len(obs), 8 + 6 * len(obs)
    assert res.rms == pytest.approx(sigma * np.sqrt(1 - p / m), rel=0.06)


def test_matches_opencv_calibrate_camera(noisy_points):
    cv2 = pytest.importorskip("cv2")
    _, obs, _ = noisy_points
    res = calibrate([X] * len(obs), obs)
    rms, K, d, _, _ = cv2.calibrateCamera([X.astype(np.float32)] * len(obs),
                                          [o.astype(np.float32).reshape(-1, 1, 2) for o in obs],
                                          (640, 480), None, None, flags=cv2.CALIB_FIX_K3)
    c = res.camera
    np.testing.assert_allclose([c.fx, c.fy, c.cx, c.cy], [K[0, 0], K[1, 1], K[0, 2], K[1, 2]], rtol=1e-5)
    np.testing.assert_allclose(c.dist, d.ravel()[:4], atol=1e-5)
    # OpenCV reports sqrt(sum |r|^2 / n_points); ours averages over x and y separately.
    assert rms == pytest.approx(res.rms * np.sqrt(2), rel=1e-4)


def test_rejects_non_planar_points():
    with pytest.raises(ValueError):
        calibrate([X + [0, 0, 1.0]] * 3, [np.zeros((len(X), 2))] * 3)


# ------------------------------------------------------------------ rendered images end to end
def test_rendered_images_end_to_end():
    pytest.importorskip("cv2")
    poses = random_poses(BOARD, GROUND_TRUTH, 10, seed=99)
    ren = Renderer(GROUND_TRUTH)
    obs, det = [], []
    for k, (r, t) in enumerate(poses):
        c = detect_corners(ren.render(BOARD, r, t, seed=k), BOARD)
        assert c is not None, f"board not found in view {k}"
        gt = GROUND_TRUTH.project(X, r, t)
        det.append(np.linalg.norm(c[:, None] - gt[None], axis=2).min(axis=1))
        obs.append(c)
    det = np.concatenate(det)
    assert np.sqrt(np.mean(det ** 2)) < 0.15          # sub-pixel corner localisation
    res = calibrate([X] * len(obs), obs)
    c = res.camera
    assert abs(c.fx - 520) / 520 < 0.003 and abs(c.fy - 515) / 515 < 0.003
    assert abs(c.cx - 322.5) < 2.0 and abs(c.cy - 236.8) < 2.0
    assert abs(c.dist[0] - (-0.28)) < 0.01
    assert res.rms < 0.15
