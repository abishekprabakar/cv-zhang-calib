"""Zhang's planar calibration (Zhang, PAMI 2000), implemented from scratch.

Pipeline:
  1. One homography per view, board plane (Z=0) -> image, via normalised DLT.
  2. Each homography H = lambda K [r1 r2 t] gives two linear constraints on the
     image of the absolute conic B = K^-T K^-1 (r1 . r2 = 0, |r1| = |r2|).
     Stack 2N constraints, take the SVD null vector, read K off B in closed form.
  3. Recover [R | t] for every view from K^-1 H, projected onto SO(3).
  4. Linear least squares for radial k1, k2 against the ideal projections.
  5. Levenberg-Marquardt over all intrinsics, distortion and 6N pose parameters,
     minimising pixel reprojection error. Parameter standard errors come from
     sigma^2 (J^T J)^-1 at the optimum.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import Camera, rodrigues, rodrigues_inv


# --------------------------------------------------------------------------- homography
def _normalizer(pts: np.ndarray) -> np.ndarray:
    """Hartley normalisation: centroid to origin, mean distance sqrt(2)."""
    c = pts.mean(axis=0)
    d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean()
    s = np.sqrt(2) / d
    return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])


def homography_dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """H with dst ~ H src, from >= 4 point pairs (N x 2 each). H[2,2] = 1."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    if len(src) < 4:
        raise ValueError("homography needs at least 4 correspondences")
    Ts, Td = _normalizer(src), _normalizer(dst)
    s = (Ts @ np.c_[src, np.ones(len(src))].T).T
    d = (Td @ np.c_[dst, np.ones(len(dst))].T).T
    rows = []
    for (x, y, _), (u, v, _) in zip(s, d):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, Vt = np.linalg.svd(np.asarray(rows))
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2]


# --------------------------------------------------------------------------- closed form
def _v(H: np.ndarray, i: int, j: int) -> np.ndarray:
    hi, hj = H[:, i], H[:, j]
    return np.array([
        hi[0] * hj[0],
        hi[0] * hj[1] + hi[1] * hj[0],
        hi[1] * hj[1],
        hi[2] * hj[0] + hi[0] * hj[2],
        hi[2] * hj[1] + hi[1] * hj[2],
        hi[2] * hj[2],
    ])


def intrinsics_from_homographies(Hs: list[np.ndarray]) -> np.ndarray:
    """Closed-form K (with skew) from >= 3 homographies."""
    if len(Hs) < 3:
        raise ValueError("closed-form calibration with free skew needs at least 3 views")
    V = []
    for H in Hs:
        V.append(_v(H, 0, 1))
        V.append(_v(H, 0, 0) - _v(H, 1, 1))
    V = np.asarray(V)
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    _, S, Vt = np.linalg.svd(V)
    if S[-2] < 1e-8 * S[0]:
        raise ValueError("degenerate view set: the conic constraints have a multi-dimensional "
                         "null space (boards fronto-parallel or all poses alike)")
    B11, B12, B22, B13, B23, B33 = Vt[-1]
    den = B11 * B22 - B12 ** 2
    v0 = (B12 * B13 - B11 * B23) / den
    lam = B33 - (B13 ** 2 + v0 * (B12 * B13 - B11 * B23)) / B11
    if not np.isfinite([den, lam]).all() or lam / B11 <= 0 or lam * B11 / den <= 0:
        raise ValueError("degenerate view set: absolute conic is not positive definite "
                         "(are the board poses all parallel?)")
    alpha = np.sqrt(lam / B11)
    beta = np.sqrt(lam * B11 / den)
    gamma = -B12 * alpha ** 2 * beta / lam
    u0 = gamma * v0 / beta - B13 * alpha ** 2 / lam
    return np.array([[alpha, gamma, u0], [0, beta, v0], [0, 0, 1.0]])


def extrinsics_from_homography(K: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[R | t] for one view, R projected to the nearest rotation (Frobenius)."""
    Kinv = np.linalg.inv(K)
    h1, h2, h3 = (Kinv @ H).T
    lam = 1.0 / np.linalg.norm(h1)
    if (lam * h3)[2] < 0:          # board must lie in front of the camera
        lam = -lam
    r1, r2, t = lam * h1, lam * h2, lam * h3
    Q = np.column_stack([r1, r2, np.cross(r1, r2)])
    U, _, Vt = np.linalg.svd(Q)
    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    return rodrigues_inv(R), t


def radial_init(cam: Camera, obj: list[np.ndarray], img: list[np.ndarray],
                rvecs, tvecs) -> np.ndarray:
    """Zhang's linear estimate of k1, k2 given distortion-free K and poses."""
    A, b = [], []
    for X, uv, r, t in zip(obj, img, rvecs, tvecs):
        Xc = X @ rodrigues(r).T + t
        xy = Xc[:, :2] / Xc[:, 2:3]
        r2 = (xy ** 2).sum(axis=1)
        ideal = cam.to_pixels(xy)
        for c, centre in ((0, cam.cx), (1, cam.cy)):
            d = ideal[:, c] - centre
            A.append(np.column_stack([d * r2, d * r2 * r2]))
            b.append(uv[:, c] - ideal[:, c])
    k, *_ = np.linalg.lstsq(np.vstack(A), np.concatenate(b), rcond=None)
    return k


# --------------------------------------------------------------------------- refinement
N_INTR = 8  # fx, fy, cx, cy, k1, k2, p1, p2  (skew fixed at 0 during refinement)


def _unpack_cam(p: np.ndarray) -> Camera:
    return Camera(p[0], p[1], p[2], p[3], 0.0, p[4:8].copy())


def _view_residual(intr: np.ndarray, pose: np.ndarray, X: np.ndarray, uv: np.ndarray) -> np.ndarray:
    return (_unpack_cam(intr).project(X, pose[:3], pose[3:]) - uv).ravel()


def _jacobian(p, obj, img, n_views):
    """Block-sparse structure: view i depends on the 8 intrinsics and its own 6 pose params."""
    rows = sum(2 * len(X) for X in obj)
    J = np.zeros((rows, len(p)))
    r = np.zeros(rows)
    row = 0
    intr = p[:N_INTR]
    for i in range(n_views):
        X, uv = obj[i], img[i]
        sl = slice(N_INTR + 6 * i, N_INTR + 6 * i + 6)
        pose = p[sl]
        m = 2 * len(X)
        r[row:row + m] = _view_residual(intr, pose, X, uv)
        local = np.concatenate([intr, pose])
        cols = list(range(N_INTR)) + list(range(sl.start, sl.stop))
        for k, col in enumerate(cols):
            h = 1e-6 * max(1.0, abs(local[k]))
            lp, lm = local.copy(), local.copy()
            lp[k] += h
            lm[k] -= h
            fp = _view_residual(lp[:N_INTR], lp[N_INTR:], X, uv)
            fm = _view_residual(lm[:N_INTR], lm[N_INTR:], X, uv)
            J[row:row + m, col] = (fp - fm) / (2 * h)
        row += m
    return J, r


def levenberg_marquardt(p0, obj, img, max_iter=100, tol=1e-12):
    n_views = len(obj)
    p = p0.copy()
    J, r = _jacobian(p, obj, img, n_views)
    cost = r @ r
    lam = 1e-3
    it = 0
    for it in range(1, max_iter + 1):
        A = J.T @ J
        g = J.T @ r
        improved = False
        while lam < 1e12:
            step = np.linalg.solve(A + lam * np.diag(np.diag(A)), -g)
            p_new = p + step
            r_new = np.concatenate([
                _view_residual(p_new[:N_INTR], p_new[N_INTR + 6 * i:N_INTR + 6 * i + 6], obj[i], img[i])
                for i in range(n_views)])
            c_new = r_new @ r_new
            if c_new < cost:
                lam = max(lam / 3, 1e-12)
                improved = True
                break
            lam *= 4
        if not improved:
            break
        rel = (cost - c_new) / max(cost, 1e-300)
        p, cost = p_new, c_new
        J, r = _jacobian(p, obj, img, n_views)
        if rel < tol or np.linalg.norm(step) < 1e-12 * (1 + np.linalg.norm(p)):
            break
    return p, J, r, it


# --------------------------------------------------------------------------- driver
@dataclass
class CalibrationResult:
    camera: Camera
    rvecs: list
    tvecs: list
    rms: float                    # RMS over all x and y residuals, pixels
    per_view_rms: np.ndarray
    std_intrinsics: np.ndarray    # 1-sigma for fx, fy, cx, cy, k1, k2, p1, p2
    K_closed_form: np.ndarray
    iterations: int

    def summary(self) -> str:
        c, s = self.camera, self.std_intrinsics
        names = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"]
        vals = [c.fx, c.fy, c.cx, c.cy, *c.dist]
        lines = [f"  {n:>2} = {v:12.6f}  +/- {e:.6f}" for n, v, e in zip(names, vals, s)]
        return "\n".join(lines + [f"  RMS reprojection error = {self.rms:.4f} px "
                                  f"({self.iterations} LM iterations)"])


def calibrate(obj_points: list[np.ndarray], img_points: list[np.ndarray]) -> CalibrationResult:
    """obj_points: per view, N x 3 board points with Z = 0. img_points: per view, N x 2 pixels."""
    obj = [np.asarray(X, float) for X in obj_points]
    img = [np.asarray(u, float).reshape(-1, 2) for u in img_points]
    if len(obj) != len(img):
        raise ValueError("obj_points and img_points must have the same number of views")
    for X in obj:
        if np.abs(X[:, 2]).max() > 1e-9:
            raise ValueError("planar calibration expects board points with Z = 0")

    Hs = [homography_dlt(X[:, :2], uv) for X, uv in zip(obj, img)]
    K0 = intrinsics_from_homographies(Hs)
    poses = [extrinsics_from_homography(K0, H) for H in Hs]
    rvecs = [p[0] for p in poses]
    tvecs = [p[1] for p in poses]

    cam0 = Camera(K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2])
    k12 = radial_init(cam0, obj, img, rvecs, tvecs)

    p0 = np.concatenate([[K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2], k12[0], k12[1], 0.0, 0.0],
                         *[np.concatenate([r, t]) for r, t in zip(rvecs, tvecs)]])
    p, J, r, iters = levenberg_marquardt(p0, obj, img)

    m, n = len(r), len(p)
    sigma2 = (r @ r) / max(m - n, 1)
    cov = sigma2 * np.linalg.inv(J.T @ J)
    std = np.sqrt(np.diag(cov))[:N_INTR]

    per_view, row = [], 0
    for X in obj:
        k = 2 * len(X)
        per_view.append(np.sqrt(np.mean(r[row:row + k] ** 2)))
        row += k
    return CalibrationResult(
        camera=_unpack_cam(p),
        rvecs=[p[N_INTR + 6 * i:N_INTR + 6 * i + 3] for i in range(len(obj))],
        tvecs=[p[N_INTR + 6 * i + 3:N_INTR + 6 * i + 6] for i in range(len(obj))],
        rms=float(np.sqrt(np.mean(r ** 2))),
        per_view_rms=np.asarray(per_view),
        std_intrinsics=std,
        K_closed_form=K0,
        iterations=iters,
    )
