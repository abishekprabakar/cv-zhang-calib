"""Procedural checkerboard scenes: board geometry, random poses, a ray-traced
renderer through the distorted camera, and OpenCV corner detection.

Everything is generated from a seed, so the repo carries no image data and every
test is deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import Camera, rodrigues, rodrigues_inv


@dataclass(frozen=True)
class Board:
    cols: int = 9          # inner corners along x (OpenCV pattern width)
    rows: int = 6          # inner corners along y
    square: float = 25.0   # millimetres

    def object_points(self) -> np.ndarray:
        """Inner corners in OpenCV order (row-major, x fastest), Z = 0, origin at the first corner."""
        j, i = np.mgrid[0:self.rows, 0:self.cols]
        return np.column_stack([i.ravel() * self.square, j.ravel() * self.square,
                                np.zeros(i.size)])

    @property
    def centre(self) -> np.ndarray:
        return np.array([(self.cols - 1) * self.square / 2, (self.rows - 1) * self.square / 2, 0.0])

    def outline(self, margin_squares: float = 2.0) -> np.ndarray:
        """Four corners of the printed sheet (one ring of squares plus a white margin)."""
        s = self.square
        x0, y0 = -s * margin_squares, -s * margin_squares
        x1, y1 = (self.cols + margin_squares - 1) * s, (self.rows + margin_squares - 1) * s
        return np.array([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]])


GROUND_TRUTH = Camera(fx=520.0, fy=515.0, cx=322.5, cy=236.8, skew=0.0,
                      dist=np.array([-0.28, 0.09, 0.0012, -0.0008]))
IMAGE_SIZE = (640, 480)  # width, height


def random_poses(board: Board, cam: Camera, n: int, seed: int = 7,
                 size=IMAGE_SIZE, max_tilt_deg: float = 45.0):
    """n board poses with varied tilt, all fully in view, as (rvec, tvec) pairs.

    Tilt variety matters: Zhang's constraints are degenerate when every board is
    parallel to the image plane (pure in-plane rotation/translation).
    """
    rng = np.random.default_rng(seed)
    w, h = size
    out = []
    while len(out) < n:
        ax, ay = np.radians(rng.uniform(-max_tilt_deg, max_tilt_deg, 2))
        az = np.radians(rng.uniform(-30, 30))
        R = rodrigues([0, 0, az]) @ rodrigues([0, ay, 0]) @ rodrigues([ax, 0, 0])
        Z = rng.uniform(330, 520)
        C = np.array([rng.uniform(-0.18, 0.18) * Z, rng.uniform(-0.13, 0.13) * Z, Z])
        t = C - R @ board.centre
        sheet = cam.project(board.outline(1.0), rodrigues_inv(R), t)
        Xc = board.outline(1.0) @ R.T + t
        if (Xc[:, 2] <= 50).any():
            continue
        if (sheet[:, 0] < 8).any() or (sheet[:, 0] > w - 9).any() or \
           (sheet[:, 1] < 8).any() or (sheet[:, 1] > h - 9).any():
            continue
        out.append((rodrigues_inv(R), t))
    return out


class Renderer:
    """Ray-traces the board through the distorted lens with 3x3 supersampling.

    The undistorted ray for each sub-pixel depends only on the camera, so it is
    computed once and reused for every pose.
    """

    def __init__(self, cam: Camera, size=IMAGE_SIZE, ss: int = 3):
        w, h = size
        self.cam, self.size, self.ss = cam, size, ss
        offs = (np.arange(ss) + 0.5) / ss - 0.5
        v, u = np.mgrid[0:h, 0:w].astype(float)
        shape = (h, w, ss, ss)
        uu = np.broadcast_to(u[..., None, None] + offs[None, None, None, :], shape).ravel()
        vv = np.broadcast_to(v[..., None, None] + offs[None, None, :, None], shape).ravel()
        xy = cam.undistort(cam.from_pixels(np.column_stack([uu, vv])), iters=60)
        self.rays = np.column_stack([xy, np.ones(len(xy))])

    def render(self, board: Board, rvec, tvec, seed: int = 0,
               blur_sigma: float = 0.6, noise: float = 2.0) -> np.ndarray:
        w, h = self.size
        R, t = rodrigues(rvec), np.asarray(tvec, float)
        n = R[:, 2]
        denom = self.rays @ n
        lam = np.where(np.abs(denom) > 1e-12, (n @ t) / denom, -1.0)
        P = lam[:, None] * self.rays - t           # camera frame, relative to board origin
        B = P @ R                                   # = R^T P, board frame
        x, y = B[:, 0], B[:, 1]
        s = board.square
        lo = board.outline(2.0)
        on_sheet = (lam > 0) & (x >= lo[0, 0]) & (x <= lo[2, 0]) & (y >= lo[0, 1]) & (y <= lo[2, 1])
        checker = (x >= -s) & (x <= board.cols * s) & (y >= -s) & (y <= board.rows * s)
        parity = (np.floor(x / s) + np.floor(y / s)).astype(int) % 2
        rng = np.random.default_rng(seed)
        # smooth vertical gradient for the background (per sub-pixel row)
        row = np.repeat(np.arange(h), w * self.ss * self.ss) / max(h - 1, 1)
        bg = 0.42 - 0.18 * row
        val = np.where(on_sheet, np.where(checker & (parity == 0), 0.08, 0.92), bg)
        img = val.reshape(h, w, self.ss * self.ss).mean(axis=2)
        if blur_sigma > 0:
            img = _gaussian_blur(img, blur_sigma)
        img = img * 255 + rng.normal(0, noise, img.shape)
        return np.clip(np.round(img), 0, 255).astype(np.uint8)


def _gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(img, r, mode="edge")
    tmp = np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 1, pad)
    return np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 0, tmp)


def detect_corners(gray: np.ndarray, board: Board):
    """OpenCV chessboard detection + sub-pixel refinement. Returns N x 2 or None."""
    import cv2
    ok, corners = cv2.findChessboardCorners(
        gray, (board.cols, board.rows),
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
    corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), crit)
    return corners.reshape(-1, 2).astype(float)


def undistort_image(img: np.ndarray, cam: Camera) -> np.ndarray:
    """Resample a distorted image into an ideal pinhole image with the same K."""
    import cv2
    h, w = img.shape[:2]
    v, u = np.mgrid[0:h, 0:w].astype(float)
    xy = cam.from_pixels(np.column_stack([u.ravel(), v.ravel()]))
    src = cam.to_pixels(cam.distort(xy))
    mx = src[:, 0].reshape(h, w).astype(np.float32)
    my = src[:, 1].reshape(h, w).astype(np.float32)
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderValue=0)
