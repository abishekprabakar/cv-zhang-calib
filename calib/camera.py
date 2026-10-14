"""Pinhole camera with Brown-Conrady lens distortion (OpenCV convention).

A world point X in board coordinates maps to pixels as:

    Xc = R X + t                          (rigid transform, R from a Rodrigues vector)
    x, y = Xc / Zc, Yc / Zc               (normalised image plane)
    r2 = x^2 + y^2
    radial = 1 + k1 r2 + k2 r2^2
    xd = x radial + 2 p1 x y + p2 (r2 + 2 x^2)
    yd = y radial + p1 (r2 + 2 y^2) + 2 p2 x y
    u = fx xd + skew yd + cx,   v = fy yd + cy
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def rodrigues(rvec: np.ndarray) -> np.ndarray:
    """Axis-angle vector -> 3x3 rotation matrix."""
    rvec = np.asarray(rvec, dtype=float).reshape(3)
    theta = np.linalg.norm(rvec)
    if theta < 1e-12:
        kx = np.array([[0, -rvec[2], rvec[1]], [rvec[2], 0, -rvec[0]], [-rvec[1], rvec[0], 0]])
        return np.eye(3) + kx
    k = rvec / theta
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * kx + (1 - np.cos(theta)) * (kx @ kx)


def rodrigues_inv(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> axis-angle vector (handles angles near 0 and pi)."""
    R = np.asarray(R, dtype=float)
    cos_t = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    theta = np.arccos(cos_t)
    if theta < 1e-9:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2
    if np.pi - theta < 1e-6:
        # sin(theta) ~ 0: recover the axis from the symmetric part R = 2kk^T - I
        M = (R + np.eye(3)) / 2
        i = int(np.argmax(np.diag(M)))
        k = M[:, i] / np.sqrt(M[i, i])
        return k * theta
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * theta / (2 * np.sin(theta))


@dataclass
class Camera:
    fx: float
    fy: float
    cx: float
    cy: float
    skew: float = 0.0
    dist: np.ndarray = field(default_factory=lambda: np.zeros(4))  # k1, k2, p1, p2

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, self.skew, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    def distort(self, xy: np.ndarray) -> np.ndarray:
        return distort_normalized(xy, self.dist)

    def undistort(self, xy_d: np.ndarray, iters: int = 50) -> np.ndarray:
        return undistort_normalized(xy_d, self.dist, iters)

    def to_pixels(self, xy_d: np.ndarray) -> np.ndarray:
        u = self.fx * xy_d[:, 0] + self.skew * xy_d[:, 1] + self.cx
        v = self.fy * xy_d[:, 1] + self.cy
        return np.stack([u, v], axis=1)

    def from_pixels(self, uv: np.ndarray) -> np.ndarray:
        y = (uv[:, 1] - self.cy) / self.fy
        x = (uv[:, 0] - self.cx - self.skew * y) / self.fx
        return np.stack([x, y], axis=1)

    def project(self, X: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        Xc = np.asarray(X, float) @ rodrigues(rvec).T + np.asarray(tvec, float).reshape(1, 3)
        xy = Xc[:, :2] / Xc[:, 2:3]
        return self.to_pixels(self.distort(xy))

    def undistort_pixels(self, uv: np.ndarray) -> np.ndarray:
        """Distorted pixels -> where an ideal (distortion-free) camera with this K would see them."""
        return self.to_pixels(self.undistort(self.from_pixels(uv)))


def distort_normalized(xy: np.ndarray, dist: np.ndarray) -> np.ndarray:
    k1, k2, p1, p2 = dist
    x, y = xy[:, 0], xy[:, 1]
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 * r2
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([xd, yd], axis=1)


def undistort_normalized(xy_d: np.ndarray, dist: np.ndarray, iters: int = 50) -> np.ndarray:
    """Invert distort_normalized by fixed-point iteration (the scheme OpenCV uses)."""
    k1, k2, p1, p2 = dist
    xy_d = np.asarray(xy_d, float)
    x, y = xy_d[:, 0].copy(), xy_d[:, 1].copy()
    for _ in range(iters):
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2 * r2
        dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x = (xy_d[:, 0] - dx) / radial
        y = (xy_d[:, 1] - dy) / radial
    return np.stack([x, y], axis=1)
