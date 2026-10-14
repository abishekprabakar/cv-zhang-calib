"""End-to-end demo: render 15 checkerboard views through a distorted lens, detect
corners with OpenCV, calibrate with the from-scratch Zhang implementation, and
compare against ground truth and against cv2.calibrateCamera.

    python3 demo.py            # writes figures to output/
"""
from __future__ import annotations

import os

import numpy as np

from calib import (Board, Renderer, GROUND_TRUTH, IMAGE_SIZE, calibrate, detect_corners,
                   random_poses, undistort_image)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def draw_corners(gray, pts, colour):
    import cv2
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for k, (u, v) in enumerate(pts):
        cv2.circle(img, (int(round(u)), int(round(v))), 3, colour, -1, cv2.LINE_AA)
    cv2.circle(img, (int(round(pts[0][0])), int(round(pts[0][1]))), 7, (0, 0, 255), 2, cv2.LINE_AA)
    return img


def main(n_views: int = 15, seed: int = 7):
    import cv2
    os.makedirs(OUT, exist_ok=True)
    board, cam = Board(), GROUND_TRUTH
    poses = random_poses(board, cam, n_views, seed=seed)
    renderer = Renderer(cam)
    X = board.object_points()

    images, obs, det_err = [], [], []
    for k, (r, t) in enumerate(poses):
        g = renderer.render(board, r, t, seed=k)
        c = detect_corners(g, board)
        if c is None:
            print(f"view {k}: board not found, skipped")
            continue
        gt = cam.project(X, r, t)
        det_err.append(np.linalg.norm(c[:, None] - gt[None], axis=2).min(axis=1))
        images.append(g)
        obs.append(c)
    det_err = np.concatenate(det_err)
    print(f"Rendered {n_views} views at {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}, board found in {len(obs)}")
    print(f"Corner detection error vs ground truth: RMS {np.sqrt(np.mean(det_err**2)):.4f} px, "
          f"max {det_err.max():.4f} px")

    res = calibrate([X] * len(obs), obs)
    K0 = res.K_closed_form
    print("\nClosed-form (Zhang, distortion ignored):")
    print(f"  fx={K0[0,0]:.3f} fy={K0[1,1]:.3f} skew={K0[0,1]:.3f} cx={K0[0,2]:.3f} cy={K0[1,2]:.3f}")
    print("\nAfter Levenberg-Marquardt (this repo):")
    print(res.summary())

    gt_vals = [cam.fx, cam.fy, cam.cx, cam.cy, *cam.dist]
    est = res.camera
    est_vals = [est.fx, est.fy, est.cx, est.cy, *est.dist]
    names = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"]

    flags = cv2.CALIB_FIX_K3
    objp = [X.astype(np.float32)] * len(obs)
    imgp = [o.astype(np.float32).reshape(-1, 1, 2) for o in obs]
    cv_rms, cvK, cvd, _, _ = cv2.calibrateCamera(objp, imgp, IMAGE_SIZE, None, None, flags=flags)
    cv_vals = [cvK[0, 0], cvK[1, 1], cvK[0, 2], cvK[1, 2], *cvd.ravel()[:4]]

    print("\n  param   ground truth        this repo    cv2.calibrateCamera   |err|/sigma")
    for n, g, e, c, s in zip(names, gt_vals, est_vals, cv_vals, res.std_intrinsics):
        print(f"  {n:>5} {g:14.6f} {e:16.6f} {c:16.6f} {abs(e-g)/s:14.2f}")
    print(f"  RMS (px)              {res.rms:16.6f} {cv_rms:16.6f}")

    # Pose accuracy: detected corner order may be a 180-degree relabelling of the
    # board, so compare camera centres (invariant to that relabelling) instead of R.
    from calib import rodrigues
    centre_err = []
    for (r, t), (re, te) in zip(poses, zip(res.rvecs, res.tvecs)):
        c_true = rodrigues(r) @ board.centre + t
        corners_est = (X @ rodrigues(re).T + te).mean(axis=0)
        centre_err.append(np.linalg.norm(c_true - corners_est))
    print(f"\nBoard-centre position error across views: mean {np.mean(centre_err):.3f} mm, "
          f"max {np.max(centre_err):.3f} mm")

    # ---- figures
    tiles = [draw_corners(images[k], obs[k], (40, 200, 40)) for k in range(min(6, len(images)))]
    small = [cv2.resize(t, (320, 240), interpolation=cv2.INTER_AREA) for t in tiles]
    rows = [np.hstack(small[i:i + 3]) for i in range(0, len(small), 3)]
    cv2.imwrite(os.path.join(OUT, "views.png"), np.vstack(rows))

    und = cv2.cvtColor(undistort_image(images[0], est), cv2.COLOR_GRAY2BGR)
    # straight lines through the outer corner rows/columns: they hug the board edges only
    # once distortion has been removed through the outermost corner rows in the undistorted image
    ideal = est.undistort_pixels(obs[0])
    grid = ideal.reshape(board.rows, board.cols, 2)
    for line in (grid[0], grid[-1], grid[:, 0], grid[:, -1]):
        a, b = line[0], line[-1]
        d = (b - a) / np.linalg.norm(b - a)
        p0, p1 = a - 400 * d, b + 400 * d
        cv2.line(und, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (0, 0, 255), 1, cv2.LINE_AA)
    pair = np.hstack([cv2.cvtColor(images[0], cv2.COLOR_GRAY2BGR), und])
    cv2.putText(pair, "rendered (distorted)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
    cv2.putText(pair, "undistorted with estimate", (650, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
    cv2.imwrite(os.path.join(OUT, "undistort.png"), pair)

    # residual quiver, magnified 300x
    canvas = np.full((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), 255, np.uint8)
    for (r, t), o in zip(zip(res.rvecs, res.tvecs), obs):
        pred = est.project(X, r, t)
        for p, q in zip(o, pred):
            tip = p + 100 * (q - p)
            cv2.arrowedLine(canvas, (int(p[0]), int(p[1])), (int(tip[0]), int(tip[1])),
                            (180, 60, 20), 1, cv2.LINE_AA, tipLength=0.25)
    cv2.putText(canvas, f"reprojection residuals x100  (RMS {res.rms:.3f} px)", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(OUT, "residuals.png"), canvas)

    # convergence with number of views
    lines = ["views,fx_err_pct,fy_err_pct,cx_err_px,cy_err_px,k1_err,rms_px"]
    for n in range(3, len(obs) + 1):
        r_n = calibrate([X] * n, obs[:n])
        c = r_n.camera
        lines.append(f"{n},{100*abs(c.fx-cam.fx)/cam.fx:.4f},{100*abs(c.fy-cam.fy)/cam.fy:.4f},"
                     f"{abs(c.cx-cam.cx):.4f},{abs(c.cy-cam.cy):.4f},{abs(c.dist[0]-cam.dist[0]):.5f},{r_n.rms:.4f}")
    with open(os.path.join(OUT, "views_sweep.csv"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\nAccuracy vs number of views (output/views_sweep.csv):")
    for ln in lines[:1] + lines[1::3]:
        print("  " + ln)
    print("\nWrote output/views.png, output/undistort.png, output/residuals.png, output/views_sweep.csv")
    return res


if __name__ == "__main__":
    main()
