#!/usr/bin/env python3
from move_camera import CameraMover
import rospy
import numpy as np
from math import sqrt
import tf2_ros

from absolute_positioning_ik import set_absolute_pose
from opencv_watershed_discrete import detect_egg_once

# --- Helpers --------------------------------------------------------------
import numpy as np
import rospy
from math import hypot

def get_uv_or_throw():
    egg = detect_egg_once()
    if not egg.ok:
        egg = detect_egg_once()
    if not egg.ok:
        raise RuntimeError("Egg not found")
    h, w = egg.img_size   # (rows, cols)
    u, v = egg.center_uv  # (x_pix, y_pix)
    return (u, v, w/2.0, h/2.0)

def pixel_error():
    u, v, u0, v0 = get_uv_or_throw()
    e = np.array([[u - u0], [v - v0]], dtype=float)
    return e, float(np.linalg.norm(e))

def servo_step(x, y, z,
               delta=0.010,      # 10 mm probes for cleaner derivatives
               lam=0.7,          # gain
               damp=8.0,         # DLS damping (pixels)
               max_move=0.02):   # clamp update per iter
    # Baseline at nominal
    mover = CameraMover(speed_ratio=0.08)
    mover.moveCameraHorizontally(x, y, z)
    rospy.sleep(0.5)
    e0, err0 = pixel_error()
    if err0 < 2.0:
        return x, y, err0, (0.0, 0.0)

    # --- Central finite differences ---------------------------------------
    # x- probe
    mover.moveCameraHorizontally(x - delta, y, z)
    rospy.sleep(0.5)
    ux_m, vx_m, _, _ = get_uv_or_throw()

    # x+ probe
    mover.moveCameraHorizontally(x + delta, y, z)
    rospy.sleep(0.5)
    ux_p, vx_p, _, _ = get_uv_or_throw()

    du_dx = (ux_p - ux_m) / (2.0 * delta)
    dv_dx = (vx_p - vx_m) / (2.0 * delta)

    # back to nominal before y probes
    mover.moveCameraHorizontally(x, y, z)
    rospy.sleep(0.5)

    # y- probe
    mover.moveCameraHorizontally(x, y - delta, z)
    rospy.sleep(0.5)
    uy_m, vy_m, _, _ = get_uv_or_throw()

    # y+ probe
    mover.moveCameraHorizontally(x, y + delta, z)
    rospy.sleep(0.5)
    uy_p, vy_p, _, _ = get_uv_or_throw()

    du_dy = (uy_p - uy_m) / (2.0 * delta)
    dv_dy = (vy_p - vy_m) / (2.0 * delta)

    # restore nominal for update
    mover.moveCameraHorizontally(x, y, z)
    rospy.sleep(0.5)

    J = np.array([[du_dx, du_dy],
                  [dv_dx, dv_dy]], dtype=float)    # [pixels / meter]
    JT = J.T
    H = JT @ J + (damp**2) * np.eye(2)

    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        rospy.logwarn("Jacobian ill-conditioned; increasing damping.")
        H_inv = np.linalg.inv(JT @ J + (20*damp)**2 * np.eye(2))

    # Δx = -λ (JᵀJ + μ²I)^-1 Jᵀ e
    dx_dy = -lam * (H_inv @ JT @ e0)   # meters

    dx = float(dx_dy[0, 0]); dy = float(dx_dy[1, 0])
    # clamp
    step_norm = hypot(dx, dy)
    if step_norm > max_move and step_norm > 1e-9:
        s = max_move / step_norm
        dx *= s; dy *= s

    # Apply only if it improves the pixel error (cheap line search)
    x_try, y_try = x + dx, y + dy
    mover.moveCameraHorizontally(x_try, y_try, z)
    rospy.sleep(0.5)
    _, err_try = pixel_error()

    if err_try < err0:
        return x_try, y_try, err_try, (dx, dy)
    else:
        # halve the step and try once more
        dx *= 0.5; dy *= 0.5
        x_try2, y_try2 = x + dx, y + dy
        mover.moveCameraHorizontally(x_try2, y_try2, z)
        rospy.sleep(0.5)

        _, err_try2 = pixel_error()
        if err_try2 < err0:
            return x_try2, y_try2, err_try2, (dx, dy)
        # no improvement: stay and report
        mover.moveCameraHorizontally(x + delta, y, z)
        rospy.sleep(0.5)
        return x, y, err0, (0.0, 0.0)

# --- Main ----------------------------------------------------------------

def main():
    rospy.init_node("egg_ibvs_servo")
    
    # Fixed orientation & height from your sample pose
    # set_absolute_pose args: (name, x, y, z, qx, qy, qz, qw, ...)
    zyxw = (0.14, -0.049, 0.719, -0.687, -0.089)
    z = 0.05

    # Start near your initial guess
    x = 0.44
    y = -0.161
    set_absolute_pose("right_hand", x, y, z, 1, 0, 0, 0, speed_ratio=0.1)

    # Servo loop
    max_iterations = 20
    delta = 0.01       # 5 mm finite-diff probe
    lam = 0.6           # start a bit aggressive
    damp = 6.0          # pixels
    tol_px = 2.0

    for k in range(max_iterations):
        x, y, err_px, (dx, dy) = servo_step(
            x, y, z,
            delta=delta, lam=lam, damp=damp, max_move=0.015
        )
        rospy.loginfo("Iter %d: |e|=%.1f px, move=(%.3f, %.3f) m @ (x=%.3f,y=%.3f)",
                      k, err_px, dx, dy, x, y)

        # simple step adaptation
        if err_px < tol_px:
            rospy.loginfo("Centered (<= %.1f px).", tol_px)
            break
        # If last step made little progress, increase damping or shrink gain
        if sqrt(dx*dx + dy*dy) < 1e-4:  # <0.1 mm
            damp *= 1.5
            lam *= 0.7
        # If error got worse, shrink gain
        if k > 0 and err_px > 200.0:
            lam *= 0.7

        # Optional: shrink probe as we get closer
        delta = max(0.0015, 0.5 * delta) if err_px < 20 else delta

    # You can now align orientation to the egg angle and grasp.

if __name__ == "__main__":
    main()
