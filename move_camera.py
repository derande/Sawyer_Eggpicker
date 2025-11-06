#!/usr/bin/env python3
"""
Move Sawyer's right-hand camera to (x, y, z) in base frame
while keeping it perpendicular to the ground (optical axis down).

- Tries to solve with the camera link as the IK tip if available in URDF.
- Otherwise converts the desired camera pose to an equivalent hand pose via TF.
- Sweeps heading (yaw about down axis) to avoid over-constraining orientation.
- Reuses the last valid solution as the next seed for repeatability.

ROS: Noetic
Deps: trac_ik_python, intera_interface
"""

import rospy
import numpy as np
import tf2_ros, tf_conversions
from trac_ik_python.trac_ik import IK
from intera_interface import Limb

BASE = "base"
CAM  = "right_hand_camera"
HAND = "right_hand"


# -------------------- math helpers --------------------
def _normalize(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v

def _basis_from_z_and_x(z_des, x_hint):
    """Build rotation with +Z = z_des and +X as close as possible to x_hint."""
    z = _normalize(z_des)
    x_tilde = x_hint - np.dot(x_hint, z) * z
    if np.linalg.norm(x_tilde) < 1e-8:
        x_tilde = np.array([1, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1, 0])
    x = _normalize(x_tilde)
    y = np.cross(z, x)
    R = np.eye(3)
    R[:, 0], R[:, 1], R[:, 2] = x, y, z
    return R

def _quat_from_R(R):
    M = np.eye(4); M[:3, :3] = R
    return tf_conversions.transformations.quaternion_from_matrix(M)  # x,y,z,w

def _lookup_T(tfbuf, parent, child):
    t = tfbuf.lookup_transform(parent, child, rospy.Time(0), rospy.Duration(2.0))
    T = tf_conversions.transformations.quaternion_matrix(
        [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]
    )
    T[0, 3] = t.transform.translation.x
    T[1, 3] = t.transform.translation.y
    T[2, 3] = t.transform.translation.z
    return T

def _seed_from_current(joint_names):
    limb = Limb("right")
    cur = limb.joint_ordered_positions()
    return [cur[n] for n in joint_names]

def _seed_mid_limits(ik):
    lo, hi = ik.get_joint_limits()
    return [(a + b) / 2.0 for a, b in zip(lo, hi)]


class CameraMover:
    def __init__(self,
                 base_frame=BASE,
                 cam_frame=CAM,
                 hand_frame=HAND,
                 solve_type="Distance",
                 speed_ratio=0.15,
                 ik_timeout=0.6):
        rospy.loginfo("Initializing CameraMover…")
        self.base = base_frame
        self.cam = cam_frame
        self.hand = hand_frame
        self.ik_timeout = ik_timeout

        # TF
        self.tfbuf = tf2_ros.Buffer()
        self.tfl = tf2_ros.TransformListener(self.tfbuf)

        # URDF
        self.urdf = rospy.get_param("/robot_description")

        # Try TRAC-IK with camera tip first
        self.use_camera_tip = False
        try:
            ik_cam = IK(self.base, self.cam, urdf_string=self.urdf, solve_type=solve_type)
            if len(ik_cam.joint_names) > 0:
                self.ik = ik_cam
                self.jnames = ik_cam.joint_names
                self.use_camera_tip = True
                self.cam_T_hand = None
        except Exception:
            pass

        if not self.use_camera_tip:
            rospy.logwarn("URDF does not include '%s' as a kinematic tip; solving for '%s' and converting poses.",
                          self.cam, self.hand)
            self.ik = IK(self.base, self.hand, urdf_string=self.urdf, solve_type=solve_type)
            self.jnames = self.ik.joint_names
            hand_T_cam = _lookup_T(self.tfbuf, self.hand, self.cam)
            self.cam_T_hand = np.linalg.inv(hand_T_cam)

        # Limb
        self.limb = Limb("right")
        self.limb.set_joint_position_speed(speed_ratio)

        # Initial seed
        try:
            self.seed = _seed_from_current(self.jnames)
        except Exception:
            self.seed = _seed_mid_limits(self.ik)

        rospy.loginfo("CameraMover ready. Tip link: %s", self.cam if self.use_camera_tip else self.hand)

    def _solve_once(self, px, py, pz, qx, qy, qz, qw):
        """Single IK call with current seed; returns solution or None."""
        return self.ik.get_ik(self.seed, px, py, pz, qx, qy, qz, qw)

    def moveCameraHorizontally(self, x, y, z, heading_deg=None):
        """
        Move the camera to (x, y, z) in base frame, keeping optical axis down.
        If heading_deg is None, sweep multiple headings to find a feasible wrist posture.
        Returns True/False.
        """
        z_des = np.array([0, 0, -1.0])  # camera +Z down

        # Build heading candidates (yaw about base Z)
        if heading_deg is not None:
            headings = [heading_deg]
        else:
            headings = [0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180]

        solution = None
        used_heading = None
        last_err = None

        for deg in headings:
            yaw = np.deg2rad(deg)
            x_hint = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            R_cam = _basis_from_z_and_x(z_des, x_hint)

            # If your camera optical axis is +X (not +Z), uncomment to rotate frame:
            # R_fix = tf_conversions.transformations.rotation_matrix(np.pi/2, (0,1,0))[:3,:3]
            # R_cam = R_cam.dot(R_fix)

            qx, qy, qz, qw = _quat_from_R(R_cam)

            # If we can't solve for camera tip, convert desired camera pose -> hand pose
            px, py, pz, qq = x, y, z, (qx, qy, qz, qw)
            if self.cam_T_hand is not None:
                base_T_cam = np.eye(4); base_T_cam[:3, :3] = R_cam; base_T_cam[:3, 3] = [x, y, z]
                base_T_hand = base_T_cam.dot(self.cam_T_hand)
                px, py, pz = base_T_hand[0, 3], base_T_hand[1, 3], base_T_hand[2, 3]
                qh = tf_conversions.transformations.quaternion_from_matrix(base_T_hand)
                qq = (qh[0], qh[1], qh[2], qh[3])

            sol = self._solve_once(px, py, pz, *qq)
            if sol is not None:
                solution = sol
                used_heading = deg
                break
            last_err = "heading {}° failed".format(deg)

        if solution is None:
            rospy.logerr("No IK solution for camera at (%.3f, %.3f, %.3f). %s", x, y, z, last_err or "")
            rospy.logerr("Try: raise/lower z, move closer to base, or pass a specific heading_deg that works.")
            return False

        # Command and update seed
        cmd = dict(zip(self.jnames, solution))
        self.limb.move_to_joint_positions(cmd, timeout=30.0)
        self.seed = list(solution)  # reuse for next call

        rospy.loginfo("Camera moved to (%.3f, %.3f, %.3f) with heading %s°.",
                      x, y, z, str(used_heading) if used_heading is not None else "fixed")
        return True


# -------------------- script entry --------------------
def main():
    rospy.init_node("sawyer_move_camera")
    mover = CameraMover()

    # Example: three moves; the seed is reused so motion is consistent
    goals = [(0.50, 0.10, 0.25), (0.58, 0.00, 0.25), (0.52, 0.18, 0.25)]
    for (x, y, z) in goals:
        if not mover.moveCameraHorizontally(x, y, z):
            break
        rospy.sleep(0.5)

if __name__ == "__main__":
    main()
