#!/usr/bin/env python3

from move_camera import CameraMover
import rospy
import numpy as np
from math import sqrt
import tf2_ros
from math import sqrt
import math
from move_camera import CameraMover
from opencv_watershed_discrete import detect_egg_once
import tf2_ros
import rospy
import intera_interface
import intera_external_devices
from absolute_positioning_ik import set_absolute_pose
from get_camera_location import get_camera_position
from intera_interface import Limb


egg_basket_1 = {
        "head_pan": 0.4700849609375, 
        "right_j0": -0.7110517578125, 
        "right_j1": 1.2322021484375, 
        "right_j2": -0.2424951171875, 
        "right_j3": -1.511798828125, 
        "right_j4": -1.4862236328125, 
        "right_j5": -1.6674609375, 
        "right_j6": 3.6852060546875, 
        "torso_t0": 0.0
    }


egg_basket_2 = {
    "head_pan": 0.4700849609375, 
    "right_j0": -0.7273173828125, 
    "right_j1": 1.1624365234375, 
    "right_j2": -0.462666015625, 
    "right_j3": -1.38658984375, 
    "right_j4": -1.442958984375, 
    "right_j5": -1.527609375, 
    "right_j6": 3.4133271484375, 
    "torso_t0": 0.0
}


def quat_multiply(q1, q2):
    """Hamilton product q = q1 * q2 (w last)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
        w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
        w1*z2 + x1*y2 - y1*x2 + z1*w2,  # z
        w1*w2 - x1*x2 - y1*y2 - z1*z2   # w
    )

def quat_about_z(theta_rad):
    """Quaternion for rotation about +Z by theta (right-hand rule)."""
    s = math.sin(theta_rad/2.0)
    c = math.cos(theta_rad/2.0)
    return (0.0, 0.0, s, c)

def move_limb_to(limb, pos):
    """
    Moves the limb (limb) and into the position of joint angles (pos)
    - uses global speed for always getting the same speed on the movements
    - timeout is always 10secs => when motion takes longer than that we return
    """
    limb.set_joint_position_speed(0.1)
    limb.move_to_joint_positions(pos,timeout=60)


def func():
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    rate = rospy.Rate(10.0)

    worked = False
    while not worked:
        try:
            trans = tfBuffer.lookup_transform('base', 'right_hand_camera', rospy.Time(0), rospy.Duration(1.0))
            t = trans.transform.translation
            r = trans.transform.rotation


            #rospy.loginfo("Position: (%.3f, %.3f, %.3f)", t.x, t.y, t.z)
            #rospy.loginfo("Orientation: (%.3f, %.3f, %.3f, %.3f)", r.x, r.y, r.z, r.w)
            worked = True
            return t, r
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn("Transform not available yet")
            rate.sleep()


def ring_offsets(n, radius):
    """n points evenly spaced on a ring, as (dx, dy)."""
    return [(radius*math.cos(2*math.pi*i/n),
             radius*math.sin(2*math.pi*i/n)) for i in range(n)]

def main():
    rospy.init_node("egg_picker_main")

    gripper = intera_interface.Gripper('right_gripper')
    gripper.open()

    x = 0.556
    y = 0.356
    z = 0.03
    #set_absolute_pose("right_hand", x, y, 0.022, 0.549, 0.404, -0.525, 0.510, speed_ratio=0.1)
    

    offsets_coarse = [(0.0, 0.0)] + ring_offsets(4, 0.03)
    offsets_mid    = [(0.0, 0.0)] + ring_offsets(5, 0.01)
    offsets_fine   = [(0.0, 0.0)] + ring_offsets(9, 0.005)
    
    
    mover = CameraMover(speed_ratio=0.12)    
    for i in range(4):
        moved_by = (1.0, 1.0)
        while not moved_by == (0.0, 0.0):
            best_dist = float("inf")
            best_dx, best_dy = 0.0, 0.0

            if i == 0:
                offsets = offsets_coarse
            elif i == 1 or i == 2:
                offsets = offsets_mid
            else:
                offsets = offsets_fine

            for j, (dx, dy) in enumerate(offsets):
                # try candidate position (do NOT mutate x/y yet)
                x_try = x + dx
                y_try = y + dy
                mover.moveCameraHorizontally(x_try, y_try, z)
                rospy.sleep(1)

                t, r = func()
                egg = detect_egg_once()
                egg = detect_egg_once()
                if egg.ok:

                    dist = sqrt((egg.center_uv[0] - (egg.img_size[1] / 2))**2 +
                                (egg.center_uv[1] - (egg.img_size[0] / 2))**2)

                    if dist < best_dist:
                        best_dist = dist
                        best_dx, best_dy = dx, dy

                    rospy.loginfo("Candidate %d: distance %.1f", j, dist)

            # commit only the best move
            x += best_dx
            y += best_dy
            moved_by = (best_dx, best_dy)
            rospy.loginfo("Best distance %.1f, moving by (%.3f, %.3f)", best_dist, best_dx, best_dy)
            mover.moveCameraHorizontally(x, y, z)
            if best_dist < 70:
                rospy.loginfo("best_dist smaller than 70px, moving down immediately")
                break
        rospy.loginfo("moving down")
        z -= 0.04


    z += 0.2
    mover.moveCameraHorizontally(x, y, z)
    egg = detect_egg_once()
    egg_angle = egg.angle_deg
    t, r = get_camera_position()


    z+= 0.1
    set_absolute_pose("right_hand", x, y, z, 1, 0, 0, 0, speed_ratio=0.12)
    z -= 0.1
    set_absolute_pose("right_hand", x, y, z, 1, 0, 0, 0, speed_ratio=0.15)
    rospy.loginfo("Camera Orientation: (%.3f, %.3f, %.3f, %.3f)", r.x, r.y, r.z, r.w)


    # align the gripper with the egg
    ANGLE_SIGN = +1.0  # flip to -1.0 if rotation goes the wrong way in your setup
    # get the most recent camera (end-effector) orientation in base
    q_current = (r.x, r.y, r.z, r.w)

    # spin about tool Z by egg_angle (deg → rad)
    theta = ANGLE_SIGN * math.radians(egg_angle)
    q_spin = quat_about_z(theta)

    # local-axis spin: post-multiply current orientation by the spin
    q_target = quat_multiply(q_current, q_spin)

    # command the pose with the same (x, y, z) you’ve positioned at, but rotated
    set_absolute_pose("right_hand", x, y, z, q_target[0], q_target[1], q_target[2], q_target[3], speed_ratio=0.15)


    # move down to final height for pickup
    z -= 0.075
    set_absolute_pose("right_hand", x, y, z, q_target[0], q_target[1], q_target[2], q_target[3], speed_ratio=0.08)

    rospy.sleep(2)
    # pick up egg
    gripper.close()

    z += 0.5
    set_absolute_pose("right_hand", x, y, z, q_target[0], q_target[1], q_target[2], q_target[3], speed_ratio=0.08)



    limb = Limb()
    limb.set_joint_position_speed(0.1)
    limb.move_to_joint_positions(egg_basket_2, timeout=10.0)
    rospy.sleep(5)
    gripper.open()


    main()

if __name__ == "__main__":
    main()
