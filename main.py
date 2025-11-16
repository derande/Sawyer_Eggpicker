#!/usr/bin/env python3

from gripper_helper import check_object_gripped
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


egg_basket_1 = [-0.344201171875, 0.199333984375, 0.266794921875, 0.13100390625, -2.075982421875, -1.985689453125, -3.4116142578125]
egg_basket_2 = [-0.479912109375, 0.8671083984375, -0.411232421875, -0.995740234375, -1.3752412109375, -1.9155185546875, -2.9495322265625]
egg_basket_3 = [-0.58746484375, 0.9589541015625, -0.411376953125, -1.1520986328125, -1.4284599609375, -1.7224873046875, -2.9493251953125]


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

def main(egg_carton_index = 0):
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
    
    
    mover = CameraMover(speed_ratio=0.06)    
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
                rospy.sleep(0.25)

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

                    
                    if i < 4:
                        if dist < 70:
                            rospy.loginfo("distance smaller than 70px, moving down immediately.")
                            i += 1
                            break
                    else:
                        if dist < 40:
                            rospy.loginfo("distance smaller than 50px, egg found.")
                            i+= 1
                            break
                            

            # commit only the best move
            x += best_dx
            y += best_dy
            moved_by = (best_dx, best_dy)
            rospy.loginfo("Best distance %.1f, moving by (%.3f, %.3f)", best_dist, best_dx, best_dy)
            mover.moveCameraHorizontally(x, y, z)
        rospy.loginfo("moving down")
        z -= 0.04


    z += 0.3
    mover.moveCameraHorizontally(x, y, z)
    egg = detect_egg_once()
    egg_angle = egg.angle_deg
    t, r = get_camera_position()


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

    z += 0.4
    set_absolute_pose("right_hand", x, y, z, q_target[0], q_target[1], q_target[2], q_target[3], speed_ratio=0.08)

    # check if egg was actually gripped, otherwise repeat search procedure
    gripped = check_object_gripped()
    if not gripped:
        main(egg_carton_index)
        return


    egg_basket = egg_basket_1
    if egg_carton_index == 1:
        egg_basket = egg_basket_2
    if egg_carton_index == 2:
        egg_basket = egg_basket_3


    limb = Limb()
    joint_names = limb.joint_names()
    limb.set_joint_position_speed(0.17)
    joints = dict(zip(joint_names, egg_basket))
    print(joints)
    limb.move_to_joint_positions(joints, timeout=15.0)

    rospy.sleep(5)
    gripper.open()


    main(egg_carton_index=egg_carton_index+1)

if __name__ == "__main__":
    main()
