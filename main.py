#!/usr/bin/env python3

from math import sqrt
from absolute_positioning_ik import orient_horizontal, orient_vertical, set_absolute_orientation, set_absolute_pose, set_absolute_position
from get_camera_location import get_camera_position
from opencv_watershed_discrete import get_egg_location, detect_egg_once
import tf2_ros
import rospy

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

def main():
    # go to startingn position
    #gripper = intera_interface.Gripper(limb + '_gripper')
    gripper_x = 0.432
    #gripper.open()
    gripper_y = -0.147
    set_absolute_pose("right_hand", gripper_x, gripper_y, 0.022, 0.549, 0.404, -0.525, 0.510, speed_ratio=0.1)
    

    offsets = [(0.0, 0.0), (+0.005, 0.0), (0.0, +0.005), (-0.005, 0.0), (0.0, -0.005)]

    moved_by = (1.0, 1.0)
    while not moved_by == (0.0, 0.0):
        best_dist = float("inf")
        best_dx, best_dy = 0.0, 0.0

        for j, (dx, dy) in enumerate(offsets):
            # try candidate position (do NOT mutate gripper_x/y yet)
            x_try = gripper_x + dx
            y_try = gripper_y + dy
            set_absolute_pose("right_hand", x_try, y_try, 0.022, 0.549, 0.404, -0.525, 0.510, speed_ratio=0.1)

            t, r = func()
            egg = detect_egg_once()
            if not egg.ok:
                egg = detect_egg_once()
                if not egg.ok:
                    rospy.logerr("Egg not found!")
                    return

            dist = sqrt((egg.center_uv[0] - (egg.img_size[1] / 2))**2 +
                        (egg.center_uv[1] - (egg.img_size[0] / 2))**2)

            if dist < best_dist:
                best_dist = dist
                best_dx, best_dy = dx, dy

            rospy.loginfo("Candidate %d: distance %.1f", j, dist)

        # commit only the best move
        gripper_x += best_dx
        gripper_y += best_dy
        moved_by = (best_dx, best_dy)
        rospy.loginfo("Best distance %.1f, moving by (%.3f, %.3f)", best_dist, best_dx, best_dy)
        set_absolute_pose("right_hand", gripper_x, gripper_y, 0.022, 0.549, 0.404, -0.525, 0.510, speed_ratio=0.1)


    return
    for i in range(4):
        min_distance = 1e6
        for j in range(4):
            if j == 0:
                gripper_x += 0.01
            elif j == 1:
                gripper_y += 0.01
            elif j == 2:
                gripper_x -= 0.02
            elif j == 3:
                gripper_y -= 0.02
            
            set_absolute_pose("right_hand", gripper_x, gripper_y, 0.022, 0.549, 0.404, -0.525, 0.510, speed_ratio=0.1)

            # get camera position and orientation
            t, r = func()
            
            # get egg location from camera
            egg = detect_egg_once()
            if not egg.ok:
                egg = detect_egg_once()
                if not egg.ok:
                    rospy.logerr("Egg not found!")
                    return
            

            # distance from center of image
            print("image resolution: ", egg.img_size)
            print("egg center uv: ", egg.center_uv)
            dist = sqrt((egg.center_uv[0] - (egg.img_size[1] / 2))**2 + (egg.center_uv[1] - (egg.img_size[0] / 2))**2)
            if dist > min_distance:
                min_distance = dist
            rospy.loginfo("Distance from image center: %.1f pixels", dist)

        # move gripper in four directions, compare egg location changes     
        print("Minimum distance from center: ", min_distance, " with i = ", j)
        if j == 0:
            gripper_x += 0.02
            gripper_y += 0.01
        elif j == 1:
            gripper_x += 0.02
            gripper_y += 0.02
        elif j == 2:
            gripper_x += 0.00
            gripper_y += 0.02
        elif j == 3:
            gripper_x += 0.00
            gripper_y += 0.00

        print("Moving to new position: ", gripper_x, ", ", gripper_y)
        set_absolute_pose("right_hand", gripper_x, gripper_y, 0.022, 0.549, 0.404, -0.525, 0.510, speed_ratio=0.1)
        
    # detect final egg orientation
    

    # move gripper to last camera position


    # orient gripper to egg orientation


    # pick up egg
    #gripper.close()

if __name__ == "__main__":
    main()
