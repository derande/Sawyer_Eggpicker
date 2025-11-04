#! /usr/bin/env python
import rospy
import argparse
from intera_motion_interface import MotionTrajectory, MotionWaypoint, MotionWaypointOptions
from intera_motion_msgs.msg import TrajectoryOptions
from geometry_msgs.msg import PoseStamped
from intera_interface import Limb
import math


def set_absolute_position(tip_name, x, y, z, timeout=None, lin_speed=0.6, lin_acc=0.6, rot_speed=1.57, rot_acc=1.57):
    rospy.init_node("go_to_cartesian_pose_py", anonymous=True)
    limb = Limb()

    traj_options = TrajectoryOptions()
    traj_options.interpolation_type = TrajectoryOptions.CARTESIAN
    traj = MotionTrajectory(trajectory_options=traj_options, limb=limb)

    wpt_opts = MotionWaypointOptions(
        max_linear_speed=lin_speed,
        max_linear_accel=lin_acc,
        max_rotational_speed=rot_speed,
        max_rotational_accel=rot_acc,
        max_joint_speed_ratio=1.0
    )
    waypoint = MotionWaypoint(options=wpt_opts.to_msg(), limb=limb)

    endpoint_state = limb.tip_state(tip_name)
    if endpoint_state is None:
        rospy.logerr("Endpoint state not found for tip '%s'", tip_name)
        return

    pose = endpoint_state.pose
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z

    pose_st = PoseStamped()
    pose_st.header.stamp = rospy.Time.now()
    pose_st.header.frame_id = "base"
    pose_st.pose = pose

    # Nullraum-Bias = aktuelle Gelenkwinkel
    joint_angles = limb.joint_ordered_angles()
    waypoint.set_cartesian_pose(pose_st, tip_name, joint_angles)

    rospy.loginfo("Sending waypoint:\n%s", waypoint.to_string())
    traj.append_waypoint(waypoint.to_msg())

    result = traj.send_trajectory(timeout=timeout)
    if result is None:
        rospy.logerr("Trajectory FAILED to send")
        return
    if result.result:
        rospy.loginfo("Motion controller successfully finished the trajectory!")
    else:
        rospy.logerr("Motion controller failed with error %s", result.errorId)

def set_absolute_orientation(tip_name, qx, qy, qz, qw, timeout=None, lin_speed=0.6, lin_acc=0.6, rot_speed=1.57, rot_acc=1.57):
    rospy.init_node("go_to_cartesian_pose_py", anonymous=True)
    limb = Limb()

    traj_options = TrajectoryOptions()
    traj_options.interpolation_type = TrajectoryOptions.CARTESIAN
    traj = MotionTrajectory(trajectory_options=traj_options, limb=limb)

    wpt_opts = MotionWaypointOptions(
        max_linear_speed=lin_speed,
        max_linear_accel=lin_acc,
        max_rotational_speed=rot_speed,
        max_rotational_accel=rot_acc,
        max_joint_speed_ratio=1.0
    )
    waypoint = MotionWaypoint(options=wpt_opts.to_msg(), limb=limb)

    endpoint_state = limb.tip_state(tip_name)
    if endpoint_state is None:
        rospy.logerr("Endpoint state not found for tip '%s'", tip_name)
        return

    pose = endpoint_state.pose
    pose.orientation.x = qx
    pose.orientation.y = qy
    pose.orientation.z = qz
    pose.orientation.w = qw

    pose_st = PoseStamped()
    pose_st.header.stamp = rospy.Time.now()
    pose_st.header.frame_id = "base"
    pose_st.pose = pose

    # Nullraum-Bias = aktuelle Gelenkwinkel
    joint_angles = limb.joint_ordered_angles()
    waypoint.set_cartesian_pose(pose_st, tip_name, joint_angles)

    rospy.loginfo("Sending waypoint:\n%s", waypoint.to_string())
    traj.append_waypoint(waypoint.to_msg())

    result = traj.send_trajectory(timeout=timeout)
    if result is None:
        rospy.logerr("Trajectory FAILED to send")
        return
    if result.result:
        rospy.loginfo("Motion controller successfully finished the trajectory!")
    else:
        rospy.logerr("Motion controller failed with error %s", result.errorId)
