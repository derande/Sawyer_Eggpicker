#! /usr/bin/env python
import rospy
import argparse
from intera_motion_interface import MotionTrajectory, MotionWaypoint, MotionWaypointOptions
from intera_motion_msgs.msg import TrajectoryOptions
from geometry_msgs.msg import PoseStamped
from intera_interface import Limb
import math

# --- IK imports (from Intera example) ---
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import Header
from sensor_msgs.msg import JointState
from intera_core_msgs.srv import SolvePositionIK, SolvePositionIKRequest


def solve_ik(pose_st, tip_name='right_hand', limb_name='right', seed_angles=None,
             use_nullspace_goal=False, nullspace_goal=None, nullspace_gain=0.4, wait_timeout=5.0):
    """
    Call Intera's IK service to get joint angles for a desired PoseStamped.
    Returns: dict(name->position) on success, or None on failure.
    """
    ns = "ExternalTools/{}/PositionKinematicsNode/IKService".format(limb_name)
    iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
    ikreq = SolvePositionIKRequest()

    ikreq.pose_stamp.append(pose_st)
    ikreq.tip_names.append(tip_name)

    # Seed: if provided, prefer user seed; otherwise rely on solver's current-angle seed.
    if seed_angles is not None:
        ikreq.seed_mode = ikreq.SEED_USER
        seed = JointState()
        seed.name = list(seed_angles.keys())
        seed.position = [seed_angles[n] for n in seed.name]
        ikreq.seed_angles.append(seed)

    # Optional nullspace goal
    if use_nullspace_goal and nullspace_goal:
        ikreq.use_nullspace_goal.append(True)
        goal = JointState()
        goal.name = list(nullspace_goal.keys())
        goal.position = [nullspace_goal[n] for n in goal.name]
        ikreq.nullspace_goal.append(goal)
        ikreq.nullspace_gain.append(float(nullspace_gain))

    try:
        rospy.wait_for_service(ns, wait_timeout)
        resp = iksvc(ikreq)
    except (rospy.ServiceException, rospy.ROSException) as e:
        rospy.logerr("IK service call failed: %s", e)
        return None

    if resp.result_type and resp.result_type[0] > 0 and resp.joints:
        joint_sol = dict(zip(resp.joints[0].name, resp.joints[0].position))
        return joint_sol
    else:
        rospy.logerr("INVALID POSE - No Valid Joint Solution Found. Result: %s", list(resp.result_type) if resp.result_type else None)
        return None


def _build_motion_traj(limb, lin_speed, lin_acc, rot_speed, rot_acc):
    traj_options = TrajectoryOptions()
    # We’re sending JOINT waypoints now; interpolation type can remain default (CUBIC)
    traj = MotionTrajectory(trajectory_options=traj_options, limb=limb)

    wpt_opts = MotionWaypointOptions(
        max_linear_speed=lin_speed,
        max_linear_accel=lin_acc,
        max_rotational_speed=rot_speed,
        max_rotational_accel=rot_acc,
        max_joint_speed_ratio=1.0
    )
    waypoint = MotionWaypoint(options=wpt_opts.to_msg(), limb=limb)
    return traj, waypoint


def set_absolute_position(tip_name, x, y, z, timeout=None, lin_speed=0.6, lin_acc=0.6, rot_speed=1.57, rot_acc=1.57):
    rospy.init_node("go_to_pose_with_ik_py", anonymous=True)
    limb = Limb()

    # Start from the current endpoint pose and update position
    endpoint_state = limb.tip_state(tip_name)
    if endpoint_state is None:
        rospy.logerr("Endpoint state not found for tip '%s'", tip_name)
        return

    pose = endpoint_state.pose
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = float(z)

    pose_st = PoseStamped()
    pose_st.header = Header(stamp=rospy.Time.now(), frame_id="base")
    pose_st.pose = pose

    # Use current joint angles as seed
    current = limb.joint_ordered_angles()
    seed_angles = dict(zip(limb.joint_names(), current))

    # --- Solve IK ---
    joint_sol = solve_ik(pose_st, tip_name=tip_name, limb_name=limb.name(), seed_angles=seed_angles)
    if joint_sol is None:
        rospy.logerr("IK failed for target position (%.3f, %.3f, %.3f)", x, y, z)
        return

    # --- Build and send JOINT waypoint ---
    traj, waypoint = _build_motion_traj(limb, lin_speed, lin_acc, rot_speed, rot_acc)
    if not waypoint.set_joint_angles(joint_sol):
        rospy.logerr("Failed to set joint angles on waypoint")
        return

    rospy.loginfo("Sending JOINT waypoint (from IK):\n%s", waypoint.to_string())
    traj.append_waypoint(waypoint.to_msg())
    result = traj.send_trajectory(timeout=timeout)

    if result is None:
        rospy.logerr("Trajectory FAILED to send")
    elif result.result:
        rospy.loginfo("Motion controller successfully finished the trajectory!")
    else:
        rospy.logerr("Motion controller failed with error %s", result.errorId)


def set_absolute_orientation(tip_name, qx, qy, qz, qw, timeout=None, lin_speed=0.6, lin_acc=0.6, rot_speed=1.57, rot_acc=1.57):
    rospy.init_node("go_to_pose_with_ik_py", anonymous=True)
    limb = Limb()

    endpoint_state = limb.tip_state(tip_name)
    if endpoint_state is None:
        rospy.logerr("Endpoint state not found for tip '%s'", tip_name)
        return

    pose = endpoint_state.pose
    pose.orientation = Quaternion(x=float(qx), y=float(qy), z=float(qz), w=float(qw))

    pose_st = PoseStamped()
    pose_st.header = Header(stamp=rospy.Time.now(), frame_id="base")
    pose_st.pose = pose

    # Use current joint angles as seed
    current = limb.joint_ordered_angles()
    seed_angles = dict(zip(limb.joint_names(), current))

    # --- Solve IK ---
    joint_sol = solve_ik(pose_st, tip_name=tip_name, limb_name=limb.name(), seed_angles=seed_angles)
    if joint_sol is None:
        rospy.logerr("IK failed for target orientation (%.3f, %.3f, %.3f, %.3f)", qx, qy, qz, qw)
        return

    # --- Build and send JOINT waypoint ---
    traj, waypoint = _build_motion_traj(limb, lin_speed, lin_acc, rot_speed, rot_acc)
    if not waypoint.set_joint_angles(joint_sol):
        rospy.logerr("Failed to set joint angles on waypoint")
        return

    rospy.loginfo("Sending JOINT waypoint (from IK):\n%s", waypoint.to_string())
    traj.append_waypoint(waypoint.to_msg())
    result = traj.send_trajectory(timeout=timeout)

    if result is None:
        rospy.logerr("Trajectory FAILED to send")
    elif result.result:
        rospy.loginfo("Motion controller successfully finished the trajectory!")
    else:
        rospy.logerr("Motion controller failed with error %s", result.errorId)


def orient_vertical(tip_name):
    # Example: rotate so Y axis aligns (matches your original intent)
    set_absolute_orientation(tip_name, 0.0, 1.0, 0.0, 0.0)

def orient_horizontal(tip_name):
    set_absolute_orientation(tip_name, 1.0, 0.0, 0.0, 0.0)
