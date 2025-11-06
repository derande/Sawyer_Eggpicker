#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import Header
from sensor_msgs.msg import JointState

from intera_core_msgs.srv import SolvePositionIK, SolvePositionIKRequest
from intera_interface import Limb


# ---------------------------
# Helpers
# ---------------------------

def _ensure_node(node_name="go_to_cartesian_pose_py"):
    """Initialize ROS node once."""
    if not rospy.core.is_initialized():
        rospy.init_node(node_name, anonymous=True)

def _current_pose_as_ps(limb, tip_name, frame_id="base"):
    """Read current tip pose and return as PoseStamped."""
    endpoint_state = limb.tip_state(tip_name)
    if endpoint_state is None:
        rospy.logerr("Endpoint state not found for tip '%s'", tip_name)
        return None
    ps = PoseStamped()
    ps.header = Header(stamp=rospy.Time.now(), frame_id=frame_id)
    ps.pose = endpoint_state.pose
    return ps

def _solve_ik(pose_stamped, tip_name='right_hand', limb_name='right',
              seed_joint_positions=None,
              nullspace_goal=None, nullspace_gain=0.4, timeout=5.0):
    """
    Call Intera IK service and return a dict(name->pos) or None on failure.

    - seed_joint_positions: dict of joint_name->pos used as IK seed
    - nullspace_goal: dict of joint_name->pos used as nullspace target
    """
    ns = "ExternalTools/{}/PositionKinematicsNode/IKService".format(limb_name)
    rospy.wait_for_service(ns, timeout=timeout)
    iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
    req = SolvePositionIKRequest()
    req.pose_stamp.append(pose_stamped)
    req.tip_names.append(tip_name)

    # Seed (optional)
    if seed_joint_positions:
        req.seed_mode = req.SEED_USER
        seed = JointState()
        seed.name = list(seed_joint_positions.keys())
        seed.position = [seed_joint_positions[n] for n in seed.name]
        req.seed_angles.append(seed)

    # Nullspace (optional)
    if nullspace_goal:
        req.use_nullspace_goal.append(True)
        goal = JointState()
        goal.name = list(nullspace_goal.keys())
        goal.position = [nullspace_goal[n] for n in goal.name]
        req.nullspace_goal.append(goal)
        req.nullspace_gain.append(nullspace_gain)

    try:
        resp = iksvc(req)
    except (rospy.ServiceException, rospy.ROSException) as e:
        rospy.logerr("IK service call failed: %s", e)
        return None

    if not resp.result_type or resp.result_type[0] <= 0:
        rospy.logerr("INVALID POSE - No valid IK solution. Result code: %s", 
                     str(resp.result_type[0] if resp.result_type else "None"))
        return None

    joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
    return joints


def _execute_joint_dict(joint_dict, speed_ratio=0.3, timeout=10.0):
    """
    Execute a joint position command using the Limb API.
    speed_ratio: 0.0..1.0 scales the internal max joint speeds.
    """
    limb = Limb()
    limb.set_joint_position_speed(speed_ratio)
    limb.move_to_joint_positions(joint_dict, timeout=timeout)


# ---------------------------
# Public API (compatible with your original calls)
# ---------------------------

def set_absolute_position(tip_name, x, y, z, timeout=10.0,
                          lin_speed=0.6, lin_acc=0.6,
                          rot_speed=1.57, rot_acc=1.57,
                          limb_name='right',
                          speed_ratio=0.3,
                          use_advanced_ik=False):
    """
    Move tip to absolute position (x,y,z) while keeping current orientation,
    using IK -> joint command.
    """
    _ensure_node()
    limb = Limb()

    # Build target pose from current but override position
    ps = _current_pose_as_ps(limb, tip_name)
    if ps is None:
        return
    ps.header.stamp = rospy.Time.now()
    ps.header.frame_id = 'base'
    ps.pose.position = Point(x=x, y=y, z=z)

    # Prepare IK options
    seed = limb.joint_ordered_positions() if use_advanced_ik else None
    ns_goal = None  # customize if you want a bias
    joints = _solve_ik(ps, tip_name=tip_name, limb_name=limb_name,
                       seed_joint_positions=seed,
                       nullspace_goal=ns_goal)

    if joints is None:
        return

    #rospy.loginfo("IK Joint Solution: %s", joints)
    _execute_joint_dict(joints, speed_ratio=speed_ratio, timeout=timeout)
    #rospy.loginfo("Move complete.")

def set_absolute_orientation(tip_name, qx, qy, qz, qw, timeout=10.0,
                             lin_speed=0.6, lin_acc=0.6,
                             rot_speed=1.57, rot_acc=1.57,
                             limb_name='right',
                             speed_ratio=0.3,
                             use_advanced_ik=False):
    """
    Rotate tip to absolute orientation (qx,qy,qz,qw) while keeping current position,
    using IK -> joint command.
    """
    _ensure_node()
    limb = Limb()

    # Build target pose from current but override orientation
    ps = _current_pose_as_ps(limb, tip_name)
    if ps is None:
        return
    ps.header.stamp = rospy.Time.now()
    ps.header.frame_id = 'base'
    ps.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)

    # Prepare IK options
    seed = limb.joint_ordered_positions() if use_advanced_ik else None
    ns_goal = None  # customize if desired
    joints = _solve_ik(ps, tip_name=tip_name, limb_name=limb_name,
                       seed_joint_positions=seed,
                       nullspace_goal=ns_goal)

    if joints is None:
        return

    #rospy.loginfo("IK Joint Solution: %s", joints)
    _execute_joint_dict(joints, speed_ratio=speed_ratio, timeout=timeout)
    #rospy.loginfo("Move complete.")

def orient_vertical(tip_name, **kwargs):
    """Convenience: x=0,y=1,z=0,w=0 quaternion."""
    set_absolute_orientation(tip_name, 0.0, 1.0, 0.0, 0.0, **kwargs)

def orient_horizontal(tip_name, **kwargs):
    """Convenience: x=1,y=0,z=0,w=0 quaternion."""
    set_absolute_orientation(tip_name, 1.0, 0.0, 0.0, 0.0, **kwargs)


def set_absolute_pose(tip_name, x, y, z, qx, qy, qz, qw, timeout=10.0,
                      lin_speed=0.6, lin_acc=0.6,
                      rot_speed=1.57, rot_acc=1.57,
                      limb_name='right',
                      speed_ratio=0.3,
                      use_advanced_ik=False):
    """
    Move tip to absolute (x,y,z) and absolute orientation (qx,qy,qz,qw),
    using IK -> joint command in a single solve/command.

    Returns:
        dict of joint_name->position on success, or None on failure.
    """
    _ensure_node()
    limb = Limb()

    # Start from current pose to preserve any fields we don't touch
    ps = _current_pose_as_ps(limb, tip_name)
    if ps is None:
        return None

    ps.header.stamp = rospy.Time.now()
    ps.header.frame_id = 'base'
    ps.pose.position = Point(x=x, y=y, z=z)
    ps.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)

    # Optional: seed & nullspace like your other helpers
    seed = limb.joint_ordered_positions() if use_advanced_ik else None
    ns_goal = None

    joints = _solve_ik(ps, tip_name=tip_name, limb_name=limb_name,
                       seed_joint_positions=seed,
                       nullspace_goal=ns_goal)
    if joints is None:
        return None

    #rospy.loginfo("IK Joint Solution: %s", joints)
    _execute_joint_dict(joints, speed_ratio=speed_ratio, timeout=timeout)
    #rospy.loginfo("Move complete.")
    return joints



# ---------------------------
# Example usage (optional)
# ---------------------------
if __name__ == "__main__":
    # Example: move right_hand to x,y,z while keeping current orientation
    set_absolute_position('right_hand', 0.45, 0.16, 0.22, speed_ratio=0.3)
    # Example: set a specific absolute orientation (quaternion)
    set_absolute_orientation('right_hand', 0.7040, 0.7101, 0.00244, 0.00194, speed_ratio=0.3)
