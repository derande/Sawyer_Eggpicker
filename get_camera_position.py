#!/usr/bin/env python
import rospy
import tf2_ros
import geometry_msgs.msg

def get_camera_position():
    try:
        tfBuffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(tfBuffer)
        rate = rospy.Rate(10.0)

        trans = tfBuffer.lookup_transform('base', 'right_hand_camera', rospy.Time(0), rospy.Duration(1.0))
        t = trans.transform.translation
        r = trans.transform.rotation
        return t, r
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        rospy.logwarn("Transform not available yet")
        rate.sleep()
        return get_camera_position()
