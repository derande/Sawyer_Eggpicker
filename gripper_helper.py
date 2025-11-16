import argparse
import time

import rospy

import intera_interface
import intera_external_devices
from intera_interface import CHECK_VERSION

CLOSED = 0.0
OPEN = 0.0407741357142857
limb = "right"

def check_object_gripped() -> bool :
    gripper = None
    original_deadzone = None
    def clean_shutdown():
        if gripper and original_deadzone:
            gripper.set_dead_zone(original_deadzone)
        print("Exiting example.")
    try:
        gripper = intera_interface.Gripper(limb + '_gripper')
    except (ValueError, OSError) as e:
        rospy.logerr("Could not detect an electric gripper attached to the robot.")
        clean_shutdown()
        return

    rospy.on_shutdown(clean_shutdown)
    pos = gripper.get_position()

    return pos > CLOSED and pos < OPEN
