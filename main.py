from absolute_positioning import set_absolute_orientation, set_absolute_position


def main():
    gripper = intera_interface.Gripper(limb + '_gripper')
    gripper.open()
    set_absolute_position("right_hand", 0.518, -0.068, 0.389)
    set_absolute_orientation("right_hand", 1.0, 0.0, 0.0, 0.0)
    set_absolute_position("right_hand", 0.577, -0.099, 0.0)
    set_absolute_position("right_hand", 0.518, -0.068, 0.389)
    gripper.close()

if __name__ == "__main__":
    main()
