import cv2
import rospy
import numpy as np

def orientation_from_contour(cnt):
    area = cv2.contourArea(cnt)
    if area < 50:  # very small → ignore
        return None

    if len(cnt) >= 5:
        # fitEllipse needs >= 5 points and returns (center, (MA, ma), angle)
        ellipse = cv2.fitEllipse(cnt)
        (cx, cy), (MA, ma), angle_deg = ellipse
        major = max(MA, ma); minor = min(MA, ma)
        # Angle correction: OpenCV's angle is for the reported "MA" axis
        if MA < ma:
            angle_deg = (angle_deg + 90.0) % 180.0
        return (cx, cy), major, minor, angle_deg

    # Fallback: PCA if fitEllipse not possible
    pts = cnt.reshape(-1, 2).astype(np.float32)
    mean, eigenvectors = cv2.PCACompute(pts, mean=None)  # EV[0] = main direction
    cx, cy = mean[0]
    v = eigenvectors[0]  # (vx, vy)
    angle_rad = np.arctan2(v[1], v[0])
    angle_deg = (np.degrees(angle_rad)) % 180.0
    # Rough major/minor
    return (cx, cy), 1.0, 0.5, angle_deg