#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from dataclasses import dataclass
from typing import Optional, Tuple
from orientation_from_contour import orientation_from_contour


def color_mask_bgr(img_bgr):
    """
    Example "not-dark" mask in HSV. Adjust for your egg color if needed.
    For a specific plastic egg color, tighten these bounds.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0,  0,  50], dtype=np.uint8)
    upper = np.array([180, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    return mask


@dataclass
class EggDetection:
    ok: bool
    img_size: Tuple[int, int]             # (height, width)
    center_uv: Optional[Tuple[float,float]]  # (u,v) pixel center of detected egg (x,y)
    offset_uv: Optional[Tuple[float,float]]  # (du,dv) from image center: u - w/2, v - h/2 (pixels)
    offset_norm: Optional[Tuple[float,float]] # normalized offset in [-1,1] approx: (2u/w-1, 2v/h-1)
    angle_deg: Optional[float]            # orientation in degrees ∈ [0,180)
    bbox: Optional[Tuple[int,int,int,int]]   # (x,y,w,h)
    area: Optional[float]                 # contour area
    label_id: Optional[int]               # watershed label chosen
    # For debugging (optional): could add masks/visualizations if you want to save images


def _watershed_and_pick_center(img_bgr, min_pixels=50):
    """
    Runs your watershed pipeline and selects the label whose centroid is closest
    to the image center. Returns (best_label, contour, bbox, angle, area, centers, markers).
    """
    img = img_bgr.copy()
    h, w = img.shape[:2]

    # 1) Preprocess
    blur = cv2.GaussianBlur(img, (5,5), 0)

    # 2) Binary mask
    mask = color_mask_bgr(blur)

    # 3) Morphological cleanup
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # 4) Background
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 5) Foreground via distance transform
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    # 6) Unknown
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 7) Markers
    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1               # background => 1
    markers[unknown == 255] = 0         # unknown => 0

    # 8) Watershed
    markers = cv2.watershed(img, markers)  # borders = -1

    # Pick most-centered valid label (labels: 2..num_labels+1)
    center = np.array([w/2.0, h/2.0], dtype=np.float32)
    best_label, best_dist = None, 1e12
    best_cnt, best_bbox, best_angle, best_area = None, None, None, None

    for label in range(2, num_labels+1):
        ys, xs = np.where(markers == label)
        if xs.size < min_pixels:
            continue
        cx, cy = xs.mean(), ys.mean()
        d = np.linalg.norm(np.array([cx, cy], dtype=np.float32) - center)
        if d < best_dist:
            best_dist = d
            best_label = label

    if best_label is None:
        return None

    # Build contour for the best label (external contour)
    mask_label = (markers == best_label).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    x,y,wbb,hbb = cv2.boundingRect(c)
    area = float(cv2.contourArea(c))

    orient = orientation_from_contour(c)
    angle_deg = None
    if orient is not None:
        (_, _), _, _, angle_deg = orient
        angle_deg = float(angle_deg % 180.0)

    return {
        "label": best_label,
        "contour": c,
        "bbox": (int(x), int(y), int(wbb), int(hbb)),
        "angle_deg": angle_deg,
        "area": area,
        "markers": markers
    }


# -----------------------------
# One-shot public API
# -----------------------------
def detect_egg_once(
    image_topic="/io/internal_camera/right_hand_camera/image_raw",
    timeout=2.0,
    min_pixels=50,
    return_debug=False
) -> EggDetection:
    """
    Grab ONE image from the camera topic, segment eggs via watershed,
    choose the blob whose centroid is closest to image center,
    and return its pixel center offset and orientation (deg).

    Returns EggDetection with ok=False if nothing valid was found.
    """
    bridge = CvBridge()
    try:
        msg = rospy.wait_for_message(image_topic, Image, timeout=timeout)
    except rospy.ROSException:
        return EggDetection(False, (0,0), None, None, None, None, None, None, None)

    try:
        frame = bridge.imgmsg_to_cv2(msg, "bgr8")
    except Exception as e:
        rospy.logerr("cv_bridge: %s", e)
        return EggDetection(False, (0,0), None, None, None, None, None, None, None)

    h, w = frame.shape[:2]
    pick = _watershed_and_pick_center(frame, min_pixels=min_pixels)
    if pick is None:
        return EggDetection(False, (h, w), None, None, None, None, None, None, None)

    # compute centroid precisely from contour moments
    c = pick["contour"]
    M = cv2.moments(c)
    if M["m00"] <= 1e-5:
        return EggDetection(False, (h, w), None, None, None, None, None, None, None)

    u = float(M["m10"] / M["m00"])  # x
    v = float(M["m01"] / M["m00"])  # y

    # Offsets (pixels) relative to image center
    du = u - (w / 2.0)
    dv = v - (h / 2.0)

    # Normalized offsets in [-1,1] (useful for simple proportional controllers)
    # NOTE: dv grows downward in image coords; keep as-is unless you prefer flip.
    off_u = (2.0 * u / w) - 1.0
    off_v = (2.0 * v / h) - 1.0

    return EggDetection(
        ok=True,
        img_size=(h, w),
        center_uv=(u, v),
        offset_uv=(du, dv),
        offset_norm=(off_u, off_v),
        angle_deg=pick["angle_deg"],
        bbox=pick["bbox"],
        area=pick["area"],
        label_id=pick["label"]
    )


# -----------------------------
# Example usage (call in a loop)
# -----------------------------
if __name__ == "__main__":
    rospy.init_node("egg_detector_discrete")
    topic = "/io/internal_camera/right_hand_camera/image_raw"  # adjust if needed

    # Example: try up to N times to refine position
    N = 5
    for i in range(N):
        det = detect_egg_once(topic, timeout=2.0, min_pixels=80)
        if not det.ok:
            rospy.logwarn("[step %d] No egg found.", i)
            break

        rospy.loginfo(
            "[step %d] center=(%.1f, %.1f) px  offset=(%+.1f, %+.1f) px  norm=(%+.3f, %+.3f)  angle=%.1f deg  bbox=%s  area=%.0f",
            i, det.center_uv[0], det.center_uv[1],
            det.offset_uv[0], det.offset_uv[1],
            det.offset_norm[0], det.offset_norm[1],
            det.angle_deg if det.angle_deg is not None else float("nan"),
            det.bbox, det.area if det.area is not None else -1
        )

        # Here you’d use det.offset_uv or det.offset_norm to move the camera
        # slightly toward the egg center (e.g., via IK target nudge). Then loop.

    # Final call to get orientation for grasp alignment:
    det = detect_egg_once(topic, timeout=2.0, min_pixels=80)
    if det.ok and det.angle_deg is not None:
        rospy.loginfo("Final orientation estimate: %.1f deg", det.angle_deg)
    else:
        rospy.logwarn("Could not determine final orientation.")
