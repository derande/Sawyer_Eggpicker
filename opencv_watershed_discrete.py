#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Egg detection via watershed + geometric filtering, with live visualization.

Features:
- One-shot API: detect_egg_once(...)
- Smarter candidate selection (egg filter + distance + area)
- Edge-touch penalty so partially visible eggs are de-prioritized
- Side-clearance filter: reject eggs with neighbors along the minor axis mid-section
- Persistent OpenCV window "Egg Detection" that updates on each call
- Optional ROS Image publisher for overlays (view with rqt_image_view / image_view)
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
from orientation_from_contour import orientation_from_contour

# -----------------------------
# Tunables
# -----------------------------
WINDOW_NAME = "Egg Detection"
WINDOW_ENABLED_DEFAULT = True           # set False to disable OpenCV window by default
PUBLISH_OVERLAY_DEFAULT = True          # publish overlay to ROS for image viewers
OVERLAY_TOPIC_DEFAULT = "/egg_detection/overlay"
MIN_PIXELS_DEFAULT = 50                 # min pixels per connected component to consider
FG_THRESH_DEFAULT = 0.5                 # distance-transform threshold factor (0..1)
USE_TIGHT_NEUTRAL_MASK = False          # toggle color mask style (see color_mask_bgr)

# Edge-touch penalty controls
EDGE_MARGIN_PX_DEFAULT = 2              # within this many pixels of any edge counts as "touching"
EDGE_PENALTY_PER_EDGE_FRAC_DEFAULT = 0.25
# ↑ penalty per touched edge as a fraction of max(image_h, image_w), added to the score
# (score is in "pixels", where lower is better)

# NEW: Side-clearance controls
SIDE_CLEARANCE_PX_DEFAULT = 12          # outward clearance required on each side (minor axis)
SIDE_MID_FRACTION_DEFAULT = 0.40        # fraction of half major-axis to keep (mid-section only)
SIDE_INSET_PX_DEFAULT = 1               # start the check this many px outside the egg boundary

# Module-level singletons for viewer/publisher
_bridge_singleton: Optional[CvBridge] = None
_overlay_pub_singleton: Optional[rospy.Publisher] = None
_window_initialized: bool = False


# -----------------------------
# Geometry helpers
# -----------------------------
def _fit_ellipse_params(contour):
    """
    Returns (cx, cy, a, b, angle_deg) for a rotated ellipse fit to the contour,
    where a >= b are *radii* (half-lengths). Returns None if fit not possible.
    """
    if len(contour) < 5:
        return None
    (cx, cy), (MA, ma), angle = cv2.fitEllipse(contour)  # MA >= ma is NOT guaranteed
    a = max(MA, ma) * 0.5
    b = min(MA, ma) * 0.5
    return (cx, cy, a, b, float(angle))


def _circularity(contour):
    area = cv2.contourArea(contour)
    perim = cv2.arcLength(contour, True)
    if perim <= 1e-6:
        return 0.0
    return 4.0 * np.pi * area / (perim * perim)


def _solidity(contour):
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 1e-6:
        return 0.0
    return area / hull_area


def is_egg(contour):
    """
    Returns whether the detected object is an egg (shape heuristics).
    Tune thresholds for your setup.
    """
    area = cv2.contourArea(contour)
    if area < 100:  # reject tiny specks early
        return False

    fit = _fit_ellipse_params(contour)
    if fit is None:
        return False

    cx, cy, a, b, angle = fit
    aspect = a / b  # >= 1

    # Typical eggs: elongated but not too thin; tune these for your setup.
    if not (1.15 <= aspect <= 3.0):
        return False

    circ = _circularity(contour)         # 1 = perfect circle
    if not (0.55 <= circ <= 0.90):
        return False

    soli = _solidity(contour)            # 1 = fully convex
    if soli < 0.90:
        return False

    return True


# -----------------------------
# Color mask(s)
# -----------------------------
def _mask_not_dark(img_bgr):
    """Very permissive 'not dark' mask in HSV."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0,  0,  50], dtype=np.uint8)
    upper = np.array([180, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    return mask


def _mask_light_neutral(img_bgr):
    """
    Tighter mask for light neutral (white/beige) eggs in cluttered scenes.
    Adjust S/V bounds to your lighting.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 120], dtype=np.uint8)
    upper = np.array([180, 60, 255], dtype=np.uint8)
    mask_neutral = cv2.inRange(hsv, lower, upper)
    v = hsv[..., 2]
    mask_v = cv2.inRange(v, 80, 250)
    return cv2.bitwise_and(mask_neutral, mask_v)


def color_mask_bgr(img_bgr):
    return _mask_light_neutral(img_bgr) if USE_TIGHT_NEUTRAL_MASK else _mask_not_dark(img_bgr)


# -----------------------------
# Results dataclass
# -----------------------------
@dataclass
class EggDetection:
    ok: bool
    img_size: Tuple[Optional[int], Optional[int]]             # (height, width)
    center_uv: Optional[Tuple[float, float]]                  # (u,v) pixel center of detected egg (x,y)
    offset_uv: Optional[Tuple[float, float]]                  # (du,dv) from image center
    offset_norm: Optional[Tuple[float, float]]                # normalized offset in [-1,1]
    angle_deg: Optional[float]                                # orientation ∈ [0,180)
    bbox: Optional[Tuple[int, int, int, int]]                 # (x,y,w,h)
    area: Optional[float]                                     # contour area
    label_id: Optional[int]                                   # watershed label chosen


# -----------------------------
# Visualization
# -----------------------------
def _ensure_window():
    global _window_initialized
    if not _window_initialized:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        _window_initialized = True


def _draw_overlay(base_bgr: np.ndarray,
                  selected_contour: Optional[np.ndarray],
                  bbox: Optional[Tuple[int, int, int, int]],
                  angle_deg: Optional[float],
                  centroid: Optional[Tuple[float, float]],
                  side_band_polys: Optional[List[np.ndarray]] = None,
                  side_band_clear: Optional[List[bool]] = None) -> np.ndarray:
    """
    Returns a copy of base_bgr with overlays (contour, bbox, angle axis, centroid).
    If side bands are provided, draws them (green=clear, red=blocked).
    """
    out = base_bgr.copy()

    # Draw selected contour
    if selected_contour is not None and len(selected_contour) > 0:
        cv2.drawContours(out, [selected_contour], -1, (0, 255, 0), 2)

    # Draw bbox
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 200, 0), 2)

    # Draw centroid and orientation axis
    if centroid is not None:
        u, v = int(round(centroid[0])), int(round(centroid[1]))
        cv2.circle(out, (u, v), 4, (0, 0, 255), -1)

        if angle_deg is not None:
            theta = np.deg2rad(angle_deg)
            length = max(30, int(0.1 * max(out.shape[0], out.shape[1])))
            dx = int(length * np.cos(theta))
            dy = int(length * np.sin(theta))
            p1 = (u - dx, v - dy)
            p2 = (u + dx, v + dy)
            cv2.line(out, p1, p2, (0, 0, 255), 2)
            cv2.putText(out, f"{angle_deg:.1f} deg", (u + 8, v - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 255), 1, cv2.LINE_AA)

    # Side band visualization
    if side_band_polys is not None and side_band_clear is not None:
        for poly, is_clear in zip(side_band_polys, side_band_clear):
            color = (0, 220, 0) if is_clear else (0, 0, 255)
            cv2.polylines(out, [poly], isClosed=True, color=color, thickness=2)
            cv2.fillPoly(out, [poly], color if is_clear else (0, 0, 120))

    return out


def _show_in_window(img_bgr: np.ndarray, window_enabled: bool):
    if not window_enabled:
        return
    _ensure_window()
    cv2.imshow(WINDOW_NAME, img_bgr)
    cv2.waitKey(1)


def _publish_overlay(img_bgr: np.ndarray, publish_enabled: bool, topic: str):
    global _overlay_pub_singleton, _bridge_singleton
    if not publish_enabled:
        return
    if _bridge_singleton is None:
        _bridge_singleton = CvBridge()
    if _overlay_pub_singleton is None:
        _overlay_pub_singleton = rospy.Publisher(topic, Image, queue_size=1)
    try:
        msg = _bridge_singleton.cv2_to_imgmsg(img_bgr, encoding="bgr8")
        _overlay_pub_singleton.publish(msg)
    except Exception as e:
        rospy.logwarn(f"Overlay publish failed: {e}")


# -----------------------------
# Edge-touch helpers
# -----------------------------
def _edges_touched(bbox: Tuple[int, int, int, int], w: int, h: int, margin: int) -> int:
    """
    Returns how many image edges are touched by the bbox, counting within 'margin' pixels as touching.
    Edges: left, top, right, bottom. Return value in [0..4].
    """
    x, y, bw, bh = bbox
    left   = x <= margin
    top    = y <= margin
    right  = (x + bw) >= (w - 1 - margin)
    bottom = (y + bh) >= (h - 1 - margin)
    return int(left) + int(top) + int(right) + int(bottom)


# -----------------------------
# Side-clearance helpers (NEW)
# -----------------------------
def _unit_vectors_from_angle(angle_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """Returns unit vectors (u_major, u_minor) for the given major-axis angle in degrees."""
    th = np.deg2rad(angle_deg)
    u_major = np.array([np.cos(th), np.sin(th)], dtype=np.float32)
    u_minor = np.array([-np.sin(th), np.cos(th)], dtype=np.float32)
    return u_major, u_minor


def _rotated_rect_polygon(center: Tuple[float, float],
                          size_wh: Tuple[float, float],
                          angle_deg: float) -> np.ndarray:
    """Returns 4x2 polygon (int32) for a rotated rectangle suitable for fillPoly."""
    box = ((float(center[0]), float(center[1])), (float(size_wh[0]), float(size_wh[1])), float(angle_deg))
    pts = cv2.boxPoints(box)  # 4x2 float
    return np.int32(np.round(pts))


def _side_bands_for_ellipse(cx: float, cy: float, a: float, b: float, angle_deg: float,
                            clearance_px: float, mid_fraction: float, inset_px: float) -> List[np.ndarray]:
    """
    Build two rotated rectangles (left/right along minor axis) representing the 'no neighbor' zones.
    - Each band is centered just outside the ellipse boundary along ±u_minor by (b + inset_px + clearance_px/2).
    - Each band's long dimension lies along the major axis and spans the 'mid-section':
        length_major = 2 * mid_fraction * (2a) = 4 * mid_fraction * a (but we use 2*mid_fraction*a for |t|<=mid_fraction*a)
    - Each band's thickness along minor axis = clearance_px.
    Returns [poly_left, poly_right] as 4x2 int point arrays.
    """
    u_major, u_minor = _unit_vectors_from_angle(angle_deg)

    # Mid-section length along major axis: cover from -mid_fraction*a .. +mid_fraction*a
    length_major = 2.0 * mid_fraction * a * 2.0 * 1.0  # == 4*mid_fraction*a; using explicit form for clarity
    # Minor thickness is just the clearance requirement
    thickness_minor = clearance_px

    # Centers for left/right bands (negative/positive along u_minor)
    offset = b + inset_px + 0.5 * clearance_px
    c = np.array([cx, cy], dtype=np.float32)
    c_left  = c - offset * u_minor
    c_right = c + offset * u_minor

    size_wh = (float(length_major), float(thickness_minor))
    poly_left  = _rotated_rect_polygon((c_left[0],  c_left[1]),  size_wh, angle_deg)
    poly_right = _rotated_rect_polygon((c_right[0], c_right[1]), size_wh, angle_deg)
    return [poly_left, poly_right]


def _band_has_other_labels(band_poly: np.ndarray, markers: np.ndarray, own_label: int) -> bool:
    """
    True if the band polygon overlaps any watershed label other than own_label (and not border -1 or background 1/0).
    """
    h, w = markers.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [band_poly], 255)
    # We only care about where mask==255
    roi = markers[mask == 255]
    if roi.size == 0:
        return False
    # Valid foreign labels: >1 and != own_label
    return np.any((roi > 1) & (roi != own_label))


def _is_side_clear_for_label(ellipse_params: Tuple[float, float, float, float, float],
                             markers: np.ndarray,
                             label: int,
                             clearance_px: float,
                             mid_fraction: float,
                             inset_px: float) -> Tuple[bool, List[np.ndarray], List[bool]]:
    """
    Returns (is_clear, [polys], [per-band clear bool]).
    is_clear is True only if BOTH side bands (left & right) have no foreign labels.
    """
    cx, cy, a, b, angle_deg = ellipse_params
    bands = _side_bands_for_ellipse(cx, cy, a, b, angle_deg, clearance_px, mid_fraction, inset_px)
    clear_flags = []
    for poly in bands:
        blocked = _band_has_other_labels(poly, markers, label)
        clear_flags.append(not blocked)
    is_clear = all(clear_flags)
    return is_clear, bands, clear_flags


# -----------------------------
# Segmentation & selection
# -----------------------------
def _segment_and_select_egg(
    img_bgr: np.ndarray,
    min_pixels: int = MIN_PIXELS_DEFAULT,
    fg_thresh_factor: float = FG_THRESH_DEFAULT,
    edge_margin_px: int = EDGE_MARGIN_PX_DEFAULT,
    edge_penalty_per_edge_frac: float = EDGE_PENALTY_PER_EDGE_FRAC_DEFAULT,
    side_clearance_px: int = SIDE_CLEARANCE_PX_DEFAULT,
    side_mid_fraction: float = SIDE_MID_FRACTION_DEFAULT,
    side_inset_px: int = SIDE_INSET_PX_DEFAULT,
) -> Optional[Dict[str, Any]]:
    """
    Runs watershed pipeline, filters labels via is_egg, applies side-clearance rule,
    and selects best candidate.
    Returns dict with keys: label, contour, bbox, angle_deg, area, centroid, markers, mask,
                            sure_fg, sure_bg, edges_touched, side_band_polys, side_band_clear
    or None if nothing valid found.
    """
    img = img_bgr.copy()
    h, w = img.shape[:2]

    # 1) Preprocess
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    # 2) Binary mask
    mask = color_mask_bgr(blur)

    # 3) Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # 4) Background
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 5) Foreground via distance transform
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, fg_thresh_factor * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    # 6) Unknown
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 7) Markers
    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1               # background => 1
    markers[unknown == 255] = 0         # unknown => 0

    # 8) Watershed
    markers = cv2.watershed(img, markers)  # borders = -1

    # 9) First pass: collect egg-like candidates
    center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
    prelim = []  # store all egg-like candidates with ellipse params
    for label in range(2, num_labels + 1):
        ys, xs = np.where(markers == label)
        if xs.size < min_pixels:
            continue

        mask_label = (markers == label).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)

        # Filter by egg shape (kept separate)
        if not is_egg(c):
            continue

        # Fit / orientation (angle is major-axis angle modulo 180)
        angle_deg = None
        orient = orientation_from_contour(c)
        if orient is not None:
            (_, _), axes, _, angle_deg = orient
            angle_deg = float(angle_deg % 180.0)
        else:
            # Fallback: try local fit
            fit = _fit_ellipse_params(c)
            if fit is None:
                continue
            _, _, _, _, angle_deg = fit

        # Basic measures
        M = cv2.moments(c)
        if M["m00"] <= 1e-5:
            continue
        u = float(M["m10"] / M["m00"])
        v = float(M["m01"] / M["m00"])
        area = float(cv2.contourArea(c))
        x, y, wbb, hbb = cv2.boundingRect(c)

        # Fit ellipse radii for bands
        fit = _fit_ellipse_params(c)
        if fit is None:
            continue
        cx, cy, a, b, _ = fit

        prelim.append({
            "label": label,
            "contour": c,
            "bbox": (int(x), int(y), int(wbb), int(hbb)),
            "area": area,
            "centroid": (u, v),
            "ellipse": (cx, cy, a, b, angle_deg),
        })

    if not prelim:
        return None

    # 10) Apply side-clearance rule (reject if neighbor in minor-axis mid-section)
    filtered: List[Dict[str, Any]] = []
    for cand in prelim:
        cx, cy, a, b, angle_deg = cand["ellipse"]
        ok_side, bands, clear_flags = _is_side_clear_for_label(
            (cx, cy, a, b, angle_deg),
            markers,
            cand["label"],
            clearance_px=float(side_clearance_px),
            mid_fraction=float(side_mid_fraction),
            inset_px=float(side_inset_px),
        )
        if not ok_side:
            # Disqualify: neighbor too close at the sides
            continue

        # Keep for scoring
        cand["side_band_polys"] = bands
        cand["side_band_clear"] = clear_flags
        filtered.append(cand)

    if not filtered:
        return None

    # 11) Score remaining candidates: prefer centered, larger, and fewer edge touches
    side_max = float(max(h, w))
    scored = []
    for cand in filtered:
        u, v = cand["centroid"]
        d_center = np.linalg.norm(np.array([u, v], dtype=np.float32) - center)
        x, y, wbb, hbb = cand["bbox"]
        edges = _edges_touched((x, y, wbb, hbb), w, h, edge_margin_px)
        edge_penalty = edges * (edge_penalty_per_edge_frac * side_max)
        score = d_center - 0.001 * cand["area"] + edge_penalty
        scored.append({
            **cand,
            "score": score,
            "edges_touched": edges,
        })

    pick = min(scored, key=lambda c: c["score"])
    # Enrich with debug mats
    pick.update({
        "markers": markers,
        "mask": mask,
        "sure_fg": sure_fg,
        "sure_bg": sure_bg
    })
    return pick


# -----------------------------
# One-shot public API
# -----------------------------
@dataclass
class _DetectCfg:
    min_pixels: int = MIN_PIXELS_DEFAULT
    fg_thresh_factor: float = FG_THRESH_DEFAULT
    edge_margin_px: int = EDGE_MARGIN_PX_DEFAULT
    edge_penalty_per_edge_frac: float = EDGE_PENALTY_PER_EDGE_FRAC_DEFAULT
    side_clearance_px: int = SIDE_CLEARANCE_PX_DEFAULT
    side_mid_fraction: float = SIDE_MID_FRACTION_DEFAULT
    side_inset_px: int = SIDE_INSET_PX_DEFAULT


def detect_egg_once(
    image_topic: str = "/io/internal_camera/right_hand_camera/image_raw",
    timeout: float = 2.0,
    min_pixels: int = MIN_PIXELS_DEFAULT,
    fg_thresh_factor: float = FG_THRESH_DEFAULT,
    window_enabled: bool = WINDOW_ENABLED_DEFAULT,
    publish_overlay: bool = PUBLISH_OVERLAY_DEFAULT,
    overlay_topic: str = OVERLAY_TOPIC_DEFAULT,
    edge_margin_px: int = EDGE_MARGIN_PX_DEFAULT,
    edge_penalty_per_edge_frac: float = EDGE_PENALTY_PER_EDGE_FRAC_DEFAULT,
    side_clearance_px: int = SIDE_CLEARANCE_PX_DEFAULT,
    side_mid_fraction: float = SIDE_MID_FRACTION_DEFAULT,
    side_inset_px: int = SIDE_INSET_PX_DEFAULT,
) -> EggDetection:
    """
    Grab ONE image from the camera topic, segment eggs via watershed,
    enforce egg shape and side-clearance, choose the best candidate
    (centered & sufficiently large), de-prioritize eggs touching edges,
    and return pose info.

    Side effects:
      - Updates the OpenCV window "Egg Detection" with selection overlay.
      - Optionally publishes the overlay to /egg_detection/overlay (configurable).

    Returns EggDetection with ok=False if nothing valid was found.

    Notes:
      - Normalized offsets in [-1,1]; dv grows downward in image coords.
      - angle_deg is modulo 180° (ellipse major axis orientation).
    """
    global _bridge_singleton
    if _bridge_singleton is None:
        _bridge_singleton = CvBridge()

    try:
        msg = rospy.wait_for_message(image_topic, Image, timeout=timeout)
        rospy.loginfo("image update")
    except rospy.ROSException:
        return EggDetection(False, (None, None), None, None, None, None, None, None, None)

    try:
        frame = _bridge_singleton.imgmsg_to_cv2(msg, "bgr8")
    except Exception as e:
        rospy.logerr("cv_bridge: %s", e)
        return EggDetection(False, (None, None), None, None, None, None, None, None, None)

    h, w = frame.shape[:2]
    pick = _segment_and_select_egg(
        frame,
        min_pixels=min_pixels,
        fg_thresh_factor=fg_thresh_factor,
        edge_margin_px=edge_margin_px,
        edge_penalty_per_edge_frac=edge_penalty_per_edge_frac,
        side_clearance_px=side_clearance_px,
        side_mid_fraction=side_mid_fraction,
        side_inset_px=side_inset_px,
    )

    overlay = frame
    if pick is not None:
        # Build overlay with side bands shown
        overlay = _draw_overlay(
            frame,
            pick["contour"],
            pick["bbox"],
            pick["ellipse"][4],  # angle_deg
            pick["centroid"],
            side_band_polys=pick.get("side_band_polys"),
            side_band_clear=pick.get("side_band_clear"),
        )
        # If edges were touched, annotate the overlay
        if pick.get("edges_touched", 0) > 0:
            cv2.putText(
                overlay,
                f"Edge penalty: {pick['edges_touched']} edge(s)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA
            )
        # Annotate side clearance params
        cv2.putText(
            overlay,
            f"Side clearance={side_clearance_px}px, mid={side_mid_fraction:.2f}",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 0), 2, cv2.LINE_AA
        )

    _show_in_window(overlay, window_enabled=window_enabled)
    _publish_overlay(overlay, publish_enabled=publish_overlay, topic=overlay_topic)

    if pick is None:
        return EggDetection(False, (h, w), None, None, None, None, None, None, None)

    # Use centroid from selection
    u, v = pick["centroid"]

    # Offsets (pixels) relative to image center
    du = u - (w / 2.0)
    dv = v - (h / 2.0)

    # Normalized offsets in [-1,1]
    off_u = (2.0 * u / w) - 1.0
    off_v = (2.0 * v / h) - 1.0

    return EggDetection(
        ok=True,
        img_size=(h, w),
        center_uv=(u, v),
        offset_uv=(du, dv),
        offset_norm=(off_u, off_v),
        angle_deg=pick["ellipse"][4],
        bbox=pick["bbox"],
        area=pick["area"],
        label_id=pick["label"]
    )


# -----------------------------
# Optional: run as a script
# -----------------------------
def _parse_bool(param_name: str, default: bool) -> bool:
    try:
        return bool(rospy.get_param(param_name, default))
    except Exception:
        return default


def _parse_float(param_name: str, default: float) -> float:
    try:
        return float(rospy.get_param(param_name, default))
    except Exception:
        return default


def _parse_int(param_name: str, default: int) -> int:
    try:
        return int(rospy.get_param(param_name, default))
    except Exception:
        return default


def main():
    rospy.init_node("egg_detection_one_shot", anonymous=True)

    image_topic = rospy.get_param("~image_topic", "/io/internal_camera/right_hand_camera/image_raw")
    timeout = _parse_float("~timeout", 2.0)
    min_pixels = _parse_int("~min_pixels", MIN_PIXELS_DEFAULT)
    fg_thresh_factor = _parse_float("~fg_thresh_factor", FG_THRESH_DEFAULT)
    window_enabled = _parse_bool("~window_enabled", WINDOW_ENABLED_DEFAULT)
    publish_overlay = _parse_bool("~publish_overlay", PUBLISH_OVERLAY_DEFAULT)
    overlay_topic = rospy.get_param("~overlay_topic", OVERLAY_TOPIC_DEFAULT)

    # Edge penalty params
    edge_margin_px = _parse_int("~edge_margin_px", EDGE_MARGIN_PX_DEFAULT)
    edge_penalty_per_edge_frac = _parse_float("~edge_penalty_per_edge_frac", EDGE_PENALTY_PER_EDGE_FRAC_DEFAULT)

    # Side-clearance params (NEW)
    side_clearance_px = _parse_int("~side_clearance_px", SIDE_CLEARANCE_PX_DEFAULT)
    side_mid_fraction = _parse_float("~side_mid_fraction", SIDE_MID_FRACTION_DEFAULT)
    side_inset_px = _parse_int("~side_inset_px", SIDE_INSET_PX_DEFAULT)

    rate_hz = _parse_float("~rate_hz", 2.0)  # how often to run detection when launched as a node
    rate = rospy.Rate(rate_hz)

    rospy.loginfo("egg_detection: running. Press Ctrl+C to stop.")
    while not rospy.is_shutdown():
        result = detect_egg_once(
            image_topic=image_topic,
            timeout=timeout,
            min_pixels=min_pixels,
            fg_thresh_factor=fg_thresh_factor,
            window_enabled=window_enabled,
            publish_overlay=publish_overlay,
            overlay_topic=overlay_topic,
            edge_margin_px=edge_margin_px,
            edge_penalty_per_edge_frac=edge_penalty_per_edge_frac,
            side_clearance_px=side_clearance_px,
            side_mid_fraction=side_mid_fraction,
            side_inset_px=side_inset_px,
        )
        if result.ok:
            rospy.loginfo_throttle(
                1.0,
                f"Egg OK: center={result.center_uv}, angle={result.angle_deg:.1f}°, "
                f"offset_px={result.offset_uv}, norm={result.offset_norm}"
            )
        else:
            rospy.loginfo_throttle(1.0, "No valid egg found.")
        rate.sleep()

    if _window_initialized:
        cv2.destroyWindow(WINDOW_NAME)


if __name__ == "__main__":
    main()
