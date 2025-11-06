#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from orientation_from_contour import orientation_from_contour


bridge = CvBridge()

def color_mask_bgr(img):
    # Beispielhafte „nicht-dunkel“-Maske – passe ggf. für deine Farbe an
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0,  0,  50], dtype=np.uint8)
    upper = np.array([180, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    return mask

def watershed_segment(img_bgr):
    img = img_bgr.copy()
    h, w = img.shape[:2]

    # 1) Vorverarbeitung
    blur = cv2.GaussianBlur(img, (5,5), 0)

    # 2) Binärmaske
    mask = color_mask_bgr(blur)

    # 3) Morphologische Reinigung
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # 4) Hintergrund
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 5) Vordergrund via Distance Transform
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.5*dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    # 6) Unbekannt
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 7) Marker
    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1              # Hintergrund = 1
    markers[unknown == 255] = 0        # unbekannt = 0

    # 8) Watershed
    markers = cv2.watershed(img, markers)  # Ränder = -1

    # 9) Visualisierungen
    # Grenzen rot einzeichnen
    borders_vis = img.copy()
    borders_vis[markers == -1] = [0, 0, 255]

    # Zufällige Färbung pro Label (zur Übersicht)
    colored = np.zeros_like(img)
    for label in range(2, num_labels+1):  # 2..N sind echte Objekte
        colored[markers == label] = np.random.randint(0, 255, (1,3), dtype=np.uint8)

    # 10) Mittigstes Segment bestimmen
    # Schwerpunkt je Label, Abstand zur Bildmitte
    center = np.array([w/2.0, h/2.0], dtype=np.float32)
    best_label, best_dist = None, 1e12
    for label in range(2, num_labels+1):
        ys, xs = np.where(markers == label)
        if xs.size < 50:  # sehr kleine Flecken ignorieren (Tuning)
            continue
        cx, cy = xs.mean(), ys.mean()
        d = np.linalg.norm(np.array([cx, cy], dtype=np.float32) - center)
        if d < best_dist:
            best_dist = d
            best_label = label

    # 11) Ansicht „nur mittigstes Objekt“ mit eingezeichnetem Winkel
    highlight = img.copy()
    dimmed = (highlight * 0.2).astype(np.uint8)
    if best_label is not None:
        mask_label = (markers == best_label)
        highlight_view = dimmed.copy()
        # Originalfarben für bestes Label zurückholen
        highlight_view[mask_label] = img[mask_label]
        # Kontur/Bounding Box zeichnen
        mask_uint8 = mask_label.astype(np.uint8)*255
        cnts, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            x,y,wbb,hbb = cv2.boundingRect(c)
            cv2.rectangle(highlight_view, (x,y), (x+wbb,y+hbb), (0,255,0), 2)
            M = cv2.moments(c)
            if M["m00"] > 1e-5:
                cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                cv2.circle(highlight_view, (cx,cy), 5, (0,255,0), -1)
            orientation = orientation_from_contour(c)
            if orientation is not None:
                (ocx, ocy), major, minor, angle_deg = orientation
                length = int(0.5 * major)
                angle_rad = np.radians(angle_deg)
                x2 = int(ocx + length * np.cos(angle_rad))
                y2 = int(ocy + length * np.sin(angle_rad))
                x1 = int(ocx - length * np.cos(angle_rad))
                y1 = int(ocy - length * np.sin(angle_rad))
                cv2.line(highlight_view, (x1,y1), (x2,y2), (255,0,0), 2)
                cv2.circle(highlight_view, (int(ocx), int(ocy)), 4, (255,0,0), -1)
                cv2.putText(highlight_view, f"{angle_deg:.1f} deg", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        title = "Central object only"
    else:
        # Kein valides Label gefunden
        highlight_view = dimmed
        title = "Central object only (none found)"

    # 12) Nebeneinander-Ansichten
    vis_mask  = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    vis_open  = cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR)
    panel1 = cv2.hconcat([img, vis_mask, vis_open])
    panel2 = cv2.hconcat([borders_vis, colored, highlight_view])

    return panel1, panel2, title

def image_cb(msg):
    try:
        frame = bridge.imgmsg_to_cv2(msg, "bgr8")
    except Exception as e:
        rospy.logerr("cv_bridge: %s", e); return

    p1, p2, title = watershed_segment(frame)
    cv2.imshow("Watershed: [orig | mask | open]", p1)
    cv2.imshow(f"Watershed: [borders | labels | {title}]", p2)
    cv2.waitKey(3)

def main():
    rospy.init_node("opencv_watershed_subscriber", anonymous=True)
    topic = "/io/internal_camera/right_hand_camera/image_raw"  # ggf. anpassen
    rospy.Subscriber(topic, Image, image_cb, queue_size=1)
    rospy.loginfo("Subscribing to %s (Watershed with central-object highlight)", topic)
    rospy.spin()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
