"""
Task A — Single-instance product detection
===========================================

Each product appears at most once per shelf, so a classic keypoint-matching
pipeline is sufficient:

    SIFT keypoints/descriptors  ->  FLANN matching  ->  Lowe's ratio test
    ->  RANSAC homography  ->  projected bounding box

This file is a SHOWCASE: it keeps the high-level pipeline and the function
signatures, while the heavier implementations are omitted.
"""

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Setup: SIFT detector + FLANN matcher
# --------------------------------------------------------------------------- #
FLANN_INDEX_KDTREE = 1

sift = cv2.SIFT_create()
index_params  = dict(algorithm=FLANN_INDEX_KDTREE, trees=12)
search_params = dict(checks=500)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Tunable thresholds
MIN_MATCH_COUNT      = 80
MIN_INLIERS          = 60
LOWE_RATIO           = 0.55
RANSAC_REPROJ_THRESH = 3.0


# --------------------------------------------------------------------------- #
# Key functions (signatures only)
# --------------------------------------------------------------------------- #
def classify_krave_variant_by_color(img_bgr: np.ndarray, box: np.ndarray):
    """Disambiguate colour-only variants (blue '1' vs. orange '11' Choco Krave).

    SIFT works on greyscale and cannot separate boxes that differ mainly by
    colour. This crops the detected quadrilateral, converts it to HSV and
    compares the amount of blue vs. orange pixels.

    Returns
    -------
    (predicted_product_id, scores) : (str | None, dict)
    """
    ...  # implementation withheld


# --------------------------------------------------------------------------- #
# Main pipeline (orchestration)
# --------------------------------------------------------------------------- #
def detect_single_instances(scene_path: str, references: list) -> dict:
    """Detect one instance of each reference product in a single shelf image."""
    img_bgr    = cv2.imread(scene_path)
    gray_scene = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kp_scene, desc_scene = sift.detectAndCompute(gray_scene, None)

    detections_per_product = {}

    for ref_path, img_query, kp_query, desc_query in references:
        # 1) match descriptors and keep good matches (Lowe's ratio test)
        matches = flann.knnMatch(desc_query, desc_scene, k=2)
        good = [m for m, n in matches if m.distance < LOWE_RATIO * n.distance]
        if len(good) < MIN_MATCH_COUNT:
            continue

        # 2) estimate homography with RANSAC
        src = np.float32([kp_query[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_scene[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_REPROJ_THRESH)
        if H is None or int(mask.sum()) < MIN_INLIERS:
            continue

        # 3) project the reference corners into the scene -> bounding box
        h, w = img_query.shape
        corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]])
        box = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H).reshape(4, 2)

        # 4) colour-based disambiguation for the Choco Krave variants
        product_id = ref_path  # simplified
        if product_id in ("1", "11"):
            variant, _ = classify_krave_variant_by_color(img_bgr, box)
            product_id = variant or product_id

        detections_per_product.setdefault(product_id, []).append(box)

    return detections_per_product
