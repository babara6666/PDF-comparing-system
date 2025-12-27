"""
Image preprocessing and alignment utilities.
"""

import cv2
import numpy as np

from config import (
    GAUSSIAN_BLUR_KERNEL,
    GAUSSIAN_BLUR_SIGMA,
    ALIGNMENT_SCALE,
    SIFT_TREES,
    SIFT_CHECKS,
    LOWE_RATIO,
    RANSAC_THRESHOLD,
    MIN_MATCHES
)


def preprocess_image(img):
    """
    Preprocess image for comparison.
    
    Args:
        img: BGR image
    
    Returns:
        Grayscale image with noise reduction
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply slight Gaussian blur to reduce high-frequency noise from rendering
    blurred = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_KERNEL, GAUSSIAN_BLUR_SIGMA)
    return blurred


def align_images(ref_img, target_img):
    """
    Align target image to reference using SIFT and homography.
    Optimized: Calculate homography on downscaled images, apply to full resolution.
    
    Args:
        ref_img: Reference image (grayscale)
        target_img: Target image to align (grayscale)
    
    Returns:
        Aligned target image, homography matrix, match quality (int)
    """
    # Downscale for faster feature detection
    scale = ALIGNMENT_SCALE
    ref_small = cv2.resize(ref_img, None, fx=scale, fy=scale)
    target_small = cv2.resize(target_img, None, fx=scale, fy=scale)
    
    # SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(ref_small, None)
    kp2, des2 = sift.detectAndCompute(target_small, None)
    
    if des1 is None or des2 is None or len(kp1) < MIN_MATCHES or len(kp2) < MIN_MATCHES:
        print("Warning: Not enough keypoints detected. Returning unaligned image.")
        return target_img, None, 0
    
    # Match features using FLANN
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=SIFT_TREES)
    search_params = dict(checks=SIFT_CHECKS)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Apply Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < LOWE_RATIO * n.distance:
                good_matches.append(m)
    
    if len(good_matches) < MIN_MATCHES:
        print(f"Warning: Only {len(good_matches)} good matches found. Returning unaligned image.")
        return target_img, None, len(good_matches)
    
    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Calculate homography
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, RANSAC_THRESHOLD)
    
    if H is None:
        print("Warning: Homography calculation failed. Returning unaligned image.")
        return target_img, None, len(good_matches)
    
    # Scale homography matrix back to full resolution
    scale_matrix = np.array([[1/scale, 0, 0], [0, 1/scale, 0], [0, 0, 1]])
    H_full = scale_matrix @ H @ np.linalg.inv(scale_matrix)
    
    # Warp target image to align with reference
    h, w = ref_img.shape
    aligned = cv2.warpPerspective(target_img, H_full, (w, h))
    
    return aligned, H_full, len(good_matches)
