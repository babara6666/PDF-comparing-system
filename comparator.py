"""
Image comparison and heatmap generation utilities.
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from config import DIFF_THRESHOLD, HEATMAP_ALPHA


def compute_difference(ref_img, target_img):
    """
    Compute SSIM and difference map between two images.
    
    Args:
        ref_img: Reference image (grayscale)
        target_img: Target image (grayscale)
    
    Returns:
        SSIM score, difference map (0-255)
    """
    # Compute SSIM
    score, diff = ssim(ref_img, target_img, full=True)
    
    # Convert difference map to 0-255 range
    diff = (1 - diff) * 255
    diff = diff.astype(np.uint8)
    
    return score, diff


def generate_heatmap(ref_img_color, diff_map, threshold=DIFF_THRESHOLD):
    """
    Generate heatmap overlay on reference image.
    
    Args:
        ref_img_color: Original reference image (BGR)
        diff_map: Difference map (grayscale, 0-255)
        threshold: Threshold for highlighting differences
    
    Returns:
        Heatmap overlay image (BGR)
    """
    # Threshold the difference map
    _, diff_thresh = cv2.threshold(diff_map, threshold, 255, cv2.THRESH_BINARY)
    
    # Apply colormap (JET: red = high difference, blue = no difference)
    heatmap = cv2.applyColorMap(diff_map, cv2.COLORMAP_JET)
    
    # Create mask for blending (only show heatmap where differences exist)
    mask = diff_thresh.astype(float) / 255.0
    mask = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR).astype(float) / 255.0
    
    # Blend heatmap with original image
    alpha = HEATMAP_ALPHA  # Heatmap opacity
    overlay = (alpha * heatmap * mask + (1 - alpha * mask) * ref_img_color).astype(np.uint8)
    
    return overlay
