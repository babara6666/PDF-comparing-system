"""
Configuration settings for PDF comparison tool.
"""

# Default settings
DEFAULT_DPI = 300
DEFAULT_INPUT_DIR = r"D:\DownloadD\Stanley\FS\IMG_diff\input"
DEFAULT_OUTPUT_DIR = "output"

# Image processing parameters
GAUSSIAN_BLUR_KERNEL = (3, 3)
GAUSSIAN_BLUR_SIGMA = 0

# Alignment parameters
ALIGNMENT_SCALE = 0.5  # Downscale factor for faster feature detection
SIFT_TREES = 5
SIFT_CHECKS = 50
LOWE_RATIO = 0.7
RANSAC_THRESHOLD = 5.0
MIN_MATCHES = 4

# Heatmap parameters
DIFF_THRESHOLD = 30
HEATMAP_ALPHA = 0.6
JPEG_QUALITY = 95
