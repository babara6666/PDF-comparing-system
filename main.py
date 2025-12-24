#!/usr/bin/env python3
"""
High-Precision Engineering Drawing Comparator
Compares two multi-page PDF files at high resolution (300 DPI),
aligns them geometrically, and generates difference heatmaps.
"""

import os
import sys
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


def pdf_to_images(pdf_path, zoom=4.0):
    """
    Convert PDF pages to high-resolution images.
    
    Args:
        pdf_path: Path to PDF file
        zoom: Zoom factor (4.0 = ~288 DPI, 4.17 = 300 DPI)
    
    Returns:
        List of numpy arrays (BGR format for OpenCV)
    """
    doc = fitz.open(pdf_path)
    images = []
    mat = fitz.Matrix(zoom, zoom)  # High resolution matrix
    
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        # Convert to numpy array (H, W, channels)
        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        
        # Convert to BGR for OpenCV
        if pix.n == 4:  # RGBA
            img_data = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
        else:  # RGB
            img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
        
        images.append(img_data)
    
    doc.close()
    return images


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
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
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
    scale = 0.5
    ref_small = cv2.resize(ref_img, None, fx=scale, fy=scale)
    target_small = cv2.resize(target_img, None, fx=scale, fy=scale)
    
    # SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(ref_small, None)
    kp2, des2 = sift.detectAndCompute(target_small, None)
    
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        print("Warning: Not enough keypoints detected. Returning unaligned image.")
        return target_img, None, 0
    
    # Match features using FLANN
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Apply Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
    
    if len(good_matches) < 4:
        print(f"Warning: Only {len(good_matches)} good matches found. Returning unaligned image.")
        return target_img, None, len(good_matches)
    
    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Calculate homography
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    
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


def generate_heatmap(ref_img_color, diff_map, threshold=30):
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
    alpha = 0.6  # Heatmap opacity
    overlay = (alpha * heatmap * mask + (1 - alpha * mask) * ref_img_color).astype(np.uint8)
    
    return overlay


def compare_pdfs(ref_path, target_path, output_dir, dpi=300):
    """
    Compare two multi-page PDFs and generate heatmaps.
    
    Args:
        ref_path: Path to reference PDF
        target_path: Path to target PDF
        output_dir: Directory to save output heatmaps
        dpi: Desired DPI for rendering
    
    Returns:
        List of SSIM scores per page
    """
    # Calculate zoom factor from DPI
    zoom = dpi / 72.0
    
    print(f"Loading PDFs at {dpi} DPI (zoom factor: {zoom:.2f})...")
    ref_images = pdf_to_images(ref_path, zoom)
    target_images = pdf_to_images(target_path, zoom)
    
    num_pages = min(len(ref_images), len(target_images))
    print(f"Processing {num_pages} pages...")
    
    if len(ref_images) != len(target_images):
        print(f"Warning: PDFs have different page counts ({len(ref_images)} vs {len(target_images)})")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    ssim_scores = []
    
    # Process each page
    for i in tqdm(range(num_pages), desc="Comparing pages"):
        ref_img = ref_images[i]
        target_img = target_images[i]
        
        # Preprocess
        ref_gray = preprocess_image(ref_img)
        target_gray = preprocess_image(target_img)
        
        # Align
        aligned_target, homography, num_matches = align_images(ref_gray, target_gray)
        
        # Compute difference
        score, diff_map = compute_difference(ref_gray, aligned_target)
        ssim_scores.append(score)
        
        # Generate heatmap
        heatmap = generate_heatmap(ref_img, diff_map)
        
        # Save result
        output_path = os.path.join(output_dir, f"comparison_page_{i+1}.jpg")
        cv2.imwrite(output_path, heatmap, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Print page info
        if num_matches > 0:
            print(f"  Page {i+1}: SSIM={score:.4f}, Matches={num_matches}")
        else:
            print(f"  Page {i+1}: SSIM={score:.4f}, Alignment failed")
    
    return ssim_scores


def generate_report(ssim_scores, output_dir):
    """
    Generate text report with SSIM scores.
    
    Args:
        ssim_scores: List of SSIM scores per page
        output_dir: Directory to save report
    """
    report_path = os.path.join(output_dir, "comparison_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("PDF Comparison Report\n")
        f.write("=" * 50 + "\n\n")
        
        for i, score in enumerate(ssim_scores):
            f.write(f"Page {i+1}: SSIM = {score:.4f}\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write(f"Average SSIM: {np.mean(ssim_scores):.4f}\n")
        f.write(f"Min SSIM: {np.min(ssim_scores):.4f} (Page {np.argmin(ssim_scores)+1})\n")
        f.write(f"Max SSIM: {np.max(ssim_scores):.4f} (Page {np.argmax(ssim_scores)+1})\n")
    
    print(f"\nReport saved to: {report_path}")


def main():
    # Configuration
    input_dir = r"D:\DownloadD\Stanley\FS\IMG_diff\input"
    output_dir = "output"
    dpi = 300
    
    # Find all PDF files in input directory
    pdf_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')])
    
    if len(pdf_files) < 2:
        print("=" * 60)
        print("Error: Need at least 2 PDF files in input directory")
        print("=" * 60)
        print(f"Input directory: {input_dir}")
        print(f"Found {len(pdf_files)} PDF(s): {pdf_files}")
        print("\nPlease place at least 2 PDF files in the input directory.")
        sys.exit(1)
    
    # Use first two PDFs (sorted alphabetically)
    ref_path = os.path.join(input_dir, pdf_files[0])
    target_path = os.path.join(input_dir, pdf_files[1])
    
    # Run comparison
    print("=" * 60)
    print("High-Precision Engineering Drawing Comparator")
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Found {len(pdf_files)} PDF file(s)")
    print(f"\nReference: {pdf_files[0]}")
    print(f"Target: {pdf_files[1]}")
    print(f"DPI: {dpi}")
    print(f"Output: {output_dir}/")
    print("=" * 60 + "\n")
    
    ssim_scores = compare_pdfs(ref_path, target_path, output_dir, dpi)
    
    # Generate report
    generate_report(ssim_scores, output_dir)
    
    print("\n" + "=" * 60)
    print("Comparison complete!")
    print(f"Heatmaps saved to: {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
