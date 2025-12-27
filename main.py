#!/usr/bin/env python3
"""
High-Precision Engineering Drawing Comparator
Compares two multi-page PDF files at high resolution (300 DPI),
aligns them geometrically, and generates difference heatmaps.
"""

import os
import sys
import cv2
from tqdm import tqdm

# Import from modular components
from config import DEFAULT_DPI, DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR, JPEG_QUALITY
from pdf_converter import pdf_to_images, calculate_zoom_from_dpi
from image_processor import preprocess_image, align_images
from comparator import compute_difference, generate_heatmap
from report_generator import generate_report


def compare_pdfs(ref_path, target_path, output_dir, dpi=DEFAULT_DPI):
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
    zoom = calculate_zoom_from_dpi(dpi)
    
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
        cv2.imwrite(output_path, heatmap, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        
        # Print page info
        if num_matches > 0:
            print(f"  Page {i+1}: SSIM={score:.4f}, Matches={num_matches}")
        else:
            print(f"  Page {i+1}: SSIM={score:.4f}, Alignment failed")
    
    return ssim_scores


def main():
    # Configuration
    input_dir = DEFAULT_INPUT_DIR
    output_dir = DEFAULT_OUTPUT_DIR
    dpi = DEFAULT_DPI
    
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

