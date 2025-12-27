"""
Report generation utilities.
"""

import os
import numpy as np


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
