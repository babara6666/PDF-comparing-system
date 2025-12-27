"""
PDF to image conversion utilities.
"""

import cv2
import fitz  # PyMuPDF
import numpy as np


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


def calculate_zoom_from_dpi(dpi):
    """
    Calculate zoom factor from desired DPI.
    
    Args:
        dpi: Desired DPI for rendering
    
    Returns:
        Zoom factor for PyMuPDF
    """
    return dpi / 72.0
