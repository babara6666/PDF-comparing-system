# PDF Comparison System - Modular Structure

## Overview
This project compares two multi-page PDF files at high resolution (300 DPI), aligns them geometrically using SIFT feature matching, and generates difference heatmaps.

## Project Structure

```
IMG_diff/
├── main.py                 # Entry point and orchestration
├── config.py              # Configuration parameters
├── pdf_converter.py       # PDF to image conversion
├── image_processor.py     # Image preprocessing and alignment
├── comparator.py          # Difference computation and heatmap generation
├── report_generator.py    # Report generation utilities
├── input/                 # Place PDF files here
└── output/                # Generated heatmaps and reports
```

## Module Descriptions

### `config.py`
Contains all configuration parameters:
- Default DPI, input/output directories
- Image processing parameters (blur kernels, etc.)
- Alignment parameters (SIFT settings, thresholds)
- Heatmap visualization settings

### `pdf_converter.py`
Handles PDF to image conversion:
- `pdf_to_images()` - Converts PDF pages to high-res numpy arrays
- `calculate_zoom_from_dpi()` - Calculates zoom factor from DPI

### `image_processor.py`
Image preprocessing and alignment:
- `preprocess_image()` - Converts to grayscale and applies noise reduction
- `align_images()` - SIFT-based feature matching and homography alignment

### `comparator.py`
Difference computation and visualization:
- `compute_difference()` - Calculates SSIM score and difference map
- `generate_heatmap()` - Creates color-coded heatmap overlay

### `report_generator.py`
Report generation:
- `generate_report()` - Creates text summary with SSIM statistics

### `main.py`
Orchestration and CLI:
- `compare_pdfs()` - Main comparison pipeline
- `main()` - Entry point with file discovery and execution

## Usage

```bash
python main.py
```

The script will:
1. Look for PDF files in the `input/` directory
2. Compare the first two PDFs (alphabetically sorted)
3. Generate heatmaps in the `output/` directory
4. Create a comparison report

## Customization

To modify parameters, edit `config.py`:

```python
# Change DPI
DEFAULT_DPI = 600

# Adjust heatmap sensitivity
DIFF_THRESHOLD = 20

# Modify alignment quality
LOWE_RATIO = 0.8
```

## Dependencies

- OpenCV (cv2)
- PyMuPDF (fitz)
- NumPy
- scikit-image
- tqdm

