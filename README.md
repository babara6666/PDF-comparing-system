# High-Precision PDF Drawing Comparator - Usage Guide

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### Simple Command

Just place your PDF files in the `input/` folder and run:

```bash
python main.py
```

The tool will automatically:
- Find all PDF files in `D:\DownloadD\Stanley\FS\IMG_diff\input`
- Compare the first two PDFs (sorted alphabetically)
- Save results to `output/` folder

### Configuration

To change settings, edit the configuration in `main.py`:

```python
input_dir = r"D:\DownloadD\Stanley\FS\IMG_diff\input"  # Input folder
output_dir = "output"  # Output folder
dpi = 300  # Rendering DPI
```

## Output

The tool generates:

1. **Heatmap images**: `comparison_page_1.jpg`, `comparison_page_2.jpg`, etc.
   - Red areas = High differences
   - Blue areas = No differences
   - Blended with original reference for context

2. **Text report**: `comparison_report.txt`
   - SSIM score for each page
   - Average, minimum, and maximum SSIM scores
   - Page numbers with most/least differences

## How It Works

1. **High-DPI Rendering**: Converts each PDF page to 300 DPI images (4x standard resolution)
2. **Preprocessing**: Converts to grayscale and applies Gaussian blur to reduce rendering noise
3. **Alignment**: Uses SIFT feature detection to align target with reference
   - Optimized: Calculates alignment on 50% scaled images, applies to full resolution
4. **Difference Detection**: Computes SSIM (Structural Similarity Index) and pixel-wise differences
5. **Heatmap Generation**: Creates JET colormap overlay showing differences
6. **Multi-page Processing**: Processes all pages with progress bar

## Example Workflow

1. Place your PDFs in the `input/` folder:
   ```
   D:\DownloadD\Stanley\FS\IMG_diff\input\
   ├── drawing_v1.pdf
   └── drawing_v2.pdf
   ```

2. Run comparison:
   ```bash
   python main.py
   ```

3. Check results in `output/` folder:
   ```
   output/
   ├── comparison_page_1.jpg
   ├── comparison_page_2.jpg
   ├── ...
   └── comparison_report.txt
   ```

**Note**: The tool compares the first two PDFs alphabetically. If you have more than 2 PDFs, it will use the first two.

## Tips

- **Higher DPI**: Edit `dpi = 300` to `dpi = 400` or `dpi = 600` in `main.py` for finer detail detection (slower)
- **Memory Issues**: If processing very large drawings (A0 size), reduce DPI to 200-250 in `main.py`
- **File Order**: PDFs are compared alphabetically. Name them like `1_reference.pdf` and `2_target.pdf` to control order
- **Alignment Quality**: The tool reports number of SIFT matches per page - more matches = better alignment
- **SSIM Interpretation**: 
  - 1.0 = Identical
  - 0.95-0.99 = Very similar (minor differences)
  - 0.80-0.95 = Moderate differences
  - <0.80 = Significant differences

## Troubleshooting

**"Not enough keypoints detected"**
- PDFs may be too different to align
- Try reducing DPI or check if pages are completely blank

**"Alignment failed"**
- Images may have insufficient features (e.g., mostly blank pages)
- Tool will still generate difference map without alignment

**Memory errors**
- Reduce DPI: `--dpi 200`
- Process fewer pages at once (split PDFs)
