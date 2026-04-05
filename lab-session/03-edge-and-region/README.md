# Edge and Region

Source lecture: `#4 엣지와 영역`

## Contents

- `edge_and_region_lab.ipynb`: guided notebook for the main lecture flow
- `examples/01_sobel_canny_log.py`: Sobel, Canny, and LoG comparison
- `examples/02_contour_detection.py`: contour extraction from an edge map
- `examples/03_hough_lines.py`: line detection with Hough transform
- `examples/04_hough_circles.py`: circle detection with Hough transform
- `examples/05_superpixels_slic.py`: SLIC-based superpixel segmentation
- `examples/06_region_features.py`: contour moments, area, perimeter, and hull

## Run

```bash
pip install opencv-python matplotlib numpy notebook scikit-image
```

```bash
python examples/01_sobel_canny_log.py
```

```bash
jupyter notebook edge_and_region_lab.ipynb
```

## Notes

- The notebook is optimized for inline visualization with `matplotlib`.
- The scripts use OpenCV windows for side-by-side inspection.
- Some region examples use built-in `scikit-image` sample data so they work without extra downloads.
