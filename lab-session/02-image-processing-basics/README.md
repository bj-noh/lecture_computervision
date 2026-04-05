# Image Processing Basics

Source lecture: `#3 영상 처리 기초`

## Contents

- `image_processing_basics_lab.ipynb`: guided notebook version of the core lecture topics
- `examples/01_threshold_otsu.py`: fixed threshold and Otsu thresholding
- `examples/02_morphology.py`: erosion, dilation, opening, and closing
- `examples/03_gamma_histogram.py`: gamma correction and histogram equalization
- `examples/04_filtering.py`: average blur, Gaussian blur, and emboss filtering
- `examples/05_geometric_transform.py`: translation, scaling, rotation, and reflection
- `examples/06_dissolve_transition.py`: weighted dissolve between two images

## Run

```bash
pip install opencv-python matplotlib numpy notebook
```

```bash
python examples/01_threshold_otsu.py
```

```bash
jupyter notebook image_processing_basics_lab.ipynb
```

## Notes

- The notebook uses `matplotlib` for inline visualization.
- The scripts use OpenCV windows for interactive comparison.
- Outputs are saved to `outputs/` so results can be reused later.
