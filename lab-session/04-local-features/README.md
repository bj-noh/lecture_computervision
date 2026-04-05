# Local Features

Source lecture: `#5 지역 특징`

## Contents

- `local_features_lab.ipynb`: guided notebook covering Harris corners, SIFT, and descriptor matching
- `examples/01_harris_response_toy.py`: Harris response on a small synthetic pattern
- `examples/02_harris_corners_real.py`: Harris corner detection on a real image
- `examples/03_sift_keypoints.py`: SIFT keypoint detection and descriptor inspection
- `examples/04_sift_flann_match.py`: SIFT descriptor matching with FLANN and Lowe ratio test

## Run

```bash
pip install opencv-python matplotlib numpy notebook
```

```bash
python examples/01_harris_response_toy.py
```

```bash
jupyter notebook local_features_lab.ipynb
```

## Notes

- Recent `opencv-python` builds usually include `cv2.SIFT_create()`.
- If your local OpenCV build does not support SIFT, reinstall a newer OpenCV wheel.
- Outputs are saved to `outputs/` for reuse in later labs.
