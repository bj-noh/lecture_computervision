# OpenCV Practice

Source lecture: `#2 OpenCV practice`

## Contents

- `opencv_practice_lab.ipynb`: guided notebook for the main lecture flow
- `examples/01_image_read_save.py`: read, display, and save an image
- `examples/02_image_transform.py`: grayscale conversion and resizing
- `examples/03_webcam_view.py`: live webcam preview
- `examples/04_webcam_capture_strip.py`: capture frames and merge them horizontally
- `examples/05_draw_shapes.py`: draw lines, rectangles, circles, and text
- `examples/06_mouse_rectangle.py`: draw rectangles with mouse clicks
- `examples/07_paint_brush.py`: simple paint application with mouse drag

## Run

Install the common packages:

```bash
pip install opencv-python matplotlib numpy notebook
```

Run an example script:

```bash
python examples/01_image_read_save.py
```

Launch the notebook:

```bash
jupyter notebook opencv_practice_lab.ipynb
```

## Notes

- The notebook focuses on image-based examples that work well step by step.
- Webcam and mouse callback exercises are separated into scripts because they rely on GUI event loops.
- Paths are handled relative to each file, so the scripts can be run from any working directory.
