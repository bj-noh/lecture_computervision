# Vision Agent

Source lecture: `#6 비전 에이전트`

## Contents

- `vision_agent_lab.ipynb`: guided notebook that explains the core image-processing logic behind the apps
- `examples/qt_cv_utils.py`: shared helpers for converting OpenCV frames into PyQt widgets
- `examples/01_pyqt_control_panel.py`: simple control panel with alert feedback
- `examples/02_pyqt_video_capture.py`: webcam preview, capture, and save app
- `examples/03_pyqt_grabcut_agent.py`: interactive GrabCut segmentation app
- `examples/04_pyqt_color_zone_alert.py`: color-based warning agent on webcam frames
- `examples/05_pyqt_photo_effects.py`: image effect studio with save support

## Run

```bash
pip install opencv-python matplotlib numpy notebook PyQt5
```

```bash
python examples/02_pyqt_video_capture.py
```

```bash
jupyter notebook vision_agent_lab.ipynb
```

## Notes

- The notebook focuses on the vision logic because full PyQt event loops are better explored through scripts.
- The PyQt apps use file dialogs and native windows, so they are best run from a desktop Python environment.
- `GrabCut` uses a PyQt button panel plus an OpenCV painting window, following the lecture style.
