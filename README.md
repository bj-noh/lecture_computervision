# Computer Vision Lab Session

<p align="center">
  <img src="docs/assets/course_hero.png" alt="Computer Vision Lab Session banner" width="100%">
</p>

## Demo

<p align="center">
  <img src="docs/assets/lab_demo.gif" alt="Lab demo preview" width="900">
</p>

## Lab Sessions

<table>
  <tr>
    <td width="50%" valign="top">
      <a href="lab-session/01-opencv-practice">
        <img src="docs/assets/thumb_01.png" alt="01 OpenCV Practice" width="100%">
      </a>
      <b>01 OpenCV Practice</b><br>
      <sub>image I/O, webcam capture, drawing, and mouse interaction</sub>
    </td>
    <td width="50%" valign="top">
      <a href="lab-session/02-image-processing-basics">
        <img src="docs/assets/thumb_02.png" alt="02 Image Processing Basics" width="100%">
      </a>
      <b>02 Image Processing Basics</b><br>
      <sub>thresholding, morphology, filtering, histogram equalization, and transforms</sub>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <a href="lab-session/03-edge-and-region">
        <img src="docs/assets/thumb_03.png" alt="03 Edge and Region" width="100%">
      </a>
      <b>03 Edge and Region</b><br>
      <sub>Sobel, Canny, contours, Hough transform, superpixels, and region features</sub>
    </td>
    <td width="50%" valign="top">
      <a href="lab-session/04-local-features">
        <img src="docs/assets/thumb_04.png" alt="04 Local Features" width="100%">
      </a>
      <b>04 Local Features</b><br>
      <sub>Harris corners, SIFT keypoints, descriptors, and FLANN matching</sub>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <a href="lab-session/05-vision-agent">
        <img src="docs/assets/thumb_05.png" alt="05 Vision Agent" width="100%">
      </a>
      <b>05 Vision Agent</b><br>
      <sub>PyQt interfaces, webcam agents, GrabCut, monitoring logic, and photo effects</sub>
    </td>
    <td width="50%" valign="top">
      <a href="lab-session/06-deep-learning-vision">
        <img src="docs/assets/thumb_06.png" alt="06 Deep Learning Vision" width="100%">
      </a>
      <b>06 Deep Learning Vision</b><br>
      <sub>MNIST, LeNet, augmentation, pre-trained ResNet, Mask R-CNN, and YOLO</sub>
    </td>
  </tr>
  <tr>
    <td colspan="2" valign="top">
      <sub>The lab numbering starts from lecture <code>#2</code>, so <code>06-deep-learning-vision</code> corresponds to lecture <code>#7</code>.</sub>
    </td>
  </tr>
</table>

## Quick Start

```bash
pip install opencv-python matplotlib numpy notebook scikit-image PyQt5 torch torchvision torchaudio ultralytics
```

Then open any lecture folder in `lab-session/` and start with the notebook or the scripts in `examples/`.

## Repository Layout

```text
lab-session/
├── 01-opencv-practice/
├── 02-image-processing-basics/
├── 03-edge-and-region/
├── 04-local-features/
├── 05-vision-agent/
└── 06-deep-learning-vision/
```

## Notes

- Every session includes its own `README.md`, `ipynb`, `examples/`, and `data/`.
- GUI and webcam examples should be run on a local desktop environment.
- `05-vision-agent` requires `PyQt5`.
- `06-deep-learning-vision` uses `PyTorch` and `Ultralytics YOLO`, and pre-trained weights may download on first run.
