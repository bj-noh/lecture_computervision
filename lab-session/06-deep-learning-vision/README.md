# Deep Learning Vision

Source lecture: `#7 딥러닝 비전`

## Contents

- `deep_learning_vision_lab.ipynb`: guided notebook that moves from MNIST classification to detection and segmentation
- `examples/01_mnist_mlp.py`: simple multilayer perceptron for MNIST classification
- `examples/02_lenet_mnist.py`: LeNet-style CNN reproduction on MNIST
- `examples/03_data_augmentation_demo.py`: augmentation gallery using torchvision transforms
- `examples/04_pretrained_resnet_inference.py`: ImageNet pre-trained ResNet inference on sample images
- `examples/05_maskrcnn_inference.py`: instance segmentation with Mask R-CNN on a sample image
- `examples/06_yolo_inference.py`: one-stage object detection with Ultralytics YOLO

## Run

```bash
pip install torch torchvision torchaudio matplotlib numpy pillow notebook ultralytics
```

```bash
python examples/04_pretrained_resnet_inference.py --image data/dog1.jpg
```

```bash
python examples/05_maskrcnn_inference.py --image data/view.jpg
```

```bash
jupyter notebook deep_learning_vision_lab.ipynb
```

## Notes

- The lecture introduces TensorFlow examples, but this lab is organized with PyTorch so that classification, segmentation, and detection can share one stack.
- The first run of pre-trained models may download weights from the internet.
- `Mask R-CNN` and `YOLO` are heavier than the MNIST examples, so CPU execution can take a while.
