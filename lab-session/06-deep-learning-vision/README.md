# Deep Learning Vision

Source lecture: `#7 딥러닝 비전`

## Contents

- `deep_learning_vision_lab.ipynb`: guided notebook that moves from MNIST classification to detection and segmentation
- `examples/01_01_mlp_mnist.py`: simple multilayer perceptron for MNIST classification
- `examples/01_02_sgd_adam_comparison.py`: optimizer comparison on MNIST
- `examples/01_03_cifar10_overfitting.py`: overfitting demo on CIFAR-10
- `examples/02_01_lenet_mnist.py`: LeNet-style CNN reproduction on MNIST
- `examples/02_02_alexnet_cifar10.py`: AlexNet-style classifier on CIFAR-10
- `examples/02_03_zfnet_stl10.py`: ZFNet + Deconvnet visualization on STL-10
- `examples/02_04_vggnet_cifar100.py`: VGGNet training on CIFAR-100
- `examples/02_05_resnet_flowers102.py`: ResNet transfer learning on Flowers-102
- `examples/02_06_densenet_dogs.py`: DenseNet dog-breed classification on Oxford-IIIT Pet
- `examples/02_07_pretrained_resnet_inference.py`: ImageNet pre-trained ResNet inference on a sample image
- `examples/02_08_vit_flowers102.py`: Vision Transformer transfer learning on Flowers-102
- `examples/03_data_augmentation_demo.py`: augmentation gallery using torchvision transforms
- `examples/05_01_fasterrcnn_detection.py`: object detection with Faster R-CNN
- `examples/05_02_yolo_detection.py`: one-stage object detection with Ultralytics YOLO
- `examples/05_03_maskrcnn_segmentation.py`: instance segmentation with Mask R-CNN
- `examples/05_04_keypointrcnn_detection.py`: person keypoint detection with Keypoint R-CNN

## Run

```bash
pip install torch torchvision torchaudio matplotlib numpy pillow notebook ultralytics
```

```bash
python examples/05_01_fasterrcnn_detection.py --image data/object.jpg
```

```bash
python examples/05_02_yolo_detection.py --image data/object.jpg
```

```bash
python examples/05_03_maskrcnn_segmentation.py --image data/view.jpg
```

```bash
python examples/05_04_keypointrcnn_detection.py --image data/view.jpg
```

```bash
python examples/02_07_pretrained_resnet_inference.py --image data/dog1.jpg
```

```bash
python examples/02_08_vit_flowers102.py --epochs 12 --pretrained
```

```bash
jupyter notebook deep_learning_vision_lab.ipynb
```

## Notes

- The lecture introduces TensorFlow examples, but this lab is organized with PyTorch so that classification, segmentation, and detection can share one stack.
- The first run of pre-trained models may download weights from the internet.
- `Faster R-CNN`, `Mask R-CNN`, `Keypoint R-CNN`, and `YOLO` are heavier than the classification examples, so CPU execution can take a while.
