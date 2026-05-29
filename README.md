# CIFAR-10 Image Classification with ResNet CNN

A deep learning solution for CIFAR-10 image classification achieving **93% test accuracy** using a custom ResNet-style convolutional neural network built with PyTorch.

---

## Results

| Metric | Score |
|--------|-------|
| Test Accuracy (Kaggle) | **0.93** |
| Best Validation Accuracy | ~0.93 |

---

## Model Architecture

A custom ResNet with residual connections, designed specifically for 32×32 images:

- **Input**: 32×32 RGB images
- **Prep layer**: Conv2d(3→64) + BatchNorm + ReLU
- **Layer 1**: 2× ResidualBlock(64→128) + MaxPool
- **Layer 2**: 2× ResidualBlock(128→256) + MaxPool
- **Layer 3**: 2× ResidualBlock(256→512) + MaxPool
- **Classifier**: AdaptiveAvgPool → Dropout(0.4) → Linear(512→10)

Each residual block uses a skip connection that adds the input directly to the output, allowing gradients to flow more effectively during training.

---

## Training Details

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 150 |
| Batch size | 128 |
| Optimizer | SGD + Nesterov momentum (0.9) |
| Learning rate | OneCycleLR (max 0.1) |
| Weight decay | 5e-4 |
| Loss function | CrossEntropyLoss (label smoothing 0.1) |

---

## Data Augmentation

Training uses the following augmentations to reduce overfitting:

- Random crop (32×32, padding=4)
- Random horizontal flip
- Color jitter (brightness, contrast, saturation, hue)
- Random rotation (±15°)
- Random grayscale (p=0.1)
- Random erasing / Cutout (p=0.3)
- Mixup augmentation (α=0.4)

---

## Test-Time Augmentation (TTA)

At inference time, each test image is passed through 10 different augmented versions (original, flipped, cropped, center-cropped, combinations). The softmax probabilities are averaged across all versions before the final class prediction is made.

---

## Dataset

**CIFAR-10** — 60,000 32×32 color images across 10 classes:

`airplane` · `automobile` · `bird` · `cat` · `deer` · `dog` · `frog` · `horse` · `ship` · `truck`

- 50,000 training images (90/10 train/val split)
- 10,000 test images

---

## Requirements

```
torch
torchvision
numpy
pandas
matplotlib
seaborn
scikit-learn
tqdm
```

---

## How to Run

1. Open the notebook in Kaggle (GPU recommended — T4 or better)
2. Attach the competition dataset
3. Run all cells in order
4. `submission.csv` will be generated in the output directory

---

## Key Techniques

**Residual connections** prevent the vanishing gradient problem in deep networks, allowing the model to learn more effectively with more layers.

**Mixup augmentation** blends two training images and their labels together, forcing the model to make smoother, more confident predictions and significantly improving generalization.

**OneCycleLR scheduler** starts with a low learning rate, ramps it up, then decays it — this approach consistently outperforms fixed or step-decay schedules on image classification tasks.
