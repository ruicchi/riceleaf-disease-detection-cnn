# 🌾 Rice Leaf Disease Classification using CNN

> [!IMPORTANT]
> This repository contains only a proof of concept for a university research title defense. It was created for demonstration purposes and is not actively maintained or intended for production use.

A simple Convolutional Neural Network (CNN) built with PyTorch to classify rice leaf diseases.

## Disease Classes
| # | Class | Description |
|---|-------|-------------|
| 1 | Bacterial Leaf Blight | Water-soaked lesions on leaf edges |
| 2 | Brown Spot | Oval brown spots on leaves |
| 3 | Healthy | No disease present |
| 4 | Leaf Smut | Black powdery growth on leaves |

## Setup

```bash
pip install -r requirements.txt
```

## Dataset Structure

Organize your images like this:

```
dataset/
├── train/
│   ├── BrownSpot/
│   ├── Healthy/
│   ├── Hispa/
│   └── LeafBlast/
└── val/
    ├── BrownSpot/
    ├── Healthy/
    ├── Hispa/
    └── LeafBlast/
```

> 💡 You can download rice leaf disease datasets from [Kaggle](https://www.kaggle.com/datasets/vbookshelf/rice-leaf-diseases).

## Train the Model

```bash
python main.py train
```

## Predict a Disease

```bash
python main.py predict path/to/leaf_image.jpg
```

## Model Architecture

```
Input (3x128x128)
  → Conv2d(32) → BN → ReLU → MaxPool
  → Conv2d(64) → BN → ReLU → MaxPool
  → Conv2d(128) → BN → ReLU → MaxPool
  → Conv2d(256) → BN → ReLU → MaxPool
  → Flatten → FC(512) → ReLU → Dropout
  → FC(4) → Output
```
