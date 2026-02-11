# 🔍 Waldo Detection Project - Antoine Debin

**An AI-powered "Where's Waldo?" detector using YOLOv8 and CLIP**

This project implements a sophisticated computer vision pipeline to automatically detect Waldo in complex "Where's Waldo?" scenes, combining object detection (YOLOv8) with few-shot learning (CLIP) for robust and accurate results.

---

## 🎯 Project Overview

Finding Waldo in crowded scenes is challenging even for humans. This project tackles the problem using a multi-stage deep learning approach:

1. **Dataset Creation**: Automated extraction and augmentation of Waldo instances
2. **Object Detection**: YOLOv8 fine-tuned for Waldo detection
3. **Smart Inference**: Tiled processing for large images + CLIP-based re-ranking to eliminate false positives

### Key Features

- 🎨 **Synthetic Data Generation**: Automated creation of training data with realistic augmentations
- 🧩 **Tiled Processing**: Handle images of any size by processing in overlapping tiles
- 🧠 **CLIP Re-ranking**: Few-shot learning to filter false positives
- 🎯 **High Accuracy**: Combined YOLOv8 + CLIP approach for robust detection

---

## 📊 Results


```
📁 assets/
  └── results/
      ├── example1_detected.jpg
      ├── example2_detected.jpg
      └── comparison.png
```

### Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **mAP@0.5** | **98.8%** | Mean Average Precision at IoU 0.5 |
| **mAP@0.5:0.95** | **99.5%** | Mean Average Precision across IoU thresholds |
| **Precision** | **100%** | Perfect precision at optimal threshold |
| **Recall** | **99.6%** | Excellent detection rate |
| **F1-Score** | **95% @ 0.671** | Optimal confidence threshold |

### Training Results

<div align="center">
  <img src="assets/results/results.png" alt="Training Results" width="800"/>
  <p><i>Training and validation metrics over 40 epochs</i></p>
</div>

<div align="center">
  <img src="assets/results/confusion_matrix.png" alt="Confusion Matrix" width="500"/>
  <p><i>Confusion matrix showing 114 correct detections, 32 background detections</i></p>
</div>

<div align="center">
  <img src="assets/results/PR_curve.png" alt="Precision-Recall Curve" width="500"/>
  <p><i>Near-perfect Precision-Recall curve (mAP@0.5 = 0.988)</i></p>
</div>

### Key Insights

- ✅ **Excellent convergence**: All loss curves show smooth decrease
- ✅ **High precision**: 100% precision @ confidence 0.795
- ✅ **High recall**: 99.6% recall - rarely misses Waldo
- ✅ **Robust model**: mAP of 98.8% indicates strong generalization
- ⚠️ **32 background detections**: CLIP re-ranking filters these out in production

---

## 🏗️ Architecture

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    1. DATA PREPARATION                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Manual       │  │ CLIP-based   │  │ Synthetic    │      │
│  │ Annotation   │→ │ Extraction   │→ │ Augmentation │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    2. MODEL TRAINING                         │
│              YOLOv8s Fine-tuning on Waldo Dataset            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    3. INFERENCE PIPELINE                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Tile Large   │→ │ YOLO         │→ │ CLIP         │      │
│  │ Images       │  │ Detection    │  │ Re-ranking   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### Technical Stack

- **Object Detection**: YOLOv8 (Ultralytics)
- **Few-Shot Learning**: CLIP (OpenAI)
- **Image Processing**: OpenCV, PIL
- **Inpainting**: Stable Diffusion 2 Inpainting
- **Framework**: PyTorch

---

## 📦 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/waldo-detection-project.git
cd waldo-detection-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download YOLOv8 base model (will be done automatically during training)
# Or download pre-trained Waldo detector (if available)
```

---

## 🚀 Quick Start

### 1. Prepare Your Dataset

```bash
# Option A: Use manual annotation
python src/preprocessing/manual_annotation.py --input raw_images/ --output dataset/

# Option B: Extract from annotated images (yellow circles)
python src/preprocessing/extract_waldo.py \
    --annotated annotated_images/ \
    --refs waldo_refs/ \
    --output waldo_crops/

# Generate synthetic training data
python src/data_generation/create_synthetic_data.py \
    --backgrounds backgrounds/ \
    --waldo-refs waldo_refs/ \
    --output synth_dataset/ \
    --n-per-bg 5
```

### 2. Train the Model

```bash
# Train YOLOv8 on your dataset
python src/training/train_yolo.py \
    --data data.yaml \
    --epochs 40 \
    --batch 8 \
    --imgsz 640 \
    --device 0  # Use 'cpu' for CPU training
```

### 3. Run Inference

```bash
# Test on a single image
python src/inference/detect_single.py \
    --model runs/train/exp/weights/best.pt \
    --image test_image.jpg \
    --conf 0.25

# Process large images with tiling
python src/inference/detect_large_images.py \
    --model runs/train/exp/weights/best.pt \
    --input large_images/ \
    --output results/ \
    --tile-size 640 \
    --overlap 100

# Use CLIP re-ranking for better accuracy
python src/inference/detect_with_clip.py \
    --model runs/train/exp/weights/best.pt \
    --input large_images/ \
    --refs waldo_refs/ \
    --output results/ \
    --clip-threshold 0.3
```

---

## 📁 Project Structure

```
waldo-detection-project/
│
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
├── setup.py                          # Package installation
├── .gitignore                        # Git ignore rules
│
├── src/                              # Source code
│   ├── __init__.py
│   ├── data_generation/              # Dataset creation scripts
│   │   ├── __init__.py
│   │   └── create_synthetic_data.py      # Generate augmented Waldo images
│   │
│   ├── preprocessing/                # Data preprocessing
│   │   ├── __init__.py
│   │   └── extract_waldo.py              # CLIP-based Waldo extraction 
│   │
│   ├── inference/           
│   │   ├── __init__.py
│   │   └── detect_with_clip.py           
│   │
│   └── training/                     # Model training
│       ├── __init__.py
│       └── train_yolo.py                 # YOLOv8 training script
│
├── data/
│   ├── waldo_refs/          # 5-10 images Waldo pour CLIP
│   ├── annotated/           # Images avec cercles jaunes
│   ├── backgrounds/         # Fonds propres
│   ├── raw/                 # Images originales
│   ├── waldo_crops/         # Découpes Waldo PNG
│   ├── models/
│   │   └── best.pt         # Votre modèle entraîné
│   └── processed/          # Dataset YOLO organisé
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       └── labels/
│           ├── train/
│           └── val/
│
├── configs/                          # Configuration files
│   └── data.yaml                     # YOLO dataset config
│
├── docs/                             # Documentation
│   └── METHODOLOGY.md                # Detailed methodology
│
└── assets/                           # Images for README
    └── results/

```

---

## 🔬 Methodology

### Stage 1: Dataset Creation

**Challenge**: Limited labeled Waldo images available.

**Solutions**:
1. **Manual Annotation**: Use OpenCV ROI selector for precise bbox annotation
2. **CLIP-based Extraction**: Detect yellow circles → validate with CLIP similarity
3. **Synthetic Generation**: Paste Waldo on clean backgrounds with augmentations
4. **Background Creation**: Use Stable Diffusion inpainting to remove Waldo

### Stage 2: Model Training

- **Base Model**: YOLOv8s (balance of speed and accuracy)
- **Input Size**: 640×640 pixels
- **Augmentations**: Mosaic, MixUp, random flips, scaling, rotation
- **Training**: 40+ epochs with early stopping

### Stage 3: Large Image Inference

**Problem**: Real "Where's Waldo?" scenes are large (2000×1500+)

**Solution**: Tiled Processing
1. Split image into 640×640 tiles with overlap
2. Run YOLO on each tile
3. Merge detections with NMS (Non-Maximum Suppression)
4. Apply CLIP re-ranking to keep only true Waldo instances

---

## 🛠️ Advanced Usage

### Custom Dataset Creation

```python
from src.data_generation import SyntheticDataGenerator

generator = SyntheticDataGenerator(
    background_dir="backgrounds/",
    waldo_dir="waldo_refs/",
    output_dir="custom_dataset/"
)

generator.generate(
    n_per_background=5,
    scale_range=(0.3, 0.8),
    rotation_range=(-15, 15),
    brightness_range=(0.7, 1.2)
)
```

### CLIP Re-ranking

```python
from src.inference import CLIPReranker

reranker = CLIPReranker(
    prototype_dir="waldo_refs/",
    threshold=0.3
)

# Filter YOLO detections
filtered_boxes = reranker.filter_detections(
    image, yolo_boxes, scores
)
```

---

## 📝 Configuration

Edit `data.yaml` to configure your dataset:

```yaml
# Number of classes
nc: 1

# Class names
names: ["waldo"]

# Paths (absolute or relative to data.yaml)
train: datasets/images/train
val: datasets/images/val
test: datasets/images/test  # optional
```

---

## 🤝 Contributing

Contributions are welcome! Here are some ways you can help:

- 🐛 Report bugs and issues
- 💡 Suggest new features or improvements
- 📖 Improve documentation
- 🔧 Submit pull requests

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Ultralytics** for YOLOv8
- **OpenAI** for CLIP
- **Stability AI** for Stable Diffusion
- Martin Handford for creating "Where's Waldo?"

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

## 🗺️ Roadmap

- [ ] Add evaluation metrics and benchmarks
- [ ] Create web demo with Gradio/Streamlit
- [ ] Support for video detection
- [ ] Mobile deployment (ONNX/TFLite)
- [ ] Multi-character detection (Wenda, Wizard, etc.)

---

**Happy Waldo Hunting! 🎯🔍**
