# Quick Start Guide

Get up and running with Waldo Detection in 5 minutes!

## 🚀 Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/waldo-detection-project.git
cd waldo-detection-project

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## 📁 Prepare Your Data

### Option 1: Use Pre-trained Model (Recommended)

If you have a pre-trained model:

```bash
# Just download and place it in data/models/
mkdir -p data/models
# Place your best.pt file here
```

### Option 2: Train From Scratch

#### Step 1: Organize Your Dataset

```
data/
├── processed/
│   ├── images/
│   │   ├── train/    # Put training images here
│   │   └── val/      # Put validation images here
│   └── labels/
│       ├── train/    # YOLO .txt labels
│       └── val/      # YOLO .txt labels
└── waldo_refs/       # 5-10 reference Waldo images
```

#### Step 2: Generate Synthetic Data (Optional)

```bash
# Extract Waldo from annotated images
python src/preprocessing/extract_waldo.py \
    --annotated data/annotated/ \
    --refs data/waldo_refs/ \
    --output data/waldo_crops/

# Generate synthetic training data
python src/data_generation/create_synthetic_data.py \
    --backgrounds data/backgrounds/ \
    --waldo-refs data/waldo_crops/ \
    --output data/synthetic/ \
    --n-per-bg 5 \
    --debug
```

#### Step 3: Train the Model

```bash
# Train YOLOv8s (recommended)
python src/training/train_yolo.py \
    --data data.yaml \
    --model s \
    --epochs 40 \
    --batch 8 \
    --device 0

# For CPU training
python src/training/train_yolo.py \
    --data data.yaml \
    --model s \
    --epochs 40 \
    --batch 4 \
    --device cpu
```

Training will save results to `runs/train/waldo_detector/weights/best.pt`

## 🔍 Run Detection

### Single Image

```bash
python src/inference/detect_with_clip.py \
    --model runs/train/waldo_detector/weights/best.pt \
    --input test_images/test1.jpg \
    --output results/ \
    --refs data/waldo_refs/
```

### Batch Processing (Folder)

```bash
python src/inference/detect_with_clip.py \
    --model runs/train/waldo_detector/weights/best.pt \
    --input test_images/ \
    --output results/ \
    --refs data/waldo_refs/ \
    --yolo-conf 0.5 \
    --clip-threshold 0.3
```

### Large Images (Tiled Processing)

```bash
python src/inference/detect_with_clip.py \
    --model runs/train/waldo_detector/weights/best.pt \
    --input large_images/ \
    --output results/ \
    --refs data/waldo_refs/ \
    --tile-size 640 \
    --overlap 100
```

## 📊 View Results

Results will be saved in the `results/` directory with bounding boxes drawn on the images.

```bash
# View a result
open results/test1.jpg  # macOS
xdg-open results/test1.jpg  # Linux
start results/test1.jpg  # Windows
```

## 🛠️ Configuration

### Adjust Detection Parameters

Edit these parameters based on your needs:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--yolo-conf` | 0.5 | YOLO confidence threshold (lower = more detections) |
| `--clip-threshold` | 0.3 | CLIP similarity threshold (higher = stricter) |
| `--tile-size` | 640 | Size of tiles for large images |
| `--overlap` | 100 | Overlap between tiles (px) |

### Example Adjustments

**More sensitive detection** (find Waldo even if uncertain):
```bash
--yolo-conf 0.3 --clip-threshold 0.2
```

**Very strict detection** (only high confidence):
```bash
--yolo-conf 0.7 --clip-threshold 0.5
```

## 🐛 Troubleshooting

### Out of Memory Error

```bash
# Reduce batch size
--batch 4  # or even 2

# Use smaller model
--model n  # instead of s
```

### CUDA Not Available

```bash
# Use CPU instead
--device cpu
```

### No Waldo Detected

1. **Lower thresholds**: `--yolo-conf 0.3 --clip-threshold 0.2`
2. **Check reference images**: Need 5-10 good quality Waldo images in `data/waldo_refs/`
3. **Increase tile overlap**: `--overlap 150` for very large images

### False Positives

1. **Raise thresholds**: `--yolo-conf 0.6 --clip-threshold 0.4`
2. **Add more diverse reference images**
3. **Retrain with more negative examples**

## 💡 Pro Tips

1. **Best Reference Images**:
   - 5-10 clear, frontal views of Waldo
   - Different scales and poses
   - High quality PNG with transparent background

2. **Optimal Tile Settings**:
   - For 2000×1500 images: `--tile-size 640 --overlap 100`
   - For 4000×3000 images: `--tile-size 640 --overlap 150`

3. **GPU Memory**:
   - Batch size 8 needs ~6GB VRAM
   - Batch size 4 needs ~3GB VRAM
   - Use CPU if no GPU available (slower but works)

## 🎯 Next Steps

- Read [METHODOLOGY.md](docs/METHODOLOGY.md) for detailed explanation
- Check [examples/](examples/) for sample code
- Join discussions on GitHub Issues
- Share your results!

Happy Waldo hunting! 🔍🎯
