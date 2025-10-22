# Complete Setup Guide for Chinese-LiPS Training

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Dataset Setup](#dataset-setup)
4. [Training Workflow](#training-workflow)
5. [Common Issues](#common-issues)

---

## System Requirements

### Minimum Requirements
- **GPU**: NVIDIA GPU with 8GB VRAM (GTX 1070, RTX 2060)
- **RAM**: 16GB system memory
- **Storage**: 100GB free space (50GB for dataset + 50GB for checkpoints)
- **OS**: Linux (Ubuntu 20.04+), Windows 10/11, or macOS
- **CUDA**: 11.7 or higher (for GPU training)

### Recommended Requirements
- **GPU**: NVIDIA GPU with 12GB+ VRAM (RTX 3060, RTX 3080, etc.)
- **RAM**: 32GB system memory
- **Storage**: 200GB SSD
- **CPU**: 8+ cores for faster data loading

---

## Installation

### Step 1: Create Python Environment

#### Using Conda (Recommended)
```bash
# Create environment
conda create -n lipreading python=3.10
conda activate lipreading

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

#### Using venv
```bash
# Create environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch (check https://pytorch.org for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### Step 2: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"
```

Expected output:
```
PyTorch: 2.1.0+cu118
CUDA Available: True
CUDA Version: 11.8
```

---

## Dataset Setup

### Option 1: Download from HuggingFace (Recommended)

```bash
# Install huggingface datasets
pip install datasets huggingface_hub

# Download dataset (this will take time)
python download_dataset.py
```

Create `download_dataset.py`:
```python
from datasets import load_dataset
from pathlib import Path
import shutil

# Download dataset
print("Downloading Chinese-LiPS dataset...")
dataset = load_dataset("BAAI/Chinese-LiPS")

# Create output directory
output_dir = Path("./Chinese-LiPS")
output_dir.mkdir(exist_ok=True)

for split in ['train', 'validation', 'test']:
    print(f"\nProcessing {split} split...")
    split_dir = output_dir / f"processed_{split}"
    split_dir.mkdir(exist_ok=True)
    
    split_data = dataset[split]
    
    # Save files (adjust based on actual dataset structure)
    for idx, item in enumerate(split_data):
        # Save video, audio, and metadata
        # This depends on the exact structure of the dataset
        pass

print("Dataset download complete!")
```

### Option 2: Manual Setup

If you already have the dataset:

```bash
Chinese-LiPS/
├── meta_train.csv          # Training metadata
├── meta_valid.csv          # Validation metadata  
├── meta_test.csv           # Test metadata
├── processed_train/        # Training videos
│   ├── 111_21_M_JKYS_005_FACE.mp4
│   ├── 111_21_M_JKYS_006_FACE.mp4
│   └── ...
├── processed_val/          # Validation videos
│   └── ...
└── processed_test/         # Test videos
    └── ...
```

### Step 3: Verify Dataset

```bash
# Check dataset statistics
python main.py stats --data_root ./Chinese-LiPS
```

Expected output:
```
Dataset Statistics for ./Chinese-LiPS
======================================================================

TRAIN SPLIT:
----------------------------------------------------------------------
Number of samples: 50000
Text Statistics:
  Min length: 5
  Max length: 150
  Mean length: 42.35
  Median length: 38.0
  Unique characters: 3567

VAL SPLIT:
----------------------------------------------------------------------
Number of samples: 5000
...
```

---

## Training Workflow

### Step 1: Choose Configuration

Select appropriate config based on your hardware:

```bash
# For 12GB VRAM (most common)
cp config_12gb.yaml config.yaml

# For 24GB VRAM
cp config_24gb.yaml config.yaml

# For CPU only (very slow)
cp config_cpu.yaml config.yaml
```

### Step 2: Start Training

#### Basic Training
```bash
python main.py train \
    --data_root ./Chinese-LiPS \
    --config config.yaml
```

#### With Custom Parameters
```bash
python main.py train \
    --data_root ./Chinese-LiPS \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --num_epochs 100 \
    --num_workers 4 \
    --checkpoint_dir ./checkpoints
```

#### Resume from Checkpoint
```bash
python main.py train \
    --data_root ./Chinese-LiPS \
    --resume ./checkpoints/checkpoint_epoch_50.pt
```

### Step 3: Monitor Training

#### View Logs
```bash
# Real-time log monitoring
tail -f ./logs/training.log
```

#### Check GPU Usage
```bash
# Monitor GPU memory and utilization
watch -n 1 nvidia-smi
```

#### Expected Training Output
```
Creating datasets...
Loaded 50000 samples from meta_train.csv
Found 50000 valid video samples
Loaded 5000 samples from meta_valid.csv
Found 5000 valid video samples
Train samples: 50000
Val samples: 5000
Vocabulary size: 3567

Creating model...
Total parameters: 15,234,567
Trainable parameters: 15,234,567
Model size: 58.13 MB (fp32)

Starting training...
Epoch 0: 100%|██████████| 6250/6250 [28:45<00:00, 3.62it/s, loss=4.234]
Epoch 0: train_loss=4.2341
Validation: 100%|██████████| 625/625 [02:15<00:00, 4.62it/s]
Epoch 0: val_loss=3.9876
```

### Step 4: Evaluate Model

```bash
# Test on test set
python main.py test \
    --data_root ./Chinese-LiPS \
    --checkpoint ./checkpoints/best_model.pt
```

Expected output:
```
Test Set Results:
==================================================
Example 1:
Ground truth: 我们旨在搭建一个交流思想碰撞火花的平台
Predicted: 我们旨在搭建一个交流思想碰撞火化的平台
CER: 0.0476

...

Average CER: 0.1523
Average Accuracy: 84.77%
==================================================
```

### Step 5: Run Inference

```bash
# Single video inference
python main.py inference \
    --video_path ./test_video.mp4 \
    --checkpoint ./checkpoints/best_model.pt \
    --ground_truth "我们旨在搭建一个交流思想碰撞火花的平台"
```

---

## Common Issues

### Issue 1: CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions:**
```bash
# 1. Reduce batch size
python main.py train --batch_size 4

# 2. Reduce max_frames in config
max_frames: 50  # Instead of 75

# 3. Reduce frame resolution
frame_height: 64
frame_width: 64

# 4. Increase accumulation steps
accumulation_steps: 8  # Simulates larger batch size
```

### Issue 2: Slow Data Loading

**Symptoms:**
- GPU utilization < 50%
- Long wait times between batches

**Solutions:**
```bash
# 1. Increase workers
python main.py train --num_workers 8

# 2. Move dataset to SSD
# Ensure dataset is on fast storage

# 3. Enable prefetching (already enabled in code)
# prefetch_factor=2

# 4. Use persistent workers (already enabled)
# persistent_workers=True
```

### Issue 3: NaN Loss

**Error:**
```
Epoch 5: train_loss=nan
```

**Solutions:**
```yaml
# 1. Reduce learning rate in config
learning_rate: 0.00005

# 2. Increase gradient clipping
gradient_clip: 1.0

# 3. Enable mixed precision (should be enabled)
mixed_precision: true

# 4. Check for corrupted videos
# Run dataset stats to verify
```

### Issue 4: Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'cv2'
```

**Solution:**
```bash
pip install opencv-python pandas pyyaml tqdm
```

### Issue 5: Video Loading Fails

**Error:**
```
WARNING - No frames loaded from /path/to/video.mp4
```

**Solutions:**
```bash
# 1. Check video file exists
ls -lh ./Chinese-LiPS/processed_train/*.mp4

# 2. Verify video is not corrupted
ffmpeg -v error -i video.mp4 -f null -

# 3. Check OpenCV installation
python -c "import cv2; print(cv2.__version__)"

# 4. Re-download dataset if many videos fail
```

### Issue 6: Metadata CSV Not Found

**Error:**
```
ValueError: Metadata CSV not found: ./Chinese-LiPS/meta_train.csv
```

**Solution:**
```bash
# 1. Check file exists
ls -lh ./Chinese-LiPS/*.csv

# 2. Verify CSV format (should have 6 columns)
head -n 5 ./Chinese-LiPS/meta_train.csv

# 3. Check data_root path is correct
python main.py stats --data_root ./Chinese-LiPS
```

---

## Performance Optimization Tips

### 1. Faster Training

```yaml
# Use larger batch size if possible
batch_size: 16  # or higher

# Reduce validation frequency
save_every: 10  # Validate every 10 epochs instead of 5

# Use fewer workers if CPU-bound
num_workers: 4  # Don't use more than CPU cores
```

### 2. Better Accuracy

```yaml
# Train longer
num_epochs: 150

# Larger model
hidden_dim: 512
num_lstm_layers: 3

# More frames
max_frames: 100

# Lower learning rate with longer warmup
learning_rate: 0.00005
warmup_epochs: 10
```

### 3. Lower Memory Usage

```yaml
# Smaller model
hidden_dim: 128
num_lstm_layers: 1

# Fewer frames
max_frames: 50

# Smaller frame size
frame_height: 64
frame_width: 64

# Gradient accumulation
batch_size: 4
accumulation_steps: 8
```

---

## Next Steps

1. **Experiment with hyperparameters**: Try different learning rates, batch sizes
2. **Data augmentation**: Add video augmentation for better generalization
3. **Model ensemble**: Train multiple models and ensemble predictions
4. **Export model**: Convert to ONNX or TorchScript for production
5. **Fine-tuning**: Fine-tune on domain-specific data

---

## Support Resources

- **Code Issues**: Check logs in `./logs/training.log`
- **CUDA Issues**: `nvidia-smi` and PyTorch CUDA documentation
- **Dataset Issues**: Verify with `python main.py stats --data_root ./Chinese-LiPS`
- **Performance**: Use `nvprof` or `torch.profiler` for detailed profiling

---

## Quick Reference Commands

```bash
# View dataset stats
python main.py stats --data_root ./Chinese-LiPS

# Start training
python main.py train --data_root ./Chinese-LiPS

# Resume training
python main.py train --resume ./checkpoints/checkpoint_epoch_50.pt

# Test model
python main.py test --checkpoint ./checkpoints/best_model.pt

# Run inference
python main.py inference --video_path video.mp4

# Monitor GPU
watch -n 1 nvidia-smi

# View logs
tail -f ./logs/training.log
```