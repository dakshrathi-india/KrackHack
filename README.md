# 🏜️ KrackHack 3.0 — Off-road Semantic Scene Segmentation (AI/ML Track)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![CUDA](https://img.shields.io/badge/CUDA-12.x-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Status-Active-success)

A reproducible semantic segmentation pipeline for **autonomous off-road navigation** in desert environments, developed for **KrackHack 3.0 (AI/ML Track)**.

---

## 🚀 Live Deployment

**Try our model in action:**  
🔗 **[https://krackhack-inf.streamlit.app/](https://krackhack-inf.streamlit.app/)**

Upload an image and get real-time semantic segmentation predictions!

---

## 📌 Table of Contents

* [Project Overview](#project-overview)
* [Repository Structure](#repository-structure)
* [Dataset Setup](#dataset-setup)
* [Environment Setup](#environment-setup)
* [How to Run](#how-to-run)
  * [Training](#training)
  * [Testing / Evaluation](#testing--evaluation)
  * [Using the Interactive Demo](#using-the-interactive-demo)
* [Scripts Folder Organization](#scripts-folder-organization)
* [Pretrained Models](#pretrained-models)
* [Reports & Documentation](#reports--documentation)
* [Tech Stack](#tech-stack)
* [Team](#team)
* [Notes on Reproducibility](#notes-on-reproducibility)

---

## 🧠 Project Overview

The objective of this project is to build a **robust multiclass semantic segmentation model** for **off-road desert scenes**, capable of generalizing from **synthetic training data** to **unseen test environments**.

The development followed a failure-driven progression:

* CNN baselines (DeepLabV3+)
* Aggressive domain randomization
* Stabilization via SWA
* Transformer-based SegFormer architecture (final approach)

All experiments, results, and reasoning are documented in the reports included in this repository.

---

## 📁 Repository Structure

```
KrackHack/
├── README.md                          # Original README (legacy, for reference)
├── PIPELINE_MODULES_README.md         # Technical documentation of modules
├── requirements.txt                   # Python dependencies
├── run.txt                            # CUDA install commands
├── pipeline.ipynb                     # Original notebook (NOT REQUIRED - for reference only)
│
├── app.py                             # Streamlit deployment app (interactive demo)
│
├── scripts/                           # All Python modules organized in one folder
│   ├── 01_imports_setup.py            # 1. Imports & Environment Setup
│   ├── 02_global_config.py            # 2. Global Configuration
│   ├── 03_class_id_mapping.py         # 3. Class ID Mapping
│   ├── 04_dataset_class.py            # 4. Dataset Class
│   ├── 05_data_augmentation.py        # 5. Data Augmentation
│   ├── 06_dataloaders.py              # 6. DataLoaders
│   ├── 07_model_definition.py         # 7. Model Definition (SegFormer)
│   ├── 08_loss_functions.py           # 8. Loss Functions (Dice + Focal)
│   ├── 09_metrics.py                  # 9. Metrics (IoU)
│   ├── 10_training_loop.py            # 10. Training Loop
│   ├── 11_validation_loop.py          # 11. Validation Loop
│   ├── 12_training_execution.py       # 12. Training Execution (Main)
│   ├── 13_visualization_inference.py  # 13. Visualization & Inference
│   ├── 14a_test_dataset_loading.py    # 14A. Test Dataset Loading
│   ├── 14b_test_model_loading.py      # 14B. Test Model Loading
│   ├── 14c_test_evaluation.py         # 14C. Test Evaluation
│   ├── 14d_test_visualization.py      # 14D. Test Visualization
│   ├── main_integration.py            # Complete training script (recommended entry point)
│   └── test_integration.py            # Complete testing script
│
├── dataset/                           # Training + validation dataset 
│   ├── train/
│   │   ├── Color_Images/
│   │   └── Segmentation/
│   └── val/
│       ├── Color_Images/
│       └── Segmentation/
│
├── testing_dataset/                   # Blind test dataset 
│   ├── Color_Images/
│   └── Segmentation/
│
├── checkpoints/                       # Saved models (auto-created)
│   ├── best_model.pth
│   └── final_model.pth
│
├── training_curves.png                # Training plots
└── inference_results.png              # Qualitative results
```

---

## 📦 Dataset Setup

### Training & Validation Dataset

Download and extract **into `/dataset`**:

🔗 [Offroad_Segmentation_Training_Dataset.zip](https://storage.googleapis.com/duality-public-share/Hackathons/Duality%20Hackathon/Offroad_Segmentation_Training_Dataset.zip)

Expected structure:

```
dataset/
├── train/
│   ├── Color_Images/      (< 1000 images)
│   └── Segmentation/      (< 1000 masks)
└── val/
    ├── Color_Images/      (< 200 images)
    └── Segmentation/      (< 200 masks)
```

### Testing Dataset

Download and extract **into `/testing_dataset`**:

Same link as above (contains both training and test splits).

Expected structure:

```
testing_dataset/
├── Color_Images/          (blind test set)
└── Segmentation/          (ground truth)
```

---

## 🛠️ Environment Setup

### 1. Clone or Download Repository

```bash
git clone https://github.com/InfinityxR9/KrackHack.git

# Verify scripts folder exists
ls scripts/  # Should show all Python modules
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install PyTorch with CUDA Support (CRITICAL)

> ⚠️ **Important:** After step 3, you MUST run the CUDA installation commands from `run.txt`

```bash
# Check run.txt for the exact command matching your CUDA version
# Example for CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify installation:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

---

## ▶️ How to Run

### Training

#### Option A: Recommended — Use Integration Script

```bash
cd scripts
python main_integration.py
```

**This script:**
- ✅ Creates dataloaders
- ✅ Initializes SegFormer model
- ✅ Runs training with warmup + cosine scheduling
- ✅ Saves best model checkpoint
- ✅ Plots training curves

**Expected output:**
```
============================================================
SegFormer Semantic Segmentation Pipeline
============================================================

[1/4] Creating dataloaders...
Train: 934 samples | Val: 206 samples

[2/4] Initializing model...
Model parameters: 27,546,912

[3/4] Setting up loss and metric...

[4/4] Starting training...
Epoch 1/40 | Train Loss: 1.2345 | Val Loss: 1.0123 | Val mIoU: 0.5234 | LR: 2.00e-05
...
Training complete! Best mIoU: 0.7456
============================================================
```

#### Option B: Manual Step-by-Step (For Debugging)

If you want to run components individually from Python:

```python
import sys
sys.path.insert(0, 'scripts')

# 1. Setup
from imports_setup import device
from global_config import NUM_CLASSES, EPOCHS, ENCODER_LR, DECODER_LR, WEIGHT_DECAY, WARMUP_EPOCHS, CLASS_IDS, SAVE_DIR

# 2. Load data
from dataloaders import create_dataloaders
train_loader, val_loader = create_dataloaders()

# 3. Create model
from model_definition import SegFormerWrapper
model = SegFormerWrapper("nvidia/segformer-b1-finetuned-ade-512-512", NUM_CLASSES).to(device)

# 4. Create loss and metric
from loss_functions import DiceFocalLoss
from metrics import MulticlassIoU
criterion = DiceFocalLoss(NUM_CLASSES)
metric = MulticlassIoU(NUM_CLASSES)

# 5. Train
from training_execution import run_training
history, best_miou = run_training(
    model, train_loader, val_loader, criterion, metric, device,
    EPOCHS, ENCODER_LR, DECODER_LR, WEIGHT_DECAY, WARMUP_EPOCHS,
    CLASS_IDS, SAVE_DIR
)
```

---

### Testing / Evaluation

#### Option A: Recommended — Use Test Integration Script

```bash
cd scripts
python test_integration.py
```

**This script:**
- ✅ Loads the best trained model
- ✅ Evaluates on test dataset
- ✅ Computes per-class and mean IoU
- ✅ Visualizes predictions

**Expected output:**
```
============================================================
SegFormer Semantic Segmentation - Testing Pipeline
============================================================

[1/3] Loading test dataset...
Loaded TEST dataset with 156 samples

[2/3] Loading trained model...
Loaded best SegFormer model for testing

[3/3] Evaluating on test set...
Evaluating TEST dataset: 100%|████████| 40/40 [2:34<00:00, 3.85s/batch]

============================================================
TEST DATASET RESULTS
============================================================
Class   100: IoU = 0.6234
Class   200: IoU = 0.7456
Class   300: IoU = 0.5123
...
Class 10000: IoU = 0.4567
----
Overall TEST mIoU (valid classes only): 0.6234
============================================================
```

#### Option B: Using Evaluation Components Directly

```python
import sys
sys.path.insert(0, 'scripts')

from test_dataset_loading import create_test_loader
from test_model_loading import load_model_for_testing
from test_evaluation import evaluate_on_test
from imports_setup import device
from global_config import NUM_CLASSES

test_loader, test_dataset = create_test_loader("testing_dataset")
model = load_model_for_testing(device, "checkpoints/best_model.pth")
iou_per_class, test_miou = evaluate_on_test(model, test_loader, NUM_CLASSES, device)

print(f"Test mIoU: {test_miou:.4f}")
```

---

### Using the Interactive Demo

#### Live Version (No Installation Needed)

🔗 **Visit:** [https://krackhack-inf.streamlit.app/](https://krackhack-inf.streamlit.app/)

Upload an off-road image and get instant segmentation results with class predictions and confidence visualizations.

#### Running Locally

If you want to run the Streamlit app on your machine:

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

**Features:**
- 📤 Upload or capture images
- 🎯 Real-time segmentation
- 🎨 Color-coded class predictions
- 📊 Per-class confidence scores
- 💾 Download results as PNG

---

## 📚 Scripts Folder Organization

All Python modules are organized in the **`scripts/`** folder for clean project structure:

```
scripts/
├── 01_imports_setup.py             # Device & imports
├── 02_global_config.py             # Hyperparameters & paths
├── 03_class_id_mapping.py          # Class remapping utilities
├── 04_dataset_class.py             # Dataset class
├── 05_data_augmentation.py         # Augmentations
├── 06_dataloaders.py               # DataLoaders
├── 07_model_definition.py          # SegFormer model
├── 08_loss_functions.py            # Loss functions
├── 09_metrics.py                   # Metrics
├── 10_training_loop.py             # Training loop
├── 11_validation_loop.py           # Validation loop
├── 12_training_execution.py        # Full training pipeline
├── 13_visualization_inference.py   # Inference utilities
├── 14a_test_dataset_loading.py     # Test data loader
├── 14b_test_model_loading.py       # Model loader
├── 14c_test_evaluation.py          # Evaluation metrics
├── 14d_test_visualization.py       # Test visualization
├── main_integration.py             # ⭐ Main training entry point
└── test_integration.py             # ⭐ Main testing entry point

app.py                              # Streamlit web app (in root folder)
```

**To run scripts:** Navigate to the `scripts/` folder first:

```bash
cd scripts
python main_integration.py
```

---

## 🧩 Pretrained Models

Final trained checkpoints available here:

🔗 [Google Drive - Trained Models](https://drive.google.com/drive/folders/1cNMqY7EgFQZIi8m4-r8dFjYLib9pa6Nn)

**To use:**

```bash
# Download best_model.pth and place in checkpoints/
python test_integration.py
```

No retraining needed!

---

## 📄 Reports & Documentation

All technical details, architectural decisions, and analyses:

* 📘 **Mid Submission Report**  
  `mid_submission_krackhack.pdf`

* 📕 **Final Report**  
  `FINAL_REPORT_KRACKHACK.pdf`

* 📖 **Module Documentation**  
  `PIPELINE_MODULES_README.md`

---

## ⚙️ Tech Stack

| Component | Details |
|-----------|---------|
| **Language** | Python 3.9+ |
| **Framework** | PyTorch 2.x |
| **Model** | SegFormer (MiT-B1) |
| **Pre-training** | ADE20K |
| **Loss** | Dice + Focal |
| **Optimizer** | AdamW |
| **Augmentation** | Albumentations |
| **Scheduler** | Warmup + Cosine Decay |
| **Deployment** | Streamlit |
| **Hardware** | NVIDIA RTX 4060 (8GB VRAM) |

**Key Training Specs:**
- Input: 512×512 RGB images
- Classes: 10 (original IDs: 100, 200, 300, 500, 550, 600, 700, 800, 7100, 10000)
- Batch Size: 4
- Epochs: 40
- Encoder LR: 6e-5
- Decoder LR: 6e-4
- Mixed Precision: ✅ Enabled (AMP)

---

## 📊 Class Definitions

| ID | Class | Color |
|----|-------|-------|
| 100 | Class 0 | Dark Red |
| 200 | Class 1 | Green |
| 300 | Class 2 | Dark Blue |
| 500 | Class 3 | Yellow |
| 550 | Class 4 | Cyan |
| 600 | Class 5 | Magenta |
| 700 | Class 6 | Bright Green |
| 800 | Class 7 | Blue |
| 7100 | Class 8 | Yellow |
| 10000 | Class 9 | Cyan |


---

## ♻️ Notes on Reproducibility

### What We Guarantee

- ✅ Fixed random seeds (where applicable)
- ✅ No test-set leakage during training
- ✅ Validation used strictly for model selection
- ✅ Blind test dataset evaluated post-training only
- ✅ All checkpoints, plots, and metrics documented
- ✅ Modular code for easy experimentation
- ✅ Complete training pipeline in single command

### To Reproduce Results

1. Download datasets (links above)
2. Install dependencies (`requirements.txt` + `run.txt`)
3. Run `python main_integration.py`
4. Compare plots to `training_curves.png`
5. Run `python test_integration.py` for blind test evaluation

### If Something Doesn't Work

<details>
<summary><b>Scripts Not Found / Import Errors</b></summary>

Make sure you're in the correct directory:

```bash
# Navigate to scripts folder
cd scripts

# Then run the script
python main_integration.py
```

If running from parent directory, ensure `scripts/` is in your Python path.

</details>

<details>
<summary><b>CUDA/GPU Issues</b></summary>

```bash
# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch from run.txt
```

</details>

<details>
<summary><b>Out of Memory (OOM)</b></summary>

Edit `scripts/02_global_config.py`:
```python
BATCH_SIZE = 2  # Reduce from 4
SEGFORMER_VARIANT = "nvidia/segformer-b1-finetuned-ade-512-512"  # Use B1 instead of B2
```

</details>

<details>
<summary><b>Dataset Not Found</b></summary>

Verify paths in `scripts/02_global_config.py`:
```python
DATA_ROOT = "dataset"  # Should be in repo root (one level up from scripts)
```

Check directory structure from repo root:
```bash
ls dataset/train/Color_Images/  # Should show images
```

</details>

<details>
<summary><b>Model Checkpoint Not Found</b></summary>

Download from Google Drive link above, place in `checkpoints/` folder at repo root:
```
checkpoints/
├── best_model.pth
└── final_model.pth
```

</details>

---

## 🚀 Quick Start Checklist

- [ ] Clone repo
- [ ] Create virtual environment (`.venv`)
- [ ] Install `requirements.txt`
- [ ] Install PyTorch from `run.txt`
- [ ] Download datasets (training + testing)
- [ ] `cd scripts` and run `python main_integration.py` to train
- [ ] `cd scripts` and run `python test_integration.py` to evaluate
- [ ] Visit [Streamlit app](https://krackhack-inf.streamlit.app/) for live demo

---

