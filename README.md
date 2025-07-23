---

# Auto AIOps: LADI Image Classification Project

## Overview
This project provides tools and scripts for multi-label and single-label image classification using deep learning, focused on the LADI dataset. It includes:
- Training scripts for EfficientNet-B3 (multi-label) and CNN (single-label) models
- Data preprocessing and augmentation
- Model evaluation and checkpointing
- GPU compatibility checks

## Project Structure
```
Auto_aiops/
  ├── automation/           # (Reserved for automation scripts)
  ├── dashboard/            # (Reserved for dashboard/visualization)
  ├── data/                 # (Place for data-related utilities)
  ├── genai/                # (Reserved for generative AI tools)
  ├── main.py/              # (Main entry point, if any)
  ├── models/               # (Model definitions or checkpoints)
  ├── notebooks/            # (Jupyter notebooks for experiments)
  ├── scripts/              # Training and utility scripts
  │     ├── train_ladi_effnet.py   # Multi-label EfficientNet-B3 trainer
  │     ├── train_ladi_cnn.py      # Single-label CNN trainer
  │     └── test.py                # GPU/torch environment test
  ├── utils/                # (Utility functions)
  ├── best_effnet_b3_multilabel.pth   # Saved model weights
  ├── best_effnet_ladi_multilabel.pth # Saved model weights
  └── best_effnet_model.pth           # Saved model weights
```

## Dataset
- **LADI Dataset**: Download from [here](https://ladi.s3.amazonaws.com/ladi_v2_resized.tar.gz)
- Place the extracted images in a directory (e.g., `D:/images`), and ensure the CSV label file is at `D:/images/v2/ladi_v2_labels_train_full_resized.csv` (or update the paths in the scripts accordingly).

## Requirements
Install dependencies with pip:
```bash
pip install torch torchvision timm pandas scikit-learn pillow tqdm numpy
```

- Python 3.7+
- CUDA-enabled GPU recommended for training

## Usage

### 1. Test GPU/torch setup
```bash
python scripts/test.py
```

### 2. Train EfficientNet-B3 (Multi-label)
```bash
python scripts/train_ladi_effnet.py
```
- Edits may be needed to adjust `csv_path` and `images_root` in the script.
- Model checkpoints will be saved as `best_effnet_b3_multilabel.pth`.

### 3. Train Simple CNN (Single-label)
```bash
python scripts/train_ladi_cnn.py
```
- Edits may be needed to adjust `csv_path` and `images_root` in the script.

## Notes
- The scripts expect the dataset to be organized as described above.
- For custom experiments, modify the scripts in `scripts/` or add new notebooks in `notebooks/`.

## License
This project is for research and educational purposes. Dataset usage is subject to its own license.

---

Let me know if you want to add or change anything!
