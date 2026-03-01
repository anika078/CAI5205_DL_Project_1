# README

------------------------------------------------------------

## 1. Project Overview

Project Title: Facial Expression Based Mood Recognition using MiniXception

Model Type:
MiniXception (lightweight variant of Xception with separable convolutions)

Objective:
Classification

Dataset Used:
FER2013 — https://www.kaggle.com/datasets/deadskull7/fer2013

Expected test evaluation for sanity check: Accuracy = ~0.6473

------------------------------------------------------------

## 2. Repository Structure

```
DL_Project_1/
  train.py              # Training script with argparse CLI
  test.py               # Evaluation script with argparse CLI
  requirements.txt      # Python dependencies
  README.txt            # This file
  src/
    __init__.py
    model.py            # MiniXception architecture
    dataloader.py       # FER2013 loading and augmentation
    utils.py            # Plotting, metrics, saving utilities
  models/               # Place checkpoint here
  data/                 # Place fer2013.csv here
```

------------------------------------------------------------

## 3. Dataset

### OPTION A — PUBLIC DATASET
Dataset Link:
https://www.kaggle.com/datasets/deadskull7/fer2013

Where to place the downloaded dataset:
```
data/
  fer2013.csv
```

Download fer2013.csv from the Kaggle link above. The download
gives you fer2013.csv.zip — unzip it first, then place the
extracted fer2013.csv into the data/ directory:

  unzip fer2013.csv.zip -d data/

The CSV contains a "Usage" column with built-in splits
(Training / PublicTest / PrivateTest), so no separate folder
structure is needed — the dataloader reads the single CSV and
splits automatically.

------------------------------------------------------------

## 4. Model Checkpoint

Box Link to Best Model Checkpoint:
https://usf.box.com/s/zha2ie72zb71ju9p20hzg0ehglub1nmt

Givenp'; access to:
yusun@usf.edu, kandiyana@usf.edu

Where to place the checkpoint after downloading:
```
models/
  best_minixception_final.keras
```

------------------------------------------------------------

## 5. Requirements (Dependencies)

Python Version:
3.10

How to install all dependencies:

Using pip:
```
pip install -r requirements.txt
```

Using conda:
```
conda create -n dlproj python=3.10
conda activate dlproj
pip install -r requirements.txt
```

------------------------------------------------------------

## 6. Running the Test Script

Command to run testing:
```
python test.py \
  --data data/fer2013.csv \
  --ckpt models/best_minixception_final.keras \
  --batch_size 64 \
  --out_dir outputs
```

------------------------------------------------------------

## 7. Running the Training Script

Command to run training:
```
python train.py \
  --data data/fer2013.csv \
  --epochs 200 \
  --batch_size 32 \
  --lr 0.001 \
  --dropout 0.1 \
  --l2_reg 1e-4 \
  --patience 20 \
  --seed 42 \
  --out_dir outputs \
  --ckpt_dir models
```

Optional arguments:
- `--seed` — Random seed for reproducibility (default: 42)
- `--patience` — Early stopping patience (default: 20)

------------------------------------------------------------

## 8. Submission Checklist

- [x] Dataset provided using Option A and instructions for placement included.
- [x] Model checkpoint linked and instructions for placement included.
- [x] `requirements.txt` generated and Python version specified.
- [x] Test command works.
- [x] Train command works.

------------------------------------------------------------
