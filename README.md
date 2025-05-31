# Caltech-101 Fine-Tuning Project

This repository contains code to fine-tune ImageNet‐pretrained convolutional neural networks (ResNet-18, AlexNet) on the Caltech-101 dataset, compare against training from scratch, and visualize training/evaluation results.

---

## Table of Contents

- [Project Structure](#project-structure)  
- [Requirements](#requirements)  
- [Data Preparation](#data-preparation)  
- [Usage](#usage)  
  - [1. Generate Train/Test Splits](#1-generate-traintest-splits)  
  - [2. Fine-Tune a Pretrained Model](#2-fine-tune-a-pretrained-model)  
  - [3. Train from Scratch (Baseline)](#3-train-from-scratch-baseline)  
  - [4. Evaluate & Plot Curves](#4-evaluate--plot-curves)  
  - [5. Compare Multiple Runs](#5-compare-multiple-runs)  
- [Result Artifacts](#result-artifacts)  
- [Project Notes](#project-notes)  
- [License](#license)  

---

## Project Structure

```
caltech101-finetune/
├── README.md
├── requirements.txt
├── data/
│   └── caltech-101/
│       ├── <class1>/image.jpg
│       ├── <class2>/image.jpg
│       └── splits/
│           ├── train.txt
│           └── test.txt
├── datasets/
│   └── caltech.py            # Custom PyTorch Dataset for Caltech-101
├── models/
│   └── model_factory.py      # Functions to load & modify pretrained models
├── utils/
│   ├── train_utils.py        # Training/validation loops, metrics, seeding
│   └── transforms.py         # TorchVision transforms for train/val
├── scripts/
│   ├── make_split.py         # Script to generate train/test splits
│   ├── finetune.py           # Fine-tuning script (pretrained backbone + new head)
│   ├── scratch.py            # Baseline script (train from random init)
│   └── evaluate.py           # Evaluation & plotting (loss, accuracy, confusion)
└── outputs/
    ├── finetune/             # Checkpoints & TensorBoard logs for fine-tune runs
    ├── scratch/              # Checkpoints & logs for baseline runs
    └── eval/                 # Evaluation results (reports, plots, confusion matrices)
```

---

## Requirements

- Python 3.8+  
- PyTorch (≥1.8.0) with CUDA support  
- Torchvision  
- TensorBoard  
- NumPy  
- Pillow  
- scikit-learn  
- matplotlib  
- tqdm  

Install all dependencies via:

```bash
pip install -r requirements.txt
```

> **Note:** If you are using a Conda environment, you can install PyTorch + CUDA via:
> ```bash
> conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
> ```

---

## Data Preparation

1. **Download Caltech-101**  
   - Go to [Caltech-101 Dataset](https://data.caltech.edu/records/mzrjq-6wc02) and download `101_ObjectCategories.tar.gz`.  
   - Extract the archive. You should see a folder named `101_ObjectCategories` with 101 subfolders (one per class).

2. **Organize Directory**  
   - Rename or move the `101_ObjectCategories` folder to:  
     ```
     caltech101-finetune/data/caltech-101/
       ├── accordion/
       ├── airplanes/
       └── … (other classes)
     ```
   - Create a subdirectory `splits/` inside `data/caltech-101/`:
     ```
     caltech101-finetune/data/caltech-101/splits/
     ```

3. **Generate Train/Test Splits**  
   Run the provided split script to automatically generate `train.txt` and `test.txt`:

   ```bash
   cd caltech101-finetune
   python scripts/make_split.py
   ```

   - This will take the first 30 images of each class for training and put the rest into testing.  
   - The `train.txt` and `test.txt` files will be saved under `data/caltech-101/splits/`, each line formatted as:
     ```
     class_name/image_filename.jpg  label_index
     ```

---

## Usage

### 1. Generate Train/Test Splits

```bash
cd caltech101-finetune
python scripts/make_split.py
```

- Output:  
  - `data/caltech-101/splits/train.txt`  
  - `data/caltech-101/splits/test.txt`

### 2. Fine-Tune a Pretrained Model

```bash
python scripts/finetune.py   --data-dir    data/caltech-101   --model       resnet18   --pretrained    --epochs      30   --batch-size  32   --lr-base     1e-3   --lr-head     1e-2   --out-dir     outputs/finetune/30_32
```

- **Arguments**:  
  - `--data-dir`: Path to `caltech-101` (contains subfolders + `splits/`).  
  - `--model`: Choose `resnet18` or `alexnet`.  
  - `--pretrained`: If present, loads ImageNet weights.  
  - `--epochs`: Number of training epochs.  
  - `--batch-size`: Batch size (e.g., 32).  
  - `--lr-base`: Learning rate for backbone layers.  
  - `--lr-head`: Learning rate for newly added classification head.  
  - `--out-dir`: Directory to save TensorBoard logs and `best_model.pth`.  

TensorBoard logs will be written under `outputs/finetune/30_32/runs/`.  
The best-performing model checkpoint is saved to `outputs/finetune/30_32/best_model.pth`.

### 3. Train from Scratch (Baseline)

```bash
python scripts/scratch.py   --data-dir    data/caltech-101   --model       resnet18   --epochs      30   --batch-size  32   --lr-base     1e-2   --out-dir     outputs/scratch/30_32
```

- Same arguments as `finetune.py` except:
  - `--pretrained` is not set, so the model is randomly initialized.  
  - You can still specify `--lr-base` for the single learning rate.  

### 4. Evaluate & Plot Curves

```bash
python scripts/evaluate.py   --data-dir     data/caltech-101   --models       resnet18   --checkpoints  outputs/finetune/30_32/best_model.pth   --log-dirs     outputs/finetune/30_32/runs   --labels       "Finetune-30x32"   --batch-size   32   --output-dir   outputs/eval/finetune_30x32
```

- **This will**:  
  1. Load `best_model.pth` and run inference on the test set.  
  2. Generate a classification report (`.txt`) and confusion matrix (`.png`).  
  3. Load TensorBoard logs from `runs/` to extract and plot train/val loss & accuracy curves.  
  4. Save all outputs to `outputs/eval/finetune_30x32/`.

### 5. Compare Multiple Runs

To compare multiple hyperparameter settings on the same plot, pass multiple model/checkpoint/log-dir triplets:

```bash
python scripts/evaluate.py   --data-dir     data/caltech-101   --models       resnet18 resnet18 resnet18   --checkpoints  outputs/finetune/20_32/best_model.pth                  outputs/finetune/30_32/best_model.pth                  outputs/finetune/50_32/best_model.pth   --log-dirs     outputs/finetune/20_32/runs                  outputs/finetune/30_32/runs                  outputs/finetune/50_32/runs   --labels       "20_epochs" "30_epochs" "50_epochs"   --batch-size   32   --output-dir   outputs/eval/epoch_comparison
```

- **Output**:  
  - `compare_loss.png`: Overlaid train/val loss curves for each epoch setting.  
  - `compare_acc.png`: Overlaid train/val accuracy curves.  

Similarly, to compare different batch sizes at fixed epochs:

```bash
python scripts/evaluate.py   --data-dir     data/caltech-101   --models       resnet18 resnet18 resnet18   --checkpoints  outputs/finetune/30_16/best_model.pth                  outputs/finetune/30_32/best_model.pth                  outputs/finetune/30_64/best_model.pth   --log-dirs     outputs/finetune/30_16/runs                  outputs/finetune/30_32/runs                  outputs/finetune/30_64/runs   --labels       "BS16" "BS32" "BS64"   --batch-size   32   --output-dir   outputs/eval/bs_comparison
```

---

## Result Artifacts

After running the above commands, you will find:

- **Model Checkpoints**  
  - `outputs/finetune/<setting>/best_model.pth`  
  - `outputs/scratch/<setting>/best_model.pth`  

- **TensorBoard Logs**  
  - `outputs/finetune/<setting>/runs/`  
  - `outputs/scratch/<setting>/runs/`  

- **Evaluation Outputs** (e.g. `outputs/eval/finetune_30x32/`)  
  - `classification_report.txt`  
  - `confusion_matrix.png`  
  - `loss_curve.png` & `accuracy_curve.png` (for a single run)  
  - `compare_loss.png` & `compare_acc.png` (when comparing multiple runs)  

Use these artifacts to write your final report, embed plots, and share results.

---

## Project Notes

- All scripts assume you are running from the project root (`caltech101-finetune/`).  
- Adjust `--data-dir`, `--out-dir`, and hyperparameters (learning rates, batch size, epochs) as needed.  
- When evaluating multiple runs, ensure each `--log-dirs` contains a valid `runs/` subfolder with `events.out.tfevents.*`.

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
