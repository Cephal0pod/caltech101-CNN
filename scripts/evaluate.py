#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate multiple trained models and plot comparison of training/validation curves.
"""
import os
import sys
from pathlib import Path
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# Ensure project root is on sys.path
script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))

from datasets.caltech import Caltech101
from models.model_factory import get_model
from utils.transforms import get_transforms


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate models and compare training curves"
    )
    parser.add_argument('--data-dir', type=str, default='data/caltech-101',
                        help='Root directory of Caltech-101 dataset')
    parser.add_argument('--models', nargs='+', required=True,
                        help='List of model names corresponding to checkpoints and log dirs, e.g. resnet18')
    parser.add_argument('--checkpoints', nargs='+', required=True,
                        help='List of checkpoint paths for each model')
    parser.add_argument('--log-dirs', nargs='+', required=True,
                        help='List of TensorBoard log directories for each model')
    parser.add_argument('--labels', nargs='+', help='List of labels for plotting legends (same order)')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-classes', type=int, default=101)
    parser.add_argument('--output-dir', type=str, default='outputs/eval',
                        help='Directory to save evaluation outputs and plots')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    return parser.parse_args()


def plot_multiple_training_curves(log_dirs, labels, out_dir):
    plt.figure(figsize=(8,5))
    for i, log_dir in enumerate(log_dirs):
        label = labels[i] if labels and i < len(labels) else f'run-{i}'
        tb_path = Path(log_dir)
        # find event file or subdir
        runs = [p for p in tb_path.iterdir() if p.is_dir()]
        root = runs[0] if runs else tb_path
        ea = event_accumulator.EventAccumulator(str(root), size_guidance={event_accumulator.SCALARS:0})
        ea.Reload()
        train_loss = ea.Scalars('train/loss')
        val_loss   = ea.Scalars('val/loss')
        steps = [e.step for e in train_loss]
        loss_train = [e.value for e in train_loss]
        loss_val   = [e.value for e in val_loss]
        plt.plot(steps, loss_train, linestyle='-', label=f'{label} Train Loss')
        plt.plot(steps, loss_val,   linestyle='--', label=f'{label} Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training/Validation Loss Comparison')
    plt.legend()
    plt.grid(True)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(Path(out_dir)/'compare_loss.png')
    print(f"Saved combined loss plot to {out_dir}/compare_loss.png")

    plt.figure(figsize=(8,5))
    for i, log_dir in enumerate(log_dirs):
        label = labels[i] if labels and i < len(labels) else f'run-{i}'
        tb_path = Path(log_dir)
        runs = [p for p in tb_path.iterdir() if p.is_dir()]
        root = runs[0] if runs else tb_path
        ea = event_accumulator.EventAccumulator(str(root), size_guidance={event_accumulator.SCALARS:0})
        ea.Reload()
        train_acc = ea.Scalars('train/acc1')
        val_acc   = ea.Scalars('val/acc1')
        steps = [e.step for e in train_acc]
        acc_train = [e.value for e in train_acc]
        acc_val   = [e.value for e in val_acc]
        plt.plot(steps, acc_train, linestyle='-', label=f'{label} Train Acc')
        plt.plot(steps, acc_val,   linestyle='--', label=f'{label} Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training/Validation Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(Path(out_dir)/'compare_acc.png')
    print(f"Saved combined accuracy plot to {out_dir}/compare_acc.png")


def main():
    args = parse_args()
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Single-model evaluation (classification report, confusion matrix)
    for model_name, ckpt in zip(args.models, args.checkpoints):
        print(f"Evaluating {model_name} using checkpoint {ckpt}")
        _, eval_tf = get_transforms()
        test_ds = Caltech101(args.data_dir, 'test', transform=eval_tf)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                 shuffle=False, num_workers=4)
        # load model
        model = get_model(model_name, args.num_classes, pretrained=False)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.to(device).eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = outputs.topk(1, dim=1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.squeeze(1).cpu().numpy())
        report = classification_report(y_true, y_pred)
        fname = Path(args.output_dir)/f'{model_name}_classification_report.txt'
        with open(fname,'w') as f: f.write(report)
        print(f"Saved classification report for {model_name} to {fname}")
        # confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6,6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'{model_name} Confusion Matrix')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(Path(args.output_dir)/f'{model_name}_confusion_matrix.png')
        print(f"Saved confusion matrix for {model_name}")

    # Combined plots for all runs
    plot_multiple_training_curves(args.log_dirs, args.labels, args.output_dir)

if __name__ == '__main__':
    main()

