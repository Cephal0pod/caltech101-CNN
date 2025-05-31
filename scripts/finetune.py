import os
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.caltech import Caltech101
from models.model_factory import get_model
from utils.transforms import get_transforms
from utils.train_utils import train_one_epoch, validate, seed_everything


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir',   type=str, default='data\\caltech-101')
    p.add_argument('--model',      type=str, choices=['resnet18','alexnet'], default='resnet18')
    p.add_argument('--pretrained', action='store_true')
    p.add_argument('--epochs',     type=int, default=30)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr-base',    type=float, default=1e-3)
    p.add_argument('--lr-head',    type=float, default=1e-2)
    p.add_argument('--momentum',   type=float, default=0.9)
    p.add_argument('--wd',         type=float, default=1e-4)
    p.add_argument('--num-classes',type=int, default=101)
    p.add_argument('--out-dir',    type=str, default='outputs/finetune')
    return p.parse_args()

def main():
    from pathlib import Path
    args = parse_args()
    seed_everything(42)
    os.makedirs(args.out_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.out_dir, f"{args.epochs}_{args.batch_size}", 'runs'))

    # 数据加载
    train_tf, val_tf = get_transforms()
    train_ds = Caltech101(args.data_dir, 'train', transform=train_tf)
    val_ds   = Caltech101(args.data_dir, 'test',  transform=val_tf)

    print(f">>> train samples = {len(train_ds)}, val samples = {len(val_ds)}")
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("No data loaded. Check your split files and paths!")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 模型、优化器、损失
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(args.model, args.num_classes, pretrained=args.pretrained).to(device)
    print(f"Using device: {device}")

    # 区分 backbone/head 不同 lr
    head_params = []
    base_params = []
    for name, p in model.named_parameters():
        if 'fc' in name or 'classifier' in name:
            head_params.append(p)
        else:
            base_params.append(p)
    optimizer = optim.SGD([
        {'params': base_params, 'lr': args.lr_base},
        {'params': head_params, 'lr': args.lr_head},
    ], momentum=args.momentum, weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, epoch, writer, device)
        acc = validate(model, val_loader, criterion, epoch, writer, device)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(os.path.abspath(args.out_dir), f"{args.epochs}_{args.batch_size}", f'best_model.pth'))
    writer.close()

if __name__ == '__main__':
    main()
