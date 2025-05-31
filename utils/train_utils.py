import random
import numpy as np
import torch
from tqdm import tqdm

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def accuracy(output, target, topk=(1,)):
    """
    计算 top-k 准确率
    output: [batch, num_classes]
    target: [batch]
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)  # [batch, maxk]
        pred = pred.t()                             # [maxk, batch]
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append((correct_k / batch_size).item() * 100.0)
        return res  # list of topk accuracies

def train_one_epoch(model, loader, criterion, optimizer, epoch, writer, device):
    model.train()
    running_loss = 0.0
    running_acc1 = 0.0
    total = 0

    loop = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch} [Train]")
    for i, (images, labels) in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        acc1, = accuracy(outputs, labels, topk=(1,))
        running_loss += loss.item() * bs
        running_acc1 += acc1 * bs
        total += bs

        loop.set_postfix(loss=running_loss/total, acc1=running_acc1/total)

    epoch_loss = running_loss / total
    epoch_acc  = running_acc1 / total
    writer.add_scalar('train/loss', epoch_loss, epoch)
    writer.add_scalar('train/acc1', epoch_acc, epoch)

def validate(model, loader, criterion, epoch, writer, device):
    model.eval()
    running_loss = 0.0
    running_acc1 = 0.0
    total = 0

    with torch.no_grad():
        loop = tqdm(loader, total=len(loader), desc=f"Epoch {epoch} [Val]  ")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            bs = labels.size(0)
            acc1, = accuracy(outputs, labels, topk=(1,))
            running_loss += loss.item() * bs
            running_acc1 += acc1 * bs
            total += bs
            loop.set_postfix(val_loss=running_loss/total, val_acc=running_acc1/total)

    epoch_loss = running_loss / total
    epoch_acc  = running_acc1 / total
    writer.add_scalar('val/loss', epoch_loss, epoch)
    writer.add_scalar('val/acc1', epoch_acc, epoch)
    return epoch_acc
