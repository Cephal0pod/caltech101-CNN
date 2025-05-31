from torchvision import transforms

# ImageNet 上的均值和方差
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_transforms(input_size: int = 224):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(input_size * 256/224)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])
    return train_tf, val_tf
