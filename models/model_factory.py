import torch.nn as nn
import torchvision.models as tvm

def get_model(name: str, num_classes: int, pretrained: bool = True):
    """
    name: 'resnet18' or 'alexnet'
    """
    if name == 'resnet18':
        model = tvm.resnet18(pretrained=pretrained)
        in_feat = model.fc.in_features
        model.fc = nn.Linear(in_feat, num_classes)
    elif name == 'alexnet':
        model = tvm.alexnet(pretrained=pretrained)
        in_feat = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_feat, num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")
    return model
