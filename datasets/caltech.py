import os
from PIL import Image
from torch.utils.data import Dataset

class Caltech101(Dataset):
    def __init__(self, root: str, split: str, transform=None):
        """
        root: data/caltech101
        split: 'train' or 'test'
        """
        self.transform = transform
        split_file = os.path.join(os.path.abspath(os.getcwd()), root,f"{split}.txt")
        with open(split_file, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
        self.samples = []
        for line in lines:
            # 格式：classname/imagename.jpg label
            path_rel, label = line.split()
            img_path = os.path.join(os.path.abspath(os.getcwd()), root, path_rel)
            self.samples.append((img_path, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label
