import os
import random

def make_splits(data_dir: str, out_dir: str, train_per_class: int = 30, seed: int = 42):
    """
    data_dir: Caltech-101 根目录，里面每个子文件夹是一个类别
    out_dir : 将生成 train.txt 和 test.txt 的目录（如 data/caltech101/splits）
    train_per_class: 每类训练样本数
    """
    random.seed(seed)
    classes = sorted([d for d in os.listdir(data_dir)
                      if os.path.isdir(os.path.join(data_dir, d))])
    os.makedirs(out_dir, exist_ok=True)

    # 类别名到编号
    cls2idx = {cls: i for i, cls in enumerate(classes)}

    train_lines = []
    test_lines  = []

    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        imgs = [f for f in os.listdir(cls_dir)
                if f.lower().endswith(('.jpg','.jpeg','.png'))]
        random.shuffle(imgs)

        # 前 train_per_class 张为 train，其余为 test
        for i, fn in enumerate(imgs):
            rel_path = f"{cls}/{fn}"
            label    = cls2idx[cls]
            line     = f"{rel_path} {label}\n"
            if i < train_per_class:
                train_lines.append(line)
            else:
                test_lines.append(line)

    # 写文件
    with open(os.path.join(out_dir, 'train.txt'), 'w') as f:
        f.writelines(train_lines)
    with open(os.path.join(out_dir, 'test.txt'), 'w') as f:
        f.writelines(test_lines)

if __name__ == '__main__':
    # 请根据实际路径调整
    data_root = os.path.abspath(os.path.join(__file__, '..', '..', 'data', 'caltech-101'))
    make_splits(data_root, data_root, train_per_class=30)
    print("✂️  划分完成，文件生成于:", data_root)
