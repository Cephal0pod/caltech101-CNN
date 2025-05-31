#!/usr/bin/env python
# Baseline: 从头训练，不加载预训练权重

import sys
from scripts.finetune import parse_args, main as finetune_main

if __name__ == '__main__':
    args = parse_args()
    args.pretrained = False
    # 改变输出目录，避免覆盖 finetune 结果
    args.out_dir = args.out_dir.replace('finetune', 'scratch')
    finetune_main()
