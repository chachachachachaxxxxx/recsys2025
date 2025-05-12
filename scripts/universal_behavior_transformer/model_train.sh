#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0  # 使用第一个GPU
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 创建保存目录
SAVE_DIR="exp/universal_behavior_transformer/$(date +%Y%m%d_%H%M%S)"
mkdir -p $SAVE_DIR

# 训练模型
python -m baseline.universal_behavior_transformer.train \
    --data-dir /home/xxx/data/ubc_data \
    --save-dir $SAVE_DIR \
    --device cuda \
    --batch-size 128 \
    --learning-rate 1e-4 \
    --weight-decay 0.01 \
    --num-epochs 10 \
    --warmup-steps 1000 \
    --max-grad-norm 1.0 \
    --hidden-dim 512 \
    --num-layers 3 \
    --num-heads 8 \
    --dropout 0.1 \
    --max-sequence-length 100 \
    --projection-dim 128 \
    --time-windows 1 7 30 \
    --top-n-items 10 \
    --top-n-categories 5 \
    --time-decay-factor 86400 \
    --log-interval 100 \
    --eval-interval 1 \
    --save-interval 1
