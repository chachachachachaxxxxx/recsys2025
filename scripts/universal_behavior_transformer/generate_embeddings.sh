#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0  # 使用第一个GPU
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 设置路径
DATA_DIR="/home/xxx/data/ubc_data"
MODEL_PATH="exp/universal_behavior_transformer/best_model.pt"  # 使用最佳模型
EMBEDDINGS_DIR="exp/universal_behavior_transformer/embeddings/$(date +%Y%m%d_%H%M%S)"

# 创建embeddings保存目录
mkdir -p $EMBEDDINGS_DIR

# 生成embeddings
python -m baseline.universal_behavior_transformer.create_embeddings \
    --data-dir $DATA_DIR \
    --model-path $MODEL_PATH \
    --embeddings-dir $EMBEDDINGS_DIR \
    --device cuda \
    --batch-size 256 \
    --num-workers 4 \
    --max-sequence-length 100 \
    --time-windows 1 7 30 \
    --top-n-items 10 \
    --top-n-categories 5 \
    --time-decay-factor 86400
