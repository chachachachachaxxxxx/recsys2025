python -m training_pipeline.train \
  --data-dir /home/xxx/data/ubc_data \
  --embeddings-dir /home/xxx/code/recsys2025/exp/embeddings1 \
  --tasks churn propensity_category propensity_sku \
  --log-name baseline_test_v1 \
  --accelerator gpu \
  --devices 0 \
  --disable-relevant-clients-check