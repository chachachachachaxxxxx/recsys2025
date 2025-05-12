# UniversalBehaviorTransformer (UBT)

UniversalBehaviorTransformer is an advanced approach for learning universal user behavior representations using transformer-based architecture and multi-task learning.

## Key Features

1. **Time-Aware Transformer Architecture**
   - Captures temporal patterns in user behavior
   - Handles variable-length sequences
   - Learns long-range dependencies

2. **Enhanced Feature Engineering**
   - Temporal features with time decay
   - Sequence features for event transitions
   - User-item and user-category interaction features

3. **Multi-Task Learning**
   - Simultaneous optimization of multiple downstream tasks
   - Shared representation learning
   - Task-specific prediction heads

4. **Contrastive Learning**
   - Improves representation robustness
   - Enhances generalization capability
   - Reduces overfitting

## Model Architecture

The model consists of several key components:

1. **UserBehaviorEncoder**
   - Event type embeddings
   - Item embeddings
   - Category embeddings
   - Time-aware transformer encoder

2. **TimeAwareTransformerEncoder**
   - Input projection
   - Time-aware position encoding
   - Multi-head self-attention
   - Feed-forward networks

3. **MultiTaskUserModel**
   - Shared user encoder
   - Task-specific prediction heads
   - Churn prediction
   - Category propensity
   - Product propensity

4. **ContrastiveUserEncoder**
   - Projection head
   - Contrastive loss computation
   - Positive/negative sample generation

## Usage

### Training

```bash
python -m baseline.universal_behavior_transformer.train \
    --data-dir /path/to/data \
    --save-dir /path/to/save \
    --device cuda
```

### Creating Embeddings

```bash
python -m baseline.universal_behavior_transformer.create_embeddings \
    --data-dir /path/to/data \
    --model-path /path/to/model.pt \
    --embeddings-dir /path/to/embeddings \
    --device cuda
```

## Hyperparameters

The model uses the following default hyperparameters:

```python
MODEL_CONFIG = {
    "hidden_dim": 512,
    "num_layers": 3,
    "num_heads": 8,
    "dropout": 0.1,
    "max_sequence_length": 100,
    "projection_dim": 128,
}

TRAIN_CONFIG = {
    "batch_size": 128,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "num_epochs": 10,
    "warmup_steps": 1000,
    "max_grad_norm": 1.0,
}
```

## Feature Engineering

The feature engineering process includes:

1. **Temporal Features**
   - Time decay
   - Time windows (1, 7, 30 days)
   - Cyclical features (hour, day of week)

2. **Sequence Features**
   - Event transitions
   - Transition probabilities
   - Sequence patterns

3. **Interaction Features**
   - User-item interactions
   - User-category interactions
   - Weighted interaction counts

## Performance

The model is designed to achieve better performance through:

1. **Better Temporal Understanding**
   - Captures time-dependent patterns
   - Handles seasonality
   - Models user behavior evolution

2. **Improved Sequence Modeling**
   - Learns complex event sequences
   - Captures long-term dependencies
   - Models user behavior transitions

3. **Enhanced Generalization**
   - Multi-task learning
   - Contrastive learning
   - Shared representation learning

## Requirements

- Python 3.8+
- PyTorch 1.8+
- NumPy
- Pandas
- scikit-learn

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{universal_behavior_transformer,
    title={UniversalBehaviorTransformer: A Transformer-based Approach for Universal User Behavior Modeling},
    author={Your Name},
    year={2024},
    publisher={GitHub},
    journal={GitHub repository},
    howpublished={\url{https://github.com/yourusername/recsys2025}}
}
``` 