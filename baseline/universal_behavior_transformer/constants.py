"""
Constants for UniversalBehaviorTransformer implementation.
"""
from enum import Enum
from typing import Dict, List

class EventType(Enum):
    PRODUCT_BUY = "product_buy"
    ADD_TO_CART = "add_to_cart"
    REMOVE_FROM_CART = "remove_from_cart"
    PAGE_VISIT = "page_visit"
    SEARCH_QUERY = "search_query"

# Event type to columns mapping
EVENT_TYPE_TO_COLUMNS: Dict[EventType, List[str]] = {
    EventType.PRODUCT_BUY: ["sku", "category", "price"],
    EventType.ADD_TO_CART: ["sku", "category", "price"],
    EventType.REMOVE_FROM_CART: ["sku", "category", "price"],
    EventType.PAGE_VISIT: ["url"],
    EventType.SEARCH_QUERY: ["query"],
}

# Model hyperparameters
MODEL_CONFIG = {
    "hidden_dim": 512,
    "num_layers": 3,
    "num_heads": 8,
    "dropout": 0.1,
    "max_sequence_length": 100,
    "projection_dim": 128,
}

# Training hyperparameters
TRAIN_CONFIG = {
    "batch_size": 128,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "num_epochs": 10,
    "warmup_steps": 1000,
    "max_grad_norm": 1.0,
}

# Feature engineering parameters
FEATURE_CONFIG = {
    "time_windows": [1, 7, 30],  # days
    "top_n_items": 10,
    "top_n_categories": 5,
    "time_decay_factor": 24 * 3600,  # seconds
} 