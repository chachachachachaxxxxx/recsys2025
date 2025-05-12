"""
Model architecture for UniversalBehaviorTransformer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from .constants import MODEL_CONFIG

class TimeAwareTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = MODEL_CONFIG['hidden_dim'],
        num_layers: int = MODEL_CONFIG['num_layers'],
        num_heads: int = MODEL_CONFIG['num_heads'],
        dropout: float = MODEL_CONFIG['dropout']
    ):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Time-aware position encoding
        self.time_embedding = nn.Linear(1, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Project input
        x = self.input_projection(x)
        
        # Add time encoding
        time_emb = self.time_embedding(timestamps.unsqueeze(-1))
        x = x + time_emb
        
        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=mask)
        
        return x

class UserBehaviorEncoder(nn.Module):
    def __init__(
        self,
        num_event_types: int,
        num_items: int,
        num_categories: int,
        hidden_dim: int = MODEL_CONFIG['hidden_dim']
    ):
        super().__init__()
        
        # Event type embedding
        self.event_embedding = nn.Embedding(num_event_types, hidden_dim)
        
        # Item embedding
        self.item_embedding = nn.Embedding(num_items, hidden_dim)
        
        # Category embedding
        self.category_embedding = nn.Embedding(num_categories, hidden_dim)
        
        # Transformer encoder
        self.transformer = TimeAwareTransformerEncoder(
            input_dim=hidden_dim * 3  # Concatenated embeddings
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(
        self,
        event_types: torch.Tensor,
        items: torch.Tensor,
        categories: torch.Tensor,
        timestamps: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Get embeddings
        event_emb = self.event_embedding(event_types)
        item_emb = self.item_embedding(items)
        category_emb = self.category_embedding(categories)
        
        # Concatenate embeddings
        x = torch.cat([event_emb, item_emb, category_emb], dim=-1)
        
        # Apply transformer
        x = self.transformer(x, timestamps, mask)
        
        # Project to output dimension
        x = self.output_projection(x)
        
        return x

class ContrastiveUserEncoder(nn.Module):
    def __init__(
        self,
        encoder: UserBehaviorEncoder,
        projection_dim: int = MODEL_CONFIG['projection_dim']
    ):
        super().__init__()
        self.encoder = encoder
        self.projection = nn.Sequential(
            nn.Linear(encoder.transformer.transformer.layers[-1].linear2.out_features, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
    def forward(
        self,
        anchor_events: Dict[str, torch.Tensor],
        positive_events: Dict[str, torch.Tensor],
        negative_events: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        # Encode different views
        anchor_emb = self.projection(self.encoder(**anchor_events))
        positive_emb = self.projection(self.encoder(**positive_events))
        negative_emb = self.projection(self.encoder(**negative_events))
        
        # Compute contrastive loss
        pos_sim = F.cosine_similarity(anchor_emb, positive_emb)
        neg_sim = F.cosine_similarity(anchor_emb, negative_emb)
        
        loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim)))
        
        return loss

class MultiTaskUserModel(nn.Module):
    def __init__(
        self,
        user_encoder: UserBehaviorEncoder,
        num_categories: int,
        num_products: int
    ):
        super().__init__()
        self.user_encoder = user_encoder
        
        # Task-specific heads
        self.churn_head = nn.Sequential(
            nn.Linear(user_encoder.output_projection.out_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.category_head = nn.Sequential(
            nn.Linear(user_encoder.output_projection.out_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_categories)
        )
        
        self.product_head = nn.Sequential(
            nn.Linear(user_encoder.output_projection.out_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_products)
        )
        
    def forward(
        self,
        event_types: torch.Tensor,
        items: torch.Tensor,
        categories: torch.Tensor,
        timestamps: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Get user representation
        user_emb = self.user_encoder(
            event_types=event_types,
            items=items,
            categories=categories,
            timestamps=timestamps,
            mask=mask
        )
        
        # Task-specific predictions
        predictions = {
            'churn': self.churn_head(user_emb),
            'category': self.category_head(user_emb),
            'product': self.product_head(user_emb)
        }
        
        return predictions 