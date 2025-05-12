"""
Training script for UniversalBehaviorTransformer.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional
import logging
from pathlib import Path
import argparse
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .model import UserBehaviorEncoder, MultiTaskUserModel, ContrastiveUserEncoder
from .constants import TRAIN_CONFIG, MODEL_CONFIG, FEATURE_CONFIG
from .dataset import create_dataloaders
from data_utils.data_dir import DataDir

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Trainer:
    def __init__(
        self,
        model: MultiTaskUserModel,
        contrastive_encoder: ContrastiveUserEncoder,
        device: torch.device,
        save_dir: Path
    ):
        self.model = model
        self.contrastive_encoder = contrastive_encoder
        self.device = device
        self.save_dir = save_dir
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=TRAIN_CONFIG['learning_rate'],
            weight_decay=TRAIN_CONFIG['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=TRAIN_CONFIG['num_epochs']
        )
        
        # Loss functions
        self.loss_fns = {
            'churn': nn.BCELoss(),
            'category': nn.CrossEntropyLoss(),
            'product': nn.CrossEntropyLoss()
        }
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        task_losses = {task: 0 for task in self.loss_fns.keys()}
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            predictions = self.model(
                event_types=batch['event_types'],
                items=batch['items'],
                categories=batch['categories'],
                timestamps=batch['timestamps'],
                mask=batch['mask']
            )
            
            # Compute task losses
            losses = {}
            for task, pred in predictions.items():
                if task in self.loss_fns:
                    losses[task] = self.loss_fns[task](pred, batch[f'{task}_labels'])
                    task_losses[task] += losses[task].item()
            
            # Compute contrastive loss
            contrastive_loss = self.contrastive_encoder(
                anchor_events=batch['anchor_events'],
                positive_events=batch['positive_events'],
                negative_events=batch['negative_events']
            )
            
            # Total loss
            loss = sum(losses.values()) + 0.1 * contrastive_loss
            total_loss += loss.item()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                TRAIN_CONFIG['max_grad_norm']
            )
            
            self.optimizer.step()
            
            # Logging
            if batch_idx % 100 == 0:
                logger.info(
                    f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)}] '
                    f'Loss: {loss.item():.4f}'
                )
        
        # Update learning rate
        self.scheduler.step()
        
        # Compute average losses
        avg_losses = {
            task: loss / len(train_loader)
            for task, loss in task_losses.items()
        }
        avg_losses['total'] = total_loss / len(train_loader)
        
        return avg_losses
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        task_losses = {task: 0 for task in self.loss_fns.keys()}
        
        for batch in val_loader:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            predictions = self.model(
                event_types=batch['event_types'],
                items=batch['items'],
                categories=batch['categories'],
                timestamps=batch['timestamps'],
                mask=batch['mask']
            )
            
            # Compute task losses
            for task, pred in predictions.items():
                if task in self.loss_fns:
                    loss = self.loss_fns[task](pred, batch[f'{task}_labels'])
                    task_losses[task] += loss.item()
                    total_loss += loss.item()
        
        # Compute average losses
        avg_losses = {
            task: loss / len(val_loader)
            for task, loss in task_losses.items()
        }
        avg_losses['total'] = total_loss / len(val_loader)
        
        return avg_losses
    
    def save_checkpoint(self, epoch: int, val_metrics: Dict[str, float]):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_metrics': val_metrics
        }
        
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f'Saved checkpoint to {checkpoint_path}')
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = TRAIN_CONFIG['num_epochs']
    ):
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            logger.info(
                f'Train Epoch: {epoch} '
                f'Losses: {train_metrics}'
            )
            
            # Validation
            val_metrics = self.validate(val_loader)
            logger.info(
                f'Validation Epoch: {epoch} '
                f'Losses: {val_metrics}'
            )
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics)
            
            # Save best model
            if val_metrics['total'] < best_val_loss:
                best_val_loss = val_metrics['total']
                best_model_path = self.save_dir / 'best_model.pt'
                torch.save(self.model.state_dict(), best_model_path)
                logger.info(f'Saved best model to {best_model_path}')

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory with input and target data"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Directory where to save model checkpoints"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for computation"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=TRAIN_CONFIG['batch_size'],
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=TRAIN_CONFIG['learning_rate'],
        help="Learning rate"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=TRAIN_CONFIG['weight_decay'],
        help="Weight decay"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=TRAIN_CONFIG['num_epochs'],
        help="Number of training epochs"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=TRAIN_CONFIG['warmup_steps'],
        help="Number of warmup steps"
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=TRAIN_CONFIG['max_grad_norm'],
        help="Maximum gradient norm"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=MODEL_CONFIG['hidden_dim'],
        help="Hidden dimension size"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=MODEL_CONFIG['num_layers'],
        help="Number of transformer layers"
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=MODEL_CONFIG['num_heads'],
        help="Number of attention heads"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=MODEL_CONFIG['dropout'],
        help="Dropout rate"
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=MODEL_CONFIG['max_sequence_length'],
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--projection-dim",
        type=int,
        default=MODEL_CONFIG['projection_dim'],
        help="Projection dimension"
    )
    parser.add_argument(
        "--time-windows",
        nargs="+",
        type=int,
        default=FEATURE_CONFIG['time_windows'],
        help="Time windows for feature generation"
    )
    parser.add_argument(
        "--top-n-items",
        type=int,
        default=FEATURE_CONFIG['top_n_items'],
        help="Number of top items to consider"
    )
    parser.add_argument(
        "--top-n-categories",
        type=int,
        default=FEATURE_CONFIG['top_n_categories'],
        help="Number of top categories to consider"
    )
    parser.add_argument(
        "--time-decay-factor",
        type=int,
        default=FEATURE_CONFIG['time_decay_factor'],
        help="Time decay factor in seconds"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Logging interval"
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=1,
        help="Evaluation interval in epochs"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1,
        help="Model saving interval in epochs"
    )
    return parser

def main(params):
    # 设置设备
    device = torch.device(params.device)
    
    # 创建保存目录
    save_dir = Path(params.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化数据目录
    data_dir = DataDir(Path(params.data_dir))

    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=params.batch_size,
        num_workers=4,  # 可以根据需要调整
        max_sequence_length=params.max_sequence_length,
        time_windows=params.time_windows,
        top_n_items=params.top_n_items,
        top_n_categories=params.top_n_categories,
        time_decay_factor=params.time_decay_factor
    )
   
    # 获取数据集的维度信息
    num_items = 100000  # 这个值应该从数据中获取
    num_categories = 1000  # 这个值应该从数据中获取
    num_event_types = 5  # 根据EventType枚举
    
    # 创建模型
    user_encoder = UserBehaviorEncoder(
        num_event_types=num_event_types,
        num_items=num_items,
        num_categories=num_categories,
        hidden_dim=params.hidden_dim
    )
    
    model = MultiTaskUserModel(
        user_encoder=user_encoder,
        num_categories=100,  # 根据propensity_category.npy
        num_products=100  # 根据propensity_sku.npy
    )
    
    contrastive_encoder = ContrastiveUserEncoder(
        encoder=user_encoder,
        projection_dim=params.projection_dim
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        contrastive_encoder=contrastive_encoder,
        device=device,
        save_dir=save_dir
    )
    
    # 训练模型
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=params.num_epochs
    )

if __name__ == "__main__":
    parser = get_parser()
    params = parser.parse_args()
    main(params) 