"""
Dataset classes for UniversalBehaviorTransformer.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import sys
import os
import pandas as pd

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_utils.data_dir import DataDir
from .event_type import EventType

logger = logging.getLogger(__name__)

class UserBehaviorDataset(Dataset):
    def __init__(
        self,
        data_dir: DataDir,
        split: str = "train",
        max_sequence_length: int = 100,
    ):
        """
        Args:
            data_dir (DataDir): 数据目录
            split (str): 数据集划分，可选 "train" 或 "val"
            max_sequence_length (int): 最大序列长度
        """
        self.data_dir = data_dir
        self.split = split
        self.max_sequence_length = max_sequence_length
        self.load_data()
        
    def load_data(self):
        """加载所有必要的数据"""
        # 加载事件数据
        event_files = {
            'product_buy': self.data_dir.input_dir / 'product_buy.parquet',
            'add_to_cart': self.data_dir.input_dir / 'add_to_cart.parquet',
            'remove_from_cart': self.data_dir.input_dir / 'remove_from_cart.parquet',
            'page_visit': self.data_dir.input_dir / 'page_visit.parquet',
            'search_query': self.data_dir.input_dir / 'search_query.parquet'
        }
        
        # 读取所有事件数据
        events = []
        for event_type, file_path in event_files.items():
            if file_path.exists():
                df = pd.read_parquet(file_path)
                df['event_type'] = getattr(EventType, event_type.upper()).value
                events.append(df)
        
        # 合并所有事件
        self.events = pd.concat(events, ignore_index=True)
        self.events = self.events.sort_values('timestamp')
        
        # 加载目标数据
        if self.split == "train":
            target_file = self.data_dir.target_dir / 'train_target.parquet'
        else:
            target_file = self.data_dir.target_dir / 'validation_target.parquet'
            
        self.target_df = pd.read_parquet(target_file)
        
        # 加载相关客户端ID
        self.relevant_clients = np.load(self.data_dir.input_dir / 'relevant_clients.npy')
        
        # 创建客户端ID到索引的映射
        self.client_to_idx = {client_id: idx for idx, client_id in enumerate(self.relevant_clients)}
        
    def __len__(self) -> int:
        return len(self.relevant_clients)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        client_id = self.relevant_clients[idx]
        
        # 获取该客户端的所有事件
        client_events = self.events[self.events['client_id'] == client_id]
        
        # 按时间排序
        client_events = client_events.sort_values('timestamp')
        
        # 截断序列
        if len(client_events) > self.max_sequence_length:
            client_events = client_events.iloc[-self.max_sequence_length:]
            
        # 创建序列张量
        sequence = torch.zeros(self.max_sequence_length, 4)  # [event_type, item_id, category_id, timestamp]
        sequence[:len(client_events)] = torch.from_numpy(client_events[['event_type', 'sku', 'category_id', 'timestamp']].values)
        
        # 创建掩码
        mask = torch.zeros(self.max_sequence_length, dtype=torch.bool)
        mask[:len(client_events)] = True
        
        # 获取目标值
        client_target = self.target_df[self.target_df['client_id'] == client_id]
        churn_target = torch.tensor(1.0 if not client_target.empty else 0.0, dtype=torch.float)
        
        return {
            'sequence': sequence,
            'mask': mask,
            'churn_target': churn_target
        }

def create_dataloaders(
    data_dir: DataDir,
    batch_size: int = 128,
    num_workers: int = 4,
    max_sequence_length: int = 100,
    time_windows: List[int] = [1, 7, 30],
    top_n_items: int = 10,
    top_n_categories: int = 5,
    time_decay_factor: int = 86400
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据加载器
    
    Args:
        data_dir: 数据目录
        batch_size: 批次大小
        num_workers: 数据加载的工作进程数
        max_sequence_length: 最大序列长度
        time_windows: 时间窗口列表
        top_n_items: 考虑的前N个商品
        top_n_categories: 考虑的前N个类别
        time_decay_factor: 时间衰减因子
        
    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
    """
    # 创建训练集
    train_dataset = UserBehaviorDataset(
        data_dir=data_dir,
        split="train",
        max_sequence_length=max_sequence_length
    )
    
    # 创建验证集
    val_dataset = UserBehaviorDataset(
        data_dir=data_dir,
        split="val",
        max_sequence_length=max_sequence_length
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader 