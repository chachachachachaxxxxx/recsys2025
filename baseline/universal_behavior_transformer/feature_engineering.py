"""
Feature engineering module for UniversalBehaviorTransformer.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from .constants import EventType, FEATURE_CONFIG

class EnhancedFeaturesAggregator:
    def __init__(self, relevant_client_ids: np.ndarray):
        self.relevant_client_ids = relevant_client_ids
        self.features = {}
        
    def generate_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate temporal features from event timestamps."""
        current_time = df['timestamp'].max()
        
        # Time decay features
        df['time_decay'] = np.exp(
            -(current_time - df['timestamp']).dt.total_seconds() / 
            FEATURE_CONFIG['time_decay_factor']
        )
        
        # Time window features
        for window in FEATURE_CONFIG['time_windows']:
            df[f'last_{window}d'] = (
                (current_time - df['timestamp']).dt.total_seconds() <= 
                window * 24 * 3600
            )
        
        # Cyclical features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        return df
    
    def generate_sequence_features(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate sequence features from user events."""
        df = df.sort_values(['client_id', 'timestamp'])
        
        # Event transitions
        df['next_event'] = df.groupby('client_id')['event_type'].shift(-1)
        df['prev_event'] = df.groupby('client_id')['event_type'].shift(1)
        
        # Transition matrix
        transition_matrix = pd.crosstab(
            df['event_type'], 
            df['next_event'],
            normalize='index'
        )
        
        return df, transition_matrix
    
    def generate_interaction_features(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate user-item and user-category interaction features."""
        # User-item interactions
        user_item_interaction = df.groupby(['client_id', 'sku']).agg({
            'timestamp': 'count',
            'time_decay': 'sum'
        }).reset_index()
        user_item_interaction.columns = ['client_id', 'sku', 'interaction_count', 'weighted_interaction']
        
        # User-category interactions
        if 'category' in df.columns:
            user_category_interaction = df.groupby(['client_id', 'category']).agg({
                'timestamp': 'count',
                'time_decay': 'sum'
            }).reset_index()
            user_category_interaction.columns = ['client_id', 'category', 'category_count', 'weighted_category']
        else:
            user_category_interaction = pd.DataFrame()
            
        return user_item_interaction, user_category_interaction
    
    def aggregate_features(
        self, 
        event_type: EventType,
        df: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """Aggregate all features for a given event type."""
        # Generate temporal features
        df = self.generate_temporal_features(df)
        
        # Generate sequence features
        df, transition_matrix = self.generate_sequence_features(df)
        
        # Generate interaction features
        user_item_interaction, user_category_interaction = self.generate_interaction_features(df)
        
        # Aggregate features by client
        client_features = {}
        
        # Temporal features
        for window in FEATURE_CONFIG['time_windows']:
            window_features = df[df[f'last_{window}d']].groupby('client_id').agg({
                'time_decay': ['mean', 'sum', 'max'],
                'hour': ['mean', 'std'],
                'day_of_week': ['mean', 'std']
            })
            client_features[f'window_{window}d'] = window_features
        
        # Sequence features
        client_features['transition_matrix'] = transition_matrix
        
        # Interaction features
        client_features['user_item_interaction'] = user_item_interaction
        client_features['user_category_interaction'] = user_category_interaction
        
        return client_features
    
    def merge_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """Merge all features into final embeddings."""
        # Implementation will be added
        pass 