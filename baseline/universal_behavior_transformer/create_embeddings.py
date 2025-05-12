"""
Script to create user embeddings using UniversalBehaviorTransformer.
"""
import argparse
import logging
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from typing import List, Tuple

from data_utils.utils import load_with_properties
from data_utils.data_dir import DataDir
from .model import UserBehaviorEncoder
from .feature_engineering import EnhancedFeaturesAggregator
from .constants import EventType, MODEL_CONFIG

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_relevant_clients_ids(input_dir: Path) -> np.ndarray:
    """Load relevant client IDs from input directory."""
    return np.load(input_dir / "relevant_clients.npy")

def save_embeddings(
    embeddings_dir: Path,
    embeddings: np.ndarray,
    client_ids: np.ndarray
):
    """Save embeddings in competition entry format."""
    logger.info("Saving embeddings")
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_dir / "embeddings.npy", embeddings)
    np.save(embeddings_dir / "client_ids.npy", client_ids)

def create_embeddings(
    data_dir: DataDir,
    model_path: Path,
    relevant_client_ids: np.ndarray,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate user embeddings using trained UniversalBehaviorTransformer model."""
    # Load model
    model = UserBehaviorEncoder(
        num_event_types=len(EventType),
        num_items=100000,  # This should be determined from data
        num_categories=1000  # This should be determined from data
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Initialize feature aggregator
    aggregator = EnhancedFeaturesAggregator(relevant_client_ids)
    
    # Process each event type
    all_features = {}
    for event_type in EventType:
        logger.info(f"Processing {event_type.value} events")
        
        # Load data
        event_df = load_with_properties(data_dir=data_dir, event_type=event_type.value)
        event_df["timestamp"] = pd.to_datetime(event_df.timestamp)
        
        # Generate features
        features = aggregator.aggregate_features(event_type, event_df)
        all_features[event_type] = features
    
    # Merge features
    client_ids, features = aggregator.merge_features()
    
    # Generate embeddings
    with torch.no_grad():
        embeddings = model(
            event_types=torch.tensor(features['event_types'], device=device),
            items=torch.tensor(features['items'], device=device),
            categories=torch.tensor(features['categories'], device=device),
            timestamps=torch.tensor(features['timestamps'], device=device)
        )
        embeddings = embeddings.cpu().numpy()
    
    return client_ids, embeddings

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory with input and target data"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        required=True,
        help="Directory where to store generated embeddings"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for computation"
    )
    return parser

def main(params):
    # Setup
    data_dir = DataDir(Path(params.data_dir))
    model_path = Path(params.model_path)
    embeddings_dir = Path(params.embeddings_dir)
    device = torch.device(params.device)
    
    # Load relevant clients
    relevant_client_ids = load_relevant_clients_ids(input_dir=data_dir.input_dir)
    
    # Generate embeddings
    client_ids, embeddings = create_embeddings(
        data_dir=data_dir,
        model_path=model_path,
        relevant_client_ids=relevant_client_ids,
        device=device
    )
    
    # Save embeddings
    save_embeddings(
        client_ids=client_ids,
        embeddings=embeddings,
        embeddings_dir=embeddings_dir
    )

if __name__ == "__main__":
    parser = get_parser()
    params = parser.parse_args()
    main(params=params) 