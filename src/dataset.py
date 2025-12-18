import logging
import os
from torch.utils.data import Dataset
import torch
import pandas as pd
from src.audio import audio_to_mel

logger = logging.getLogger(__name__)

CLASS2IDX = {"injured": 0, "kill": 1, "headshot": 2}
IDX2CLASS = {v: k for k, v in CLASS2IDX.items()}


class HuntCryDataset(Dataset):
    """
    PyTorch Dataset for Hunt Showdown cry audio classification.
    
    Loads audio files and their labels from a CSV file.
    Each row should contain 'filepath' and 'class' columns.
    """
    
    def __init__(self, csv_path: str = "data/labels.csv"):
        """
        Initialize the dataset.
        
        Args:
            csv_path: Path to CSV file with 'filepath' and 'class' columns
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If CSV is empty or missing required columns
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        self.df = pd.read_csv(csv_path)
        logger.info(f"Loaded dataset from {csv_path} with {len(self.df)} samples")
        
        # Validate CSV structure
        required_cols = {'filepath', 'class'}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(f"CSV must contain columns: {required_cols}")
        
        if len(self.df) == 0:
            raise ValueError("CSV file is empty")
        
        # Check for invalid classes
        invalid_classes = set(self.df['class'].unique()) - set(CLASS2IDX.keys())
        if invalid_classes:
            logger.warning(f"Found invalid classes: {invalid_classes}")
        
        # Log class distribution
        class_dist = self.df['class'].value_counts()
        logger.info(f"Class distribution: {class_dist.to_dict()}")

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (mel_spectrogram, class_label)
            
        Raises:
            RuntimeError: If audio processing fails
        """
        row = self.df.iloc[idx]
        filepath = f"data/{row['filepath']}"
        
        try:
            x = audio_to_mel(filepath)
        except Exception as e:
            logger.error(f"Failed to load sample {idx}: {str(e)}")
            raise RuntimeError(f"Failed to load audio sample {idx}: {str(e)}")
        
        class_name = row["class"]
        if class_name not in CLASS2IDX:
            raise ValueError(f"Invalid class: {class_name}")
        
        y = CLASS2IDX[class_name]
        return x, torch.tensor(y, dtype=torch.long)
