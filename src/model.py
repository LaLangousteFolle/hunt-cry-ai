import logging
import torch.nn as nn
import torch

logger = logging.getLogger(__name__)


class HuntCryClassifier(nn.Module):
    """
    CNN model for Hunt Showdown cry classification.
    
    Classifies audio into 3 categories:
    - Injured: Hunter is injured
    - Kill: Regular elimination
    - Headshot: Headshot elimination
    """
    
    def __init__(self, num_classes: int = 3, dropout_rate: float = 0.3):
        """
        Initialize the model.
        
        Args:
            num_classes: Number of output classes (default: 3)
            dropout_rate: Dropout probability for regularization
        """
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Feature extraction layers with BatchNorm and Dropout for better generalization
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.MaxPool2d(2),
            
            # Block 3 - Added for better feature learning
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.MaxPool2d(2),
        )

        # Classification layers
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes),
        )
        
        logger.info(f"Initialized HuntCryClassifier with {num_classes} classes")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape [batch_size, 1, 128, time_steps]
            
        Returns:
            Output logits of shape [batch_size, num_classes]
        """
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def get_parameter_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
