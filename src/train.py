import logging
import os
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from datetime import datetime

from src.dataset import HuntCryDataset
from src.model import HuntCryClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "batch_size": 16,
    "num_epochs": 30,
    "learning_rate": 1e-3,
    "train_split": 0.8,
    "num_workers": 4,
    "seed": 42,
    "model_dir": "models",
    "csv_path": "data/labels.csv",
}


def setup_device() -> str:
    """
    Setup training device.
    
    Returns:
        Device string ('cuda' or 'cpu')
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def create_dataloaders(config: dict) -> tuple:
    """
    Create train and validation dataloaders.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    logger.info("Creating dataset...")
    dataset = HuntCryDataset(config["csv_path"])
    
    n_total = len(dataset)
    n_train = int(config["train_split"] * n_total)
    n_val = n_total - n_train
    
    logger.info(f"Dataset split: {n_train} train, {n_val} validation")
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(config["seed"])
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=False,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )
    
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    return train_loader, val_loader


def train_epoch(model: nn.Module, train_loader: DataLoader, criterion: nn.Module,
                optimizer: optim.Optimizer, device: str) -> float:
    """
    Train for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Training device
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model: nn.Module, val_loader: DataLoader, device: str) -> tuple:
    """
    Validate the model.
    
    Args:
        model: PyTorch model
        val_loader: Validation dataloader
        device: Training device
        
    Returns:
        Tuple of (accuracy, correct_samples, total_samples)
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    
    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int,
                   accuracy: float, config: dict, is_best: bool = False) -> str:
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        accuracy: Validation accuracy
        config: Configuration dictionary
        is_best: Whether this is the best model so far
        
    Returns:
        Path to saved checkpoint
    """
    os.makedirs(config["model_dir"], exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "accuracy": accuracy,
        "config": config,
    }
    
    if is_best:
        filepath = os.path.join(config["model_dir"], "hunt_cry_best.pt")
        torch.save(checkpoint, filepath)
        logger.info(f"✓ Saved best model: {filepath}")
    else:
        filepath = os.path.join(config["model_dir"], f"hunt_cry_epoch_{epoch:02d}.pt")
        torch.save(checkpoint, filepath)
        logger.debug(f"Saved checkpoint: {filepath}")
    
    return filepath


def load_checkpoint(filepath: str, model: nn.Module, optimizer: optim.Optimizer = None,
                   device: str = "cpu") -> dict:
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint
        model: Model to load weights into
        optimizer: Optimizer to load state (optional)
        device: Device to load to
        
    Returns:
        Checkpoint dictionary
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    logger.info(f"Loaded checkpoint from {filepath} (epoch {checkpoint['epoch']})")
    return checkpoint


def main():
    """
    Main training loop.
    """
    # Setup
    torch.manual_seed(CONFIG["seed"])
    device = setup_device()
    logger.info(f"Configuration: {CONFIG}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(CONFIG)
    
    # Initialize model
    logger.info("Initializing model...")
    model = HuntCryClassifier(num_classes=3, dropout_rate=0.3).to(device)
    param_count = model.get_parameter_count()
    logger.info(f"Model parameters: {param_count:,}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    # Training loop
    best_val_acc = 0.0
    logger.info("Starting training...")
    
    for epoch in range(CONFIG["num_epochs"]):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_acc, correct, total = validate(model, val_loader, device)
        
        # Logging
        logger.info(
            f"Epoch {epoch+1:2d}/{CONFIG['num_epochs']}: "
            f"loss={train_loss:.4f}, val_acc={val_acc:.2%} ({correct}/{total})"
        )
        
        # Save checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch+1, val_acc, CONFIG, is_best=True)
        
        # Learning rate scheduling
        scheduler.step(val_acc)
    
    logger.info(f"\n✓ Training completed! Best validation accuracy: {best_val_acc:.2%}")
    logger.info(f"Best model saved to: {CONFIG['model_dir']}/hunt_cry_best.pt")


if __name__ == "__main__":
    main()
