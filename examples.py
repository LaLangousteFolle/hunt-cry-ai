#!/usr/bin/env python
"""Example script showing how to use the Hunt Cry AI model."""

import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_training():
    """Example: How to train the model."""
    logger.info("\n" + "="*50)
    logger.info("EXAMPLE: Training")
    logger.info("="*50)
    
    logger.info("""
    To train the model:
    
    1. Prepare your data in data/labels.csv:
       filepath,class
       raw/audio1.wav,injured
       raw/audio2.wav,kill
       raw/audio3.wav,headshot
    
    2. Run the training script:
       python -m src.train
    
    3. Monitor training output for loss and accuracy
    
    4. Best model will be saved to models/hunt_cry_best.pt
    """)


def example_single_prediction():
    """Example: How to predict on a single audio file."""
    logger.info("\n" + "="*50)
    logger.info("EXAMPLE: Single Prediction")
    logger.info("="*50)
    
    logger.info("""
    To predict on a single audio file in Python:
    
    from src.predict import predict_one
    
    # Predict
    result = predict_one(
        audio_path="path/to/audio.wav",
        model_path="models/hunt_cry_best.pt",
        device="cuda"  # or "cpu"
    )
    
    # Access results
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Confidence: {result['predicted_confidence']:.1%}")
    print(f"All probabilities: {result['probabilities']}")
    
    # Or via command line:
    python -m src.predict your_audio.wav
    """)


def example_batch_prediction():
    """Example: How to predict on multiple files."""
    logger.info("\n" + "="*50)
    logger.info("EXAMPLE: Batch Prediction")
    logger.info("="*50)
    
    logger.info("""
    To predict on multiple audio files:
    
    from src.predict import predict_batch
    
    audio_files = [
        "audio1.wav",
        "audio2.wav",
        "audio3.wav",
    ]
    
    results = predict_batch(
        audio_files,
        model_path="models/hunt_cry_best.pt",
        device="cuda"
    )
    
    for result in results:
        if 'error' not in result:
            print(f"{result['audio_path']}: {result['predicted_class']}")
        else:
            print(f"{result['audio_path']}: ERROR - {result['error']}")
    """)


def example_model_usage():
    """Example: How to use the model directly."""
    logger.info("\n" + "="*50)
    logger.info("EXAMPLE: Direct Model Usage")
    logger.info("="*50)
    
    logger.info("""
    To use the model directly in Python:
    
    import torch
    from src.model import HuntCryClassifier
    from src.audio import audio_to_mel
    
    # Load model
    model = HuntCryClassifier()
    checkpoint = torch.load("models/hunt_cry_best.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Process audio
    mel_spec = audio_to_mel("audio.wav")
    mel_spec = mel_spec.unsqueeze(0)  # Add batch dim
    
    # Predict
    with torch.no_grad():
        logits = model(mel_spec)
        probs = torch.softmax(logits, dim=1)
        pred_idx = logits.argmax(1).item()
    
    # Results
    classes = {0: 'injured', 1: 'kill', 2: 'headshot'}
    print(f"Predicted: {classes[pred_idx]}")
    print(f"Confidence: {probs[0, pred_idx]:.1%}")
    """)


def example_dataset_usage():
    """Example: How to use the dataset."""
    logger.info("\n" + "="*50)
    logger.info("EXAMPLE: Dataset Usage")
    logger.info("="*50)
    
    logger.info("""
    To use the dataset:
    
    from src.dataset import HuntCryDataset
    from torch.utils.data import DataLoader
    
    # Load dataset
    dataset = HuntCryDataset(csv_path="data/labels.csv")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Iterate
    for mel_specs, labels in dataloader:
        print(f"Batch shape: {mel_specs.shape}")
        print(f"Labels: {labels}")
        # Your training code here
    """)


def example_configuration():
    """Example: How to configure training."""
    logger.info("\n" + "="*50)
    logger.info("EXAMPLE: Training Configuration")
    logger.info("="*50)
    
    logger.info("""
    To customize training, edit src/train.py:
    
    CONFIG = {
        "batch_size": 16,          # Larger for GPUs, smaller for CPUs
        "num_epochs": 30,          # More epochs for better accuracy
        "learning_rate": 1e-3,     # Lower for stability
        "train_split": 0.8,        # 80% train, 20% val
        "num_workers": 4,          # Match your CPU cores
        "seed": 42,                # For reproducibility
        "model_dir": "models",
        "csv_path": "data/labels.csv",
    }
    
    Tips:
    - Reduce batch_size if you get CUDA out of memory
    - Increase learning_rate if training is too slow
    - Decrease learning_rate if loss doesn't converge
    - Increase num_workers for faster data loading
    """)


def example_troubleshooting():
    """Example: Common issues and fixes."""
    logger.info("\n" + "="*50)
    logger.info("EXAMPLE: Troubleshooting")
    logger.info("="*50)
    
    logger.info("""
    Common Issues:
    
    1. CUDA out of memory
       - Reduce batch_size in CONFIG
       - Use device="cpu" if necessary
    
    2. FileNotFoundError: Audio file not found
       - Check that data/labels.csv paths are correct
       - Verify audio files exist in data/ directory
    
    3. CSV error: Missing columns
       - Ensure data/labels.csv has 'filepath' and 'class' columns
       - Check for proper formatting
    
    4. Model not found
       - Train the model first: python -m src.train
       - Check that models/hunt_cry_best.pt exists
    
    5. Low accuracy
       - Collect more training data
       - Try data augmentation
       - Increase training epochs
       - Check data quality and labels
    """)


def main():
    """Run all examples."""
    logger.info("\n" + "#"*50)
    logger.info("# Hunt Showdown Sound - Usage Examples")
    logger.info("#"*50)
    
    example_training()
    example_single_prediction()
    example_batch_prediction()
    example_model_usage()
    example_dataset_usage()
    example_configuration()
    example_troubleshooting()
    
    logger.info("\n" + "="*50)
    logger.info("For more details, see README.md")
    logger.info("="*50 + "\n")


if __name__ == "__main__":
    main()
