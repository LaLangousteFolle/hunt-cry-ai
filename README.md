# Hunt Showdown Sound - AI Model

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

An AI-powered tool to classify Hunt Showdown game sounds using deep learning. Detects and categorizes different types of hunter eliminations.

## Features 🎯

- **Audio Classification**: Identifies three types of eliminations:
  - 🤕 **Injured**: Hunter is injured
  - ☠️ **Kill**: Regular elimination  
  - 💀 **Headshot**: Headshot elimination

- **Robust Architecture**: CNN with BatchNormalization and Dropout for better generalization
- **Comprehensive Logging**: Detailed training logs and predictions
- **Model Checkpointing**: Saves best model during training
- **Batch Predictions**: Process multiple audio files at once

## Architecture 🧠

```
Audio File
    ↓
Mel-Spectrogram (128×44, dB scale)
    ↓
CNN Feature Extractor (3 blocks)
  - Conv2d + BatchNorm + ReLU + Dropout + MaxPool
    ↓
Classifier
  - Linear layers with BatchNorm
    ↓
Class Probabilities (3 classes)
```

### Model Parameters
- **Input**: Mono audio at 22.05kHz, padded to 2 seconds
- **Feature**: 128-bin mel-spectrogram
- **Dropout**: 0.3 (regularization)
- **Learning Rate**: 1e-3 with ReduceLROnPlateau scheduler
- **Batch Size**: 16
- **Epochs**: 30 (with early stopping via scheduler)

## Installation 📦

### Requirements
- Python 3.8+
- CUDA 12.1+ (for GPU support, optional)

### Setup

```bash
# Clone repository
git clone https://github.com/LaLangousteFolle/hunt-cry-ai.git
cd hunt-cry-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage 🚀

### Training

```bash
# Prepare your data first
# Create data/labels.csv with columns: filepath, class
# Example:
#   filepath,class
#   audio1.wav,injured
#   audio2.wav,kill
#   audio3.wav,headshot

# Train the model
python -m src.train
```

The script will:
- Load audio files and create mel-spectrograms
- Split into 80% train / 20% validation
- Train for up to 30 epochs with learning rate scheduling
- Save the best model to `models/hunt_cry_best.pt`
- Log detailed metrics for each epoch

### Prediction

```bash
# Predict single audio file
python -m src.predict audio_file.wav

# Predict in Python
from src.predict import predict_one

results = predict_one("path/to/audio.wav")
print(f"Predicted: {results['predicted_class']}")
print(f"Confidence: {results['predicted_confidence']:.1%}")
print(f"All probabilities: {results['probabilities']}")
```

### Batch Predictions

```python
from src.predict import predict_batch

audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
results = predict_batch(audio_files)

for result in results:
    print(f"{result['audio_path']}: {result['predicted_class']}")
```

## Data Format 📋

### CSV Labels File (`data/labels.csv`)

Must contain two columns:

| Column | Description | Values |
|--------|-------------|--------|
| `filepath` | Relative path to audio file | e.g., `raw/audio1.wav` |
| `class` | Classification label | `injured`, `kill`, `headshot` |

Example:
```csv
filepath,class
raw/hunter_injured_1.wav,injured
raw/headshot_kill_1.wav,headshot
raw/regular_kill_1.wav,kill
```

### Supported Audio Formats
- `.wav` (WAV)
- `.mp3` (MP3)
- `.flac` (FLAC)
- Any format supported by `librosa`

## Project Structure 📁

```
hunt-cry-ai/
├── src/
│   ├── __init__.py
│   ├── audio.py         # Audio processing & mel-spectrogram generation
│   ├── dataset.py       # PyTorch Dataset class
│   ├── model.py         # CNN model architecture
│   ├── train.py         # Training loop with checkpointing
│   └── predict.py       # Inference functions
├── models/              # Saved model checkpoints
│   └── hunt_cry_best.pt
├── data/
│   ├── raw/            # Raw audio files
│   ├── labels.csv      # Audio labels
│   └── processed/      # Processed features (optional)
├── notebooks/          # Jupyter notebooks for exploration
├── requirements.txt    # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

## Configuration ⚙️

Edit training parameters in `src/train.py`:

```python
CONFIG = {
    "batch_size": 16,          # Batch size for training
    "num_epochs": 30,          # Maximum epochs
    "learning_rate": 1e-3,     # Initial learning rate
    "train_split": 0.8,        # Train/val split ratio
    "num_workers": 4,          # DataLoader workers
    "seed": 42,                # Random seed for reproducibility
    "model_dir": "models",
    "csv_path": "data/labels.csv",
}
```

## Performance 📊

Current results (WIP):

| Metric | Injured | Kill | Headshot | Overall |
|--------|---------|------|----------|----------|
| Accuracy | - | - | - | WIP |
| Precision | - | - | - | WIP |
| Recall | - | - | - | WIP |

## Logging 📝

All training and inference operations are logged with detailed information:

```
2025-12-18 20:00:00 - src.train - INFO - Using device: cuda
2025-12-18 20:00:01 - src.train - INFO - Loaded dataset from data/labels.csv with 150 samples
2025-12-18 20:00:02 - src.train - INFO - Epoch  1/30: loss=0.8234, val_acc=62.50% (25/40)
...
```

Logs are printed to console. To save to file:

```python
import logging
logging.basicConfig(filename='training.log', level=logging.INFO)
```

## Improvements Made ✨

✅ Added comprehensive error handling and validation  
✅ Implemented proper model checkpointing (saves best model)  
✅ Added logging throughout the pipeline  
✅ Improved model architecture (BatchNorm, Dropout, 3 conv blocks)  
✅ Enhanced DataLoader performance (num_workers, pin_memory)  
✅ Better audio padding (reflect instead of constant)  
✅ Learning rate scheduling (ReduceLROnPlateau)  
✅ Gradient clipping for stability  
✅ Batch prediction support  
✅ Cleaned up requirements.txt  
✅ Added comprehensive docstrings  

## Troubleshooting 🔧

### CUDA Out of Memory
```python
# Reduce batch size in src/train.py
CONFIG["batch_size"] = 8
```

### Audio Loading Errors
```bash
# Install additional audio libraries if needed
pip install librosa audioread
```

### Model Not Found
```bash
# Make sure the model directory exists
mkdir -p models/
```

## Contributing 🤝

Contributions are welcome! Please ensure:
- Code follows PEP 8 style
- All functions have docstrings
- Tests pass: `pytest`
- No console.log or print() statements (use logging instead)

## License 📄

MIT License - See [LICENSE](LICENSE) file for details

## Credits 🙏

**Special thanks to:**
- **Rexnor** for providing VODs/clips as training data!
  - [Twitch](https://twitch.tv/rexnor) | [YouTube](https://youtube.com/rexnor)

**Built by:** Nono - French CS Student

## Changelog 📋

### v2.0.0 (2025-12-18)
- ✅ Complete code refactoring
- ✅ Added error handling and logging
- ✅ Improved model architecture
- ✅ Implemented checkpointing
- ✅ Added batch prediction
- ✅ Cleaned dependencies

### v1.0.0 (Initial)
- Basic CNN model
- Simple training loop

---

**Status**: 🚀 Production-Ready Core  
**Next Goals**: Improve accuracy, add data augmentation, GPU optimization
