"""Configuration and constants for Hunt Showdown Sound AI."""

# Audio processing
AUDIO = {
    "sample_rate": 22050,          # Hz
    "target_duration": 2.0,        # seconds
    "n_mels": 128,                 # mel bins
    "n_fft": 2048,
    "hop_length": 512,
}

# Model architecture
MODEL = {
    "num_classes": 3,
    "dropout_rate": 0.3,
    "input_channels": 1,           # Mono audio
    "conv_filters": [16, 32, 64],
}

# Training
TRAINING = {
    "batch_size": 16,
    "num_epochs": 30,
    "learning_rate": 1e-3,
    "train_split": 0.8,
    "num_workers": 4,
    "seed": 42,
}

# Paths
PATHS = {
    "data_dir": "data",
    "model_dir": "models",
    "notebook_dir": "notebooks",
    "csv_path": "data/labels.csv",
    "best_model": "models/hunt_cry_best.pt",
}

# Classes
CLASSES = {
    "injured": 0,
    "kill": 1,
    "headshot": 2,
}

CLASS_NAMES = {v: k for k, v in CLASSES.items()}

# Class descriptions
CLASS_DESCRIPTIONS = {
    "injured": "Hunter is injured (cry of pain)",
    "kill": "Regular elimination (death cry)",
    "headshot": "Headshot elimination (specific sound)",
}
