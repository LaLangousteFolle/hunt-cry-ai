import logging
import librosa
import numpy as np
import torch
import os

# Configuration logging
logger = logging.getLogger(__name__)

TARGET_SR = 22050
TARGET_DURATION = 2.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)


def audio_to_mel(file_path: str, sr: int = TARGET_SR) -> torch.Tensor:
    """
    Convert audio file to mel-spectrogram.
    
    Args:
        file_path: Path to audio file
        sr: Sample rate
        
    Returns:
        torch.Tensor: Mel-spectrogram as [1, n_mels, time_steps]
        
    Raises:
        FileNotFoundError: If audio file doesn't exist
        RuntimeError: If audio processing fails
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    try:
        y, _ = librosa.load(file_path, sr=sr)
        logger.debug(f"Loaded audio: {file_path}, shape: {y.shape}")
    except Exception as e:
        raise RuntimeError(f"Failed to load audio {file_path}: {str(e)}")

    # Pad or truncate to target length
    if len(y) < TARGET_SAMPLES:
        pad_width = TARGET_SAMPLES - len(y)
        y = np.pad(y, (0, pad_width), mode="reflect")  # Reflect padding better than constant
        logger.debug(f"Padded audio to {len(y)} samples")
    else:
        y = y[:TARGET_SAMPLES]
        logger.debug(f"Truncated audio to {TARGET_SAMPLES} samples")

    # Convert to mel-spectrogram
    try:
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
    except Exception as e:
        raise RuntimeError(f"Failed to compute mel-spectrogram: {str(e)}")

    return torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)
