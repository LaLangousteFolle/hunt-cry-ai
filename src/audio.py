import librosa
import numpy as np
import torch

TARGET_SR = 22050
TARGET_DURATION = 2.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)


def audio_to_mel(file_path: str, sr: int = TARGET_SR) -> torch.Tensor:
    y, _ = librosa.load(file_path, sr=sr)

    if len(y) < TARGET_SAMPLES:
        pad_width = TARGET_SAMPLES - len(y)
        y = np.pad(y, (0, pad_width), mode="constant")
    else:
        y = y[:TARGET_SAMPLES]

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    return torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)
