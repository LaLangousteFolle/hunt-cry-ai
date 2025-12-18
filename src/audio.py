import librosa
import numpy as np
import torch

def audio_to_mel(file_path, sr=22050):
    y, _ = librosa.load(file_path, sr=sr, duration=2.0)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
