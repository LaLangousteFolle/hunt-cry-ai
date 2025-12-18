from torch.utils.data import Dataset
import torch
import pandas as pd
from src.audio import audio_to_mel

CLASS2IDX = {"injured": 0, "kill": 1, "headshot": 2}


class HuntCryDataset(Dataset):
    def __init__(self, csv_path: str = "data/labels.csv"):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = f"data/{row['filepath']}"
        x = audio_to_mel(path)                     
        y = CLASS2IDX[row["class"]]
        return x, torch.tensor(y, dtype=torch.long)
