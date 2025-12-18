import torch.nn as nn


class HuntCryClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),          
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),          
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten(),                  
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 3),             
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
