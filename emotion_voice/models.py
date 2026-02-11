import torch
import torch.nn as nn

NUM_EMOTIONS = 8
EMOTION_ID_TO_NAME = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
]


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1, s=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EmotionCNN(nn.Module):
    """
    Simple CNN for log-mel spectrogram input: (B, 1, n_mels, T)
    """

    def __init__(self, n_mels: int = 64, num_classes: int = NUM_EMOTIONS):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32, k=3, p=1),
            nn.MaxPool2d((2, 2)),
            ConvBlock(32, 64, k=3, p=1),
            nn.MaxPool2d((2, 2)),
            ConvBlock(64, 128, k=3, p=1),
            nn.MaxPool2d((2, 2)),
            ConvBlock(128, 256, k=3, p=1),
            nn.MaxPool2d((2, 2)),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x
