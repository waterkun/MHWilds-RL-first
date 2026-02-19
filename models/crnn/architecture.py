"""
CRNN (CNN + BiLSTM + CTC) for damage number recognition.
~850K parameters, replaces Tesseract OCR.

Input:  grayscale image (1, 32, W), height fixed at 32px, width varies
Output: per-timestep log-probabilities over 10 digits + CTC blank
"""

import torch
import torch.nn as nn


class DamageCRNN(nn.Module):
    # 0-9 digits + CTC blank (index 0)
    BLANK = 0
    NUM_CLASSES = 11  # blank + 10 digits

    def __init__(self, num_classes=11, rnn_hidden=128):
        super().__init__()
        self.num_classes = num_classes

        # CNN backbone: 5 conv blocks with asymmetric pooling
        # Height: 32 -> 16 -> 8 -> 4 -> 2 -> 1
        # Width:  W  -> W  -> W/2 -> W/2 -> W/4 -> W/4
        self.cnn = nn.Sequential(
            # Block 1: 1 -> 32
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),  # H/2, W unchanged

            # Block 2: 32 -> 64
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),  # H/4, W/2

            # Block 3: 64 -> 128
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),  # H/8, W/2

            # Block 4: 128 -> 256
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),  # H/16, W/4

            # Block 5: 256 -> 256
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),  # H/32=1, W/4
        )

        # RNN: BiLSTM
        # Input: (batch, 256, 1, W/4) -> squeeze H -> (batch, W/4, 256)
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=rnn_hidden,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        # Output head: BiLSTM output (2*hidden) -> num_classes
        self.head = nn.Linear(rnn_hidden * 2, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, x):
        """
        Args:
            x: (batch, 1, 32, W) float tensor, pixel values in [0, 1]
        Returns:
            log_probs: (T, batch, num_classes) for CTC loss
            T = W // 4
        """
        # CNN: (B, 1, 32, W) -> (B, 256, 1, W//4)
        conv = self.cnn(x)

        # Squeeze height dimension: (B, 256, 1, W//4) -> (B, 256, W//4)
        conv = conv.squeeze(2)

        # Permute to (B, W//4, 256) for RNN
        conv = conv.permute(0, 2, 1)

        # RNN: (B, T, 256) -> (B, T, 2*hidden)
        rnn_out, _ = self.rnn(conv)

        # Head: (B, T, 2*hidden) -> (B, T, num_classes)
        output = self.head(rnn_out)
        log_probs = self.log_softmax(output)

        # CTC loss expects (T, B, C)
        log_probs = log_probs.permute(1, 0, 2)

        return log_probs


if __name__ == "__main__":
    model = DamageCRNN()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Test with a sample input
    dummy = torch.randn(2, 1, 32, 64)  # batch=2, width=64
    out = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")  # (T=16, B=2, C=11)
