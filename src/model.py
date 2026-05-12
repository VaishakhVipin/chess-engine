import torch
import torch.nn as nn
from pathlib import Path


class FischerNet(nn.Module):
    """
    Small CNN for predicting Fischer's moves given a board position.

    Input: (batch, 18, 8, 8) board tensor
    Output: (batch, 4096) logits over all possible moves
    """

    def __init__(self, in_channels: int = 18, conv_channels: int = 64, n_conv_layers: int = 4):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(),
        )

        self.body = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(conv_channels),
                    nn.ReLU(),
                )
                for _ in range(n_conv_layers)
            ]
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(conv_channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 4096),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.body(x)
        x = self.policy_head(x)
        return x


def save_checkpoint(model: FischerNet, optimizer, epoch: int, val_loss: float, val_acc: float, path: str):
    """Save model checkpoint with metadata."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_top1_acc": val_acc,
        "config": {
            "in_channels": 18,
            "conv_channels": 64,
            "n_conv_layers": 4,
        },
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(path: str, device: str = "cpu") -> tuple[FischerNet, dict]:
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model = FischerNet(**checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from {path}")
    return model, checkpoint
