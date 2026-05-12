import torch
import torch.nn as nn
import numpy as np
from src.encode import board_to_tensor


class SimpleNNUE(nn.Module):
    """
    Simple NNUE-style evaluator.
    Input: (18, 8, 8) board tensor
    Output: scalar eval (white's advantage in centipawns)
    """

    def __init__(self):
        super().__init__()

        # Feature extraction (similar structure to policy net but lighter)
        self.features = nn.Sequential(
            nn.Conv2d(18, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # Evaluation head
        self.eval_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: (batch, 18, 8, 8) or (18, 8, 8). Returns eval score."""
        if x.dim() == 3:
            x = x.unsqueeze(0)

        features = self.features(x)
        eval_score = self.eval_head(features)
        return eval_score.squeeze()


def get_position_eval(model: SimpleNNUE, board, device: str = "cpu") -> float:
    """
    Evaluate a position using the NNUE model.
    Returns eval in centipawns from White's perspective.
    """
    model.eval()

    with torch.no_grad():
        tensor = board_to_tensor(board, flip=False)
        tensor = torch.from_numpy(tensor).to(device)

        eval_score = model(tensor).item()

        # Flip sign if black to move (from white's perspective)
        if board.turn == False:  # Black to move
            eval_score = -eval_score

        return eval_score


def create_eval_dataset(positions: list[tuple[np.ndarray, int]], outcomes: list[int]):
    """
    Create training data for NNUE from Fischer game positions.
    positions: list of (board_tensor, move_idx)
    outcomes: list of game outcomes (1=white win, 0=draw, -1=black win)
    """
    # Group positions by game (we'd need game boundaries for this)
    # For simplicity: use outcome as weak supervision for all positions in that game
    data = []

    for tensor, _ in positions:
        # Very weak supervision: use game outcome as target eval
        # This is crude but works for basic training
        data.append((tensor, outcomes[0]))  # Simplified

    return data
