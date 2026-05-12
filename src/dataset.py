import chess.pgn
import torch
from torch.utils.data import Dataset, random_split
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.encode import board_to_tensor, move_to_index


def parse_pgn(pgn_path: str, player_name: str = "Fischer") -> list[tuple[np.ndarray, int]]:
    """
    Parse a PGN file and extract board positions where the specified player moved.

    Returns a list of (board_tensor, move_index) tuples.
    """
    positions = []

    with open(pgn_path, encoding="utf-8", errors="ignore") as f:
        game_num = 0
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            game_num += 1
            if game_num % 100 == 0:
                print(f"Parsed {game_num} games, {len(positions)} positions...")

            try:
                # Determine which color is Fischer
                white = game.headers.get("White", "")
                black = game.headers.get("Black", "")

                fischer_is_white = player_name.lower() in white.lower()
                fischer_is_black = player_name.lower() in black.lower()

                if not (fischer_is_white or fischer_is_black):
                    continue

                # Skip short games
                moves = list(game.mainline_moves())
                if len(moves) < 10:
                    continue

                # Walk the game
                board = game.board()
                for move in moves:
                    # Check if Fischer is moving
                    fischer_moving = (fischer_is_white and board.turn == chess.WHITE) or (
                        fischer_is_black and board.turn == chess.BLACK
                    )

                    if fischer_moving:
                        flip = fischer_is_black
                        tensor = board_to_tensor(board, flip=flip)
                        move_idx = move_to_index(move, flip=flip)
                        positions.append((tensor, move_idx))

                    board.push(move)

            except Exception as e:
                print(f"Skipped game {game_num}: {e}")
                continue

    print(f"Total positions parsed: {len(positions)}")
    return positions


class FischerDataset(Dataset):
    """PyTorch Dataset wrapping a list of (tensor, move_idx) pairs."""

    def __init__(self, positions: list[tuple[np.ndarray, int]]):
        self.positions = positions

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tensor, move_idx = self.positions[idx]
        return torch.from_numpy(tensor), torch.tensor(move_idx, dtype=torch.long)


def get_loaders(pgn_path: str, batch_size: int = 64, player_name: str = "Fischer"):
    """
    Load PGN, parse positions, create train/val split, and return DataLoaders.
    """
    print(f"Parsing {pgn_path}...")
    positions = parse_pgn(pgn_path, player_name=player_name)

    dataset = FischerDataset(positions)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader
