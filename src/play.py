import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import chess
import numpy as np
from src.model import load_checkpoint
from src.encode import board_to_tensor, index_to_move, get_legal_mask


def pick_move(model: torch.nn.Module, board: chess.Board, temperature: float = 1.0, device: str = "cpu"):
    """
    Use the model to pick a move for the current position.

    temperature=0: deterministic (argmax)
    temperature>0: sample from softmax(logits/temperature)
    """
    model.eval()

    with torch.no_grad():
        tensor = board_to_tensor(board, flip=False)
        tensor = torch.from_numpy(tensor).unsqueeze(0).to(device)

        logits = model(tensor).squeeze(0).cpu().numpy()

        # Mask illegal moves
        legal_mask = get_legal_mask(board, flip=False)
        logits[legal_mask == 0] = -1e9

        if temperature == 0:
            move_idx = logits.argmax()
        else:
            logits = logits / temperature
            logits = logits - logits.max()  # numerical stability
            probs = np.exp(logits)
            probs = probs * legal_mask
            probs = probs / probs.sum()

            move_idx = np.random.choice(4096, p=probs)

    move = index_to_move(move_idx, board, flip=False)
    return move


def main():
    parser = argparse.ArgumentParser(description="Play against Fischer model")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--color", type=str, default="white", choices=["white", "black"],
                        help="Color to play as")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on")

    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    model, _ = load_checkpoint(args.model, device=args.device)
    model.to(args.device)

    human_is_white = args.color.lower() == "white"
    fischer_is_white = not human_is_white

    board = chess.Board()

    print("\nChess position (type 'quit' to exit):")
    print(f"You play as {'White' if human_is_white else 'Black'}")
    print(f"Fischer plays as {'White' if fischer_is_white else 'Black'}\n")

    move_count = 0

    while not board.is_game_over():
        print(board)
        print()

        # Determine whose turn it is
        fischer_moving = (fischer_is_white and board.turn == chess.WHITE) or (
            fischer_is_white == False and board.turn == chess.BLACK
        )

        if fischer_moving:
            print("Fischer is thinking...")
            move = pick_move(model, board, temperature=args.temperature, device=args.device)
            print(f"Fischer plays: {board.san(move)}")
            board.push(move)
            move_count += 1

        else:
            while True:
                user_input = input("Your move (UCI format, e.g. e2e4): ").strip().lower()

                if user_input == "quit":
                    print("Exiting...")
                    return

                try:
                    move = board.parse_san(user_input) if len(user_input) <= 4 else board.parse_uci(user_input)
                    if move not in board.legal_moves:
                        print("Illegal move. Try again.")
                        continue
                    board.push(move)
                    move_count += 1
                    break

                except Exception as e:
                    print(f"Invalid move: {e}. Try again.")

        print()

    print(board)
    print()

    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        print(f"Checkmate! {winner} wins!")
    elif board.is_stalemate():
        print("Stalemate!")
    elif board.is_insufficient_material():
        print("Draw by insufficient material.")
    elif board.is_seventyfive_move_rule():
        print("Draw by 75-move rule.")
    elif board.is_fivefold_repetition():
        print("Draw by fivefold repetition.")
    else:
        print("Game over.")

    print(f"Total moves: {move_count}")


if __name__ == "__main__":
    main()
