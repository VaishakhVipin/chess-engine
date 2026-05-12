#!/usr/bin/env python3
"""UCI wrapper for Fischer hybrid chess engine."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import chess
from src.model import load_checkpoint
from src.nnue import SimpleNNUE
from src.search import HybridEngine

# Load models once
print("Loading Fischer engine...", file=sys.stderr)
policy_model, _ = load_checkpoint("checkpoints/best_model.pt")
eval_model = SimpleNNUE()
try:
    checkpoint = torch.load("checkpoints/nnue_eval.pt")
    eval_model.load_state_dict(checkpoint["model_state_dict"])
except:
    pass
eval_model.eval()
policy_model.eval()

engine = HybridEngine(policy_model, eval_model, device="cpu")
board = chess.Board()
depth = 3

def uci():
    """Handle UCI protocol."""
    print("id name FischerBot")
    print("id author Claude + Bobby Fischer games")
    print("option name depth type spin default 3 min 1 max 6")
    print("uciok")

def setoption(name, value):
    """Set UCI options."""
    global depth
    if name == "depth":
        try:
            depth = int(value)
            depth = max(1, min(6, depth))
        except:
            pass

def position(fen_or_startpos, moves=None):
    """Set board position."""
    global board
    if fen_or_startpos == "startpos":
        board = chess.Board()
    else:
        board = chess.Board(fen_or_startpos)

    if moves:
        for move_uci in moves:
            try:
                move = chess.Move.from_uci(move_uci)
                board.push(move)
            except:
                pass

def go(depth_override=None, wtime=None, btime=None, winc=None, binc=None):
    """Find best move."""
    global board, depth, engine

    search_depth = depth_override if depth_override else depth

    # Adjust depth based on time if provided
    if wtime or btime:
        our_time = wtime if board.turn == chess.WHITE else btime
        if our_time and our_time < 30000:
            search_depth = min(search_depth, 2)
        elif our_time and our_time < 120000:
            search_depth = min(search_depth, 3)

    move = engine.choose_move(board, depth=search_depth)

    if move:
        print(f"bestmove {move.uci()}")
    else:
        # No legal moves = stalemate/checkmate
        print("bestmove 0000")

def main():
    """Main UCI loop."""
    while True:
        try:
            line = input().strip()
        except EOFError:
            break

        if not line:
            continue

        parts = line.split()
        cmd = parts[0]

        if cmd == "uci":
            uci()

        elif cmd == "setoption":
            if len(parts) >= 4 and parts[1] == "name" and parts[3] == "value":
                name = parts[2]
                value = parts[4] if len(parts) > 4 else ""
                setoption(name, value)

        elif cmd == "isready":
            print("readyok")

        elif cmd == "position":
            if len(parts) < 2:
                continue
            if parts[1] == "startpos":
                position("startpos")
                if len(parts) > 2 and parts[2] == "moves":
                    position("startpos", parts[3:])
            elif parts[1] == "fen":
                fen = " ".join(parts[2:8])
                if len(parts) > 8 and parts[8] == "moves":
                    position(fen, parts[9:])
                else:
                    position(fen)

        elif cmd == "go":
            kwargs = {}
            i = 1
            while i < len(parts):
                if parts[i] == "depth" and i + 1 < len(parts):
                    kwargs["depth_override"] = int(parts[i + 1])
                    i += 2
                elif parts[i] == "wtime" and i + 1 < len(parts):
                    kwargs["wtime"] = int(parts[i + 1])
                    i += 2
                elif parts[i] == "btime" and i + 1 < len(parts):
                    kwargs["btime"] = int(parts[i + 1])
                    i += 2
                elif parts[i] == "winc" and i + 1 < len(parts):
                    kwargs["winc"] = int(parts[i + 1])
                    i += 2
                elif parts[i] == "binc" and i + 1 < len(parts):
                    kwargs["binc"] = int(parts[i + 1])
                    i += 2
                else:
                    i += 1
            go(**kwargs)

        elif cmd == "quit":
            break

if __name__ == "__main__":
    main()
