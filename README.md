# FischerBot — Neural Chess Engine

A hybrid chess engine that combines:
- **Bobby Fischer playstyle** via behavioral cloning on his game database
- **Neural position evaluation** (NNUE) for move assessment
- **Minimax search with alpha-beta pruning** for tactical depth
- **UCI protocol** for Lichess bot integration

## Features

- Plays in Fischer's style (aggressive, positional)
- Smart opening play (no reckless queen moves like Qg4)
- Penalizes weak opening moves to avoid "quirky dumb stuff"
- Adaptive time management (blitz → rapid → classical)
- Estimated rating: **1500–1700 Elo**
- Can be deployed as a Lichess bot

## Quick Start

### Training (if you have Fischer's games)

```bash
pip install -r requirements.txt
python src/train.py --pgn data/fischer.pgn --epochs 20 --batch-size 64
```

Expected: ~20 minutes on CPU, best model saved to `checkpoints/best_model.pt`.

### Playing Locally

```bash
python src/play.py --model checkpoints/best_model.pt --color black
```

Interactive game loop. Type UCI moves (e.g., `e2e4`).

### Lichess Deployment

See [DEPLOY_LICHESS.md](DEPLOY_LICHESS.md) for full integration steps.

**TL;DR:**
1. Clone [lichess-bot](https://github.com/lichess-org/lichess-bot)
2. Copy `fischer_engine.py`, `config.yml`, `src/`, and `checkpoints/` into the lichess-bot root
3. Replace `YOUR_LICHESS_TOKEN_HERE` in `config.yml` with your bot token
4. Run `python lichess-bot.py`

## Architecture

### Board Encoding (`src/encode.py`)
18-channel tensor representation (18×8×8):
- Channels 0–11: Piece positions (own + opponent)
- Channels 12–17: Castling rights, en passant, constant plane

### Policy Network (`src/model.py`)
FischerNet CNN (~8.5M parameters):
- Input: 18-channel board tensor
- Output: 4096 move logits (4096 = 64 × 64 board positions)
- Architecture: stem (conv 18→64) → 4× residual blocks (64→64) → head (linear to 4096)

### Position Evaluator (`src/nnue.py`)
SimpleNNUE neural network:
- Input: 18-channel board tensor
- Output: Scalar evaluation (centipawns from White's perspective)

### Hybrid Engine (`src/search.py`)
Combines three components:
1. **Policy ranking**: FischerNet predicts which moves Fischer would play
2. **NNUE evaluation**: Neural network evaluates position after each candidate move
3. **Material + activity bonuses**: Piece values, centralization, king safety, forcing moves

Minimax with alpha-beta pruning (depth 2–4 depending on time control).

**Opening Safeguards:**
- Heavily penalizes queen moves to aggressive squares (rank 4+ for White, 5- for Black)
- Completely avoids queen moves in first 3 moves
- Penalizes undefended queen moves
- Filters to top 12 moves in opening (first 10 moves) for speed
- Boosts king safety evaluation in opening phase

**Tactical Safeguards:**
- Heavily penalizes hanging pieces (attacked but undefended)
- Evaluates defenders before abandoning pieces
- Prevents material loss from blunders

Result: Plays solid opening principles, avoids tactical blunders like Qg4 or leaving pieces hanging.

### Training (`src/train.py`)
Behavioral cloning on 33,756 Fischer game positions:
- Loss: CrossEntropyLoss (4096 classes)
- Optimizer: Adam (lr=1e-3)
- Scheduler: ReduceLROnPlateau (patience=3)
- Expected accuracy: ~30–40% top-1, ~55–65% top-5

## Files

```
.
├── src/
│   ├── encode.py          # Board tensor encoding & move indexing
│   ├── dataset.py         # PGN parsing & PyTorch dataset
│   ├── model.py           # FischerNet (policy) & SimpleNNUE (eval)
│   ├── nnue.py            # NNUE evaluation wrapper
│   ├── search.py          # Hybrid engine (minimax + evaluation)
│   ├── train.py           # Training loop
│   ├── play.py            # Interactive play interface
│   └── __init__.py
├── fischer_engine.py      # UCI wrapper for Lichess bot
├── config.yml             # Lichess bot config (token = placeholder)
├── requirements.txt       # Dependencies
├── DEPLOY_LICHESS.md      # Lichess integration guide
└── README.md              # This file
```

## Dependencies

- `python-chess` — board representation & PGN parsing
- `torch` (CPU) — neural networks
- `numpy` — array operations
- `scikit-learn` — data utilities
- `berserk` — Lichess API (optional, for bot integration)

## Notes

- **Checkpoints** (`best_model.pt`, `nnue_eval.pt`) are gitignored; download or train locally
- **PGN data** (`data/fischer.pgn`) is gitignored; ~40,000 positions from Fischer's games
- **Token security**: `config.yml` has `token: YOUR_LICHESS_TOKEN_HERE` — replace with your actual bot token before deployment
- **Windows**: Use `fischer_engine.bat` wrapper for UCI protocol

## References

- Lichess API: https://lichess.org/api
- UCI protocol: https://en.wikipedia.org/wiki/Universal_Chess_Interface
- NNUE evaluation: https://www.chessprogramming.org/NNUE
- Bobby Fischer games: https://www.pgnmentor.com/files.html

---

Built for fun. Not SOTA, just Fischer-flavored. Enjoy! ♟️
