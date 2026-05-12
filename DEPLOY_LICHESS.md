# FischerBot — Lichess Deployment Guide

## ✅ Setup Complete!

Your bot token is already upgraded and configured.

## Installation

### 1. Install Official Lichess Bot

```bash
git clone https://github.com/lichess-org/lichess-bot
cd lichess-bot
pip install -r requirements.txt
```

### 2. Copy Fischer Engine Files

Copy these files from your chess-engine folder into lichess-bot:

```bash
# Copy to lichess-bot root:
- fischer_engine.py
- config.yml
- checkpoints/best_model.pt
- checkpoints/nnue_eval.pt
- src/  (entire folder)
```

### 3. Update config.yml

The `config.yml` is already set up, but verify:

```yaml
token: YOUR_LICHESS_TOKEN_HERE

engine:
  dir: ./
  name: fischer_engine
  protocol: uci

challenge:
  accept_bot: true
  accept_titled: true
```

### 4. Make Engine Executable

```bash
chmod +x fischer_engine.py  # Linux/Mac
```

Or on Windows, just run it directly.

## Running the Bot

```bash
python lichess-bot.py
```

Expected output:
```
INFO:__main__:Logging in as [your-username]...
INFO:__main__:Listening for new challenges...
```

## Testing

Before deploying to public, test locally:

```bash
python fischer_engine.py
```

Then type:
```
uci
position startpos
go depth 3
```

Should respond with: `bestmove e2e4` or similar

## Monitoring

Bot will:
- ✅ Auto-accept all challenges
- ✅ Play vs humans and bots
- ✅ Adapt to time control (blitz → rapid → classical)
- ✅ Log all games

Check your Lichess profile for:
- Games played
- Rating calibration
- Challenge history

## Performance

Expected strength:
- **Blitz (1+0):** Depth 2, ~0.5s/move
- **Rapid (10+0):** Depth 3, ~2-3s/move
- **Classical (30+0):** Depth 4, ~5-10s/move

Estimated rating: **1500-1700 Elo**

## Customization

Edit `fischer_engine.py` line ~67:

```python
# Change default depth
depth = 3  # Change to 2 (faster) or 4 (stronger)
```

Or adjust `config.yml`:

```yaml
engine:
  uci_options:
    depth: 4  # For classical games
```

## Troubleshooting

**"Token invalid"**
- Token is: `YOUR_LICHESS_TOKEN_HERE`
- Already upgraded ✓

**"Engine not found"**
- Ensure `fischer_engine.py` is in lichess-bot root
- Check `config.yml` has correct paths

**"Module not found"**
- Ensure `src/` folder is copied
- Ensure dependencies installed: `pip install -r requirements.txt`

**Bot too slow**
- Reduce depth in config: `depth: 2`
- Or increase time per game

**Bot crashes**
- Check console for errors
- Restart: `python lichess-bot.py`

## Next Steps

1. ✅ Account upgraded
2. ✅ Config created
3. ✅ Engine ready
4. Run: `python lichess-bot.py`
5. Go to https://lichess.org/api/account and verify bot status
6. Challenge yourself or have others challenge the bot!

---

**You're ready to launch!** 🚀♟️

Questions? Check Lichess API docs: https://lichess.org/api
