import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from src.model import FischerNet, save_checkpoint, load_checkpoint
from src.dataset import get_loaders


def evaluate(model: FischerNet, val_loader, device: str = "cpu"):
    """Evaluate model on validation set, return loss and top-1 accuracy."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_top1_correct = 0
    total_samples = 0

    with torch.no_grad():
        for boards, move_idxs in val_loader:
            boards = boards.to(device)
            move_idxs = move_idxs.to(device)

            logits = model(boards)
            loss = criterion(logits, move_idxs)
            total_loss += loss.item() * len(move_idxs)

            top1_preds = logits.argmax(dim=1)
            top1_correct = (top1_preds == move_idxs).sum().item()
            total_top1_correct += top1_correct

            total_samples += len(move_idxs)

    avg_loss = total_loss / total_samples
    top1_acc = total_top1_correct / total_samples

    return avg_loss, top1_acc


def train_epoch(model: FischerNet, train_loader, optimizer, device: str = "cpu"):
    """Train for one epoch, return average loss."""
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0

    for boards, move_idxs in train_loader:
        boards = boards.to(device)
        move_idxs = move_idxs.to(device)

        optimizer.zero_grad()
        logits = model(boards)
        loss = criterion(logits, move_idxs)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(move_idxs)
        total_samples += len(move_idxs)

    return total_loss / total_samples


def main():
    parser = argparse.ArgumentParser(description="Train Fischer playstyle model")
    parser.add_argument("--pgn", type=str, required=True, help="Path to PGN file")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default="cpu", help="Device to train on")

    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.pgn}...")
    train_loader, val_loader = get_loaders(args.pgn, batch_size=args.batch_size)

    print(f"Creating model...")
    model = FischerNet()
    model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3
    )

    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume:
        print(f"Resuming from {args.resume}...")
        model, checkpoint = load_checkpoint(args.resume, device=args.device)
        model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["val_loss"]

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device=args.device)
        val_loss, val_top1_acc = evaluate(model, val_loader, device=args.device)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_top1={val_top1_acc:.2%}"
        )

        # Save checkpoint every epoch
        checkpoint_path = checkpoint_dir / f"fischer_epoch_{epoch + 1:02d}.pt"
        save_checkpoint(model, optimizer, epoch, val_loss, val_top1_acc, str(checkpoint_path))

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = checkpoint_dir / "best_model.pt"
            save_checkpoint(model, optimizer, epoch, val_loss, val_top1_acc, str(best_path))

    print("Training complete!")


if __name__ == "__main__":
    main()
