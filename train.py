"""
Training loop for reduct.

Trains the model on pure logical forms — no factual knowledge.
Monitors train loss and holds out a validation set
for generalization tracking.
"""

import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import random_split

from model.reduct import BrainZip
from data.dataset import LogicDataset, collate_fn
from data.tokenizer import BrainZipTokenizer


def train(
    n_examples: int = 50000,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 4,
    d_ff: int = 512,
    max_len: int = 128,
    dropout: float = 0.1,
    batch_size: int = 32,
    epochs: int = 20,
    lr: float = 3e-4,
    val_split: float = 0.1,
    save_dir: str = "checkpoints",
    device: str = "auto",
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

    print(f"Device: {device}")

    tokenizer = BrainZipTokenizer()
    full_dataset = LogicDataset(n_examples=n_examples, max_len=max_len)

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    model = BrainZip(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_len=max_len,
        dropout=dropout,
    ).to(device)

    n_params = model.count_parameters()
    print(f"Model parameters: {n_params:,}")
    print(f"Training examples: {train_size:,}")
    print(f"Validation examples: {val_size:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        start = time.time()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)

            loss_mask = y != 0
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=0,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = total_loss / n_batches
        elapsed = time.time() - start

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    ignore_index=0,
                )
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches

        print(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={avg_val_loss:.4f} | "
            f"time={elapsed:.1f}s"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                    "config": {
                        "d_model": d_model,
                        "n_heads": n_heads,
                        "n_layers": n_layers,
                        "d_ff": d_ff,
                        "max_len": max_len,
                        "dropout": dropout,
                        "vocab_size": tokenizer.vocab_size,
                    },
                },
                os.path.join(save_dir, "best_model.pt"),
            )
            print(f"  -> Saved best model (val_loss={avg_val_loss:.4f})")

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    print(f"Model saved to {save_dir}/best_model.pt")

    return model


if __name__ == "__main__":
    train()