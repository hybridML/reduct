"""
Approach 1: Expanded Vocab Training + Evaluation

Train with 500 entities, evaluate with 100 held-out novel entities.
Same Transformer architecture as original experiment, but the expanded
vocab forces the model to learn positional structure over memorization.
"""

import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model.reduct import BrainZip

from exp1_expanded_vocab.generator import (
    generate_dataset, build_vocab, build_expanded_vocab,
    NOVEL_ENTITIES, TRAIN_ENTITIES, VARS,
)


class ExpandedTokenizer:
    def __init__(self, include_novel: bool = False):
        if include_novel:
            self.tok2id, self.id2tok = build_expanded_vocab()
        else:
            self.tok2id, self.id2tok = build_vocab()
        self.pad_id = self.tok2id["<PAD>"]
        self.bos_id = self.tok2id["<BOS>"]
        self.eos_id = self.tok2id["<EOS>"]
        self.unk_id = self.tok2id["<UNK>"]

    @property
    def vocab_size(self):
        return len(self.tok2id)

    def encode(self, text: str) -> list[int]:
        tokens = text.split()
        return [self.tok2id.get(t, self.unk_id) for t in tokens]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.id2tok.get(i, "<UNK>") for i in ids)

    def tokenize(self, text: str) -> list[int]:
        ids = self.encode(text)
        return [self.bos_id] + ids + [self.eos_id]


class LogicDatasetV2(Dataset):
    def __init__(self, n_examples: int = 100000, max_len: int = 128):
        self.tok = ExpandedTokenizer(include_novel=False)
        self.max_len = max_len
        raw = generate_dataset(n_examples)
        self.examples = []
        for premise, _ in raw:
            ids = self.tok.tokenize(premise)
            ids = ids[:max_len]
            self.examples.append(ids)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids = self.examples[idx]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y


def collate_fn(batch):
    xs, ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    padded_x = torch.full((len(batch), max_len), 0, dtype=torch.long)
    padded_y = torch.full((len(batch), max_len), 0, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        padded_x[i, :x.size(0)] = x
        padded_y[i, :y.size(0)] = y
    return padded_x, padded_y


def train_exp1(
    n_examples: int = 100000,
    epochs: int = 30,
    batch_size: int = 64,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 4,
    d_ff: int = 512,
    lr: float = 3e-4,
    save_dir: str = "exp1_checkpoints",
):
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")

    tok = ExpandedTokenizer(include_novel=False)
    dataset = LogicDatasetV2(n_examples=n_examples)
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = BrainZip(
        vocab_size=tok.vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_len=128,
        dropout=0.1,
    ).to(device)

    print(f"Model params: {model.count_parameters():,}")
    print(f"Vocab size: {tok.vocab_size} (training entities: {len(TRAIN_ENTITIES)})")
    print(f"Novel entities held out: {len(NOVEL_ENTITIES)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        start = time.time()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=0)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train = total_loss / n_batches
        elapsed = time.time() - start

        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=0)
                val_loss += loss.item()
                val_batches += 1
        avg_val = val_loss / val_batches

        print(f"Epoch {epoch+1:3d}/{epochs} | train={avg_train:.4f} | val={avg_val:.4f} | {elapsed:.1f}s")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": avg_val,
                "config": {
                    "d_model": d_model, "n_heads": n_heads,
                    "n_layers": n_layers, "d_ff": d_ff,
                    "max_len": 128, "dropout": 0.1,
                    "vocab_size": tok.vocab_size,
                },
            }, os.path.join(save_dir, "best_model.pt"))
            print(f"  -> Saved best model (val_loss={avg_val:.4f})")

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    return model


def evaluate_exp1(checkpoint_path: str):
    """Evaluate with novel entities the model has NEVER seen."""
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Load model with EXPANDED vocab (includes novel entities)
    tok_expanded = ExpandedTokenizer(include_novel=True)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    # Rebuild model with expanded vocab
    model = BrainZip(
        vocab_size=tok_expanded.vocab_size,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        max_len=cfg["max_len"],
        dropout=cfg["dropout"],
    ).to(device)

    state = ckpt["model_state_dict"]
    model_state = model.state_dict()
    pretrained_size = cfg["vocab_size"]
    for name, param in state.items():
        if "token_emb.weight" in name or "head.weight" in name:
            model_state[name][:pretrained_size] = param[:pretrained_size]
        elif "head.bias" in name:
            model_state[name][:pretrained_size] = param[:pretrained_size]
        else:
            model_state[name] = param
    model.load_state_dict(model_state)
    model.eval()

    print(f"Loaded model. Pretrained vocab: {pretrained_size}, Expanded vocab: {tok_expanded.vocab_size}")
    print(f"Novel entities (ent_500-ent_599) have RANDOM embeddings — model has NEVER trained on them.")

    test_cases = []

    # Known entity tests (control)
    test_cases.extend([
        ("known_chain", "PREMISE: all ent_10 are ent_20 all ent_20 are ent_30 CONCLUSION:",
         ["ent_10", "ent_30"]),
        ("known_instantiation", "PREMISE: all ent_5 are ent_15 var_a is ent_5 CONCLUSION:",
         ["var_a", "ent_15"]),
    ])

    # Novel entity tests (critical — these tokens have random embeddings)
    test_cases.extend([
        ("novel_chain", "PREMISE: all ent_500 are ent_501 all ent_501 are ent_502 CONCLUSION:",
         ["ent_500", "ent_502"]),
        ("novel_instantiation", "PREMISE: all ent_510 are ent_511 var_b is ent_510 CONCLUSION:",
         ["var_b", "ent_511"]),
        ("novel_disjunction", "PREMISE: either var_c is ent_520 or var_c is ent_521 var_c is not ent_520 CONCLUSION:",
         ["var_c", "ent_521"]),
        ("novel_double_neg", "PREMISE: not not var_d is ent_530 CONCLUSION:",
         ["var_d", "ent_530"]),
    ])

    # Cross-over: mix known and novel
    test_cases.extend([
        ("cross_chain", "PREMISE: all ent_100 are ent_500 all ent_500 are ent_501 CONCLUSION:",
         ["ent_100", "ent_501"]),
        ("cross_mixed", "PREMISE: all ent_502 are ent_200 all ent_200 are ent_503 CONCLUSION:",
         ["ent_502", "ent_503"]),
    ])

    print("\n" + "=" * 70)
    print("APPROACH 1: EXPANDED VOCAB EVALUATION")
    print("=" * 70)

    known_passes = 0
    known_total = 0
    novel_passes = 0
    novel_total = 0

    for name, prompt, expected_tokens in test_cases:
        ids = tok_expanded.tokenize(prompt)[:-1]  # remove EOS, keep BOS + prompt
        idx = torch.tensor([ids], dtype=torch.long).to(device)

        with torch.no_grad():
            output = model.generate(idx, max_new_tokens=15, temperature=0.3)

        generated = output[0].tolist()
        result_ids = generated[len(ids):]
        decoded = tok_expanded.decode(result_ids)

        is_novel = "novel" in name or "cross" in name
        passed = any(tok.lower() in decoded.lower() for tok in expected_tokens)
        status = "PASS" if passed else "FAIL"

        if is_novel:
            novel_passes += int(passed)
            novel_total += 1
        else:
            known_passes += int(passed)
            known_total += 1

        tag = "NOVEL" if is_novel else "KNOWN"
        print(f"  [{status}] {tag} {name}")
        print(f"    Input:  {prompt}")
        print(f"    Output: {decoded}")
        print()

    print("=" * 70)
    print(f"Known entity accuracy: {known_passes}/{known_total} ({known_passes/known_total*100:.0f}%)" if known_total else "N/A")
    print(f"Novel entity accuracy:  {novel_passes}/{novel_total} ({novel_passes/novel_total*100:.0f}%)" if novel_total else "N/A")
    print()
    print("If novel accuracy ≈ known accuracy → model learned structural patterns")
    print("If novel accuracy << known accuracy → model still memorizing surface forms")

    return {"known": (known_passes, known_total), "novel": (novel_passes, novel_total)}


if __name__ == "__main__":
    train_exp1()
    evaluate_exp1("exp1_checkpoints/best_model.pt")