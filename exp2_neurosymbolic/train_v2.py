"""
Approach 2: Neuro-Symbolic — Rich Translation Dataset

The key insight: the Transformer's ONLY job is to translate between
natural English and formal logic. It never reasons. It never sees
what the entities mean. It just learns the structural mapping:

  "all X are Y"           ↔  ALL(X, Y)
  "X is Y"                ↔  IS(X, Y)
  "X is not Y"            ↔  NOT_IS(X, Y)
  "all X are Y all Y are Z" → ALL(X, Z)  [via solver, not model]

Rich English paraphrases ensure the model generalizes beyond
exact template matching.

Run: python -m exp2_neurosymbolic.train_v2
"""

import os
import time
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from model.reduct import BrainZip
from exp2_neurosymbolic.solver import LogicAtom, LogicSolver, parse_to_logic, logic_to_text


ENTITIES = [f"ent_{i}" for i in range(500)]
VARS = [f"var_{c}" for c in "abcdefghij"]

# Rich natural English paraphrases for each logical form
# The model must learn ALL of these map to the same logic

UNIVERSAL_NL = [
    "all {s} are {o}",
    "every {s} is {o}",
    "each {s} is {o}",
    "any {s} is an {o}",
    "{s} are always {o}",
    "all {s} become {o}",
    "all {s} belong to {o}",
]

CHAIN_NL = [
    "all {a} are {b} all {b} are {c}",
    "all {a} are {b} and all {b} are {c}",
    "every {a} is {b} every {b} is {c}",
    "all {a} are {b} and every {b} is {c}",
    "all {a} become {b} all {b} become {c}",
]

INSTANTIATION_NL = [
    "all {a} are {b} {v} is {a}",
    "every {a} is {b} and {v} is {a}",
    "all {a} are {b} {v} is an {a}",
    "all {a} belong to {b} {v} is {a}",
    "any {a} is {b} and {v} is an {a}",
]

NEGATION_NL = [
    "{v} is {a} {v} is not {a}",
    "{v} is {a} but {v} is not {a}",
    "{v} is an {a} yet {v} is not an {a}",
]

TRIPLE_CHAIN_NL = [
    "all {a} are {b} all {b} are {c} all {c} are {d}",
    "every {a} is {b} and every {b} is {c} and every {c} is {d}",
]

CONTRAPOSITIVE_NL = [
    "all {a} are {b} therefore all not {b} are not {a}",
    "every {a} is {b} so nothing that is not {b} is an {a}",
]

DISJUNCTION_NL = [
    "either {v} is {a} or {v} is {b} {v} is not {a}",
    "{v} is {a} or {v} is {b} but {v} is not {a}",
]


def _ent(n=2):
    return random.sample(ENTITIES, n)


def _var():
    return random.choice(VARS)


def generate_pair():
    kind = random.choices(
        ["universal", "chain", "instantiation", "negation",
         "triple", "contrapositive", "disjunction"],
        weights=[3, 3, 3, 1, 2, 1, 1],
        k=1
    )[0]

    if kind == "universal":
        s, o = _ent(2)
        nl = random.choice(UNIVERSAL_NL).format(s=s, o=o)
        logic = f"ALL({s}, {o})"
        return f"<NL> PREMISE: {nl} CONCLUSION: all {s} are {o} <LOGIC> ALL({s}, {o})"

    elif kind == "chain":
        a, b, c = _ent(3)
        nl = random.choice(CHAIN_NL).format(a=a, b=b, c=c)
        logic = f"ALL({a}, {b}), ALL({b}, {c})"
        return f"<NL> PREMISE: {nl} CONCLUSION: all {a} are {c} <LOGIC> {logic} => ALL({a}, {c})"

    elif kind == "instantiation":
        a, b = _ent(2)
        v = _var()
        nl = random.choice(INSTANTIATION_NL).format(a=a, b=b, v=v)
        logic = f"ALL({a}, {b}), IS({v}, {a})"
        return f"<NL> PREMISE: {nl} CONCLUSION: {v} is {b} <LOGIC> {logic} => IS({v}, {b})"

    elif kind == "negation":
        a = random.choice(ENTITIES)
        v = _var()
        nl = random.choice(NEGATION_NL).format(v=v, a=a)
        logic = f"IS({v}, {a}), NOT_IS({v}, {a})"
        return f"<NL> PREMISE: {nl} CONCLUSION: contradiction <LOGIC> {logic} => CONTRADICTION"

    elif kind == "triple":
        a, b, c, d = _ent(4)
        nl = random.choice(TRIPLE_CHAIN_NL).format(a=a, b=b, c=c, d=d)
        logic = f"ALL({a}, {b}), ALL({b}, {c}), ALL({c}, {d})"
        return f"<NL> PREMISE: {nl} CONCLUSION: all {a} are {d} <LOGIC> {logic} => ALL({a}, {d})"

    elif kind == "contrapositive":
        a, b = _ent(2)
        nl = random.choice(CONTRAPOSITIVE_NL).format(a=a, b=b)
        logic = f"ALL({a}, {b})"
        return f"<NL> PREMISE: {nl} CONCLUSION: all not {b} are not {a} <LOGIC> {logic} => NOT_ALL({b}, {a})"

    elif kind == "disjunction":
        a, b = _ent(2)
        v = _var()
        nl = random.choice(DISJUNCTION_NL).format(v=v, a=a, b=b)
        logic = f"OR({a}, {b}), NOT_IS({v}, {a})"
        return f"<NL> PREMISE: {nl} CONCLUSION: {v} is {b} <LOGIC> {logic} => IS({v}, {b})"

    return generate_pair()


class TranslatorTokenizer:
    def __init__(self):
        self.special = ["<PAD>", "<BOS>", "<EOS>", "<UNK>",
                        "<NL>", "<LOGIC>", "<SEP>"]
        tokens = set(self.special)
        tokens.update([
            "all", "every", "each", "any", "is", "are", "not",
            "and", "or", "either", "but", "yet", "so",
            "therefore", "become", "belong", "to", "an", "always",
            "nothing", "that", "of", "PREMISE:", "CONCLUSION:",
            "CONTRADICTION", "but",
            "ALL", "IS", "NOT_IS", "NOT_ALL", "OR", "SOME",
            "(", ")", ",", "=>",
        ])
        tokens.update(ENTITIES)
        tokens.update(VARS)
        # Also add novel tokens for generalization
        tokens.update([f"ent_{i}" for i in range(500, 600)])
        self.tokens = sorted(tokens)
        self.tok2id = {t: i for i, t in enumerate(self.tokens)}
        self.id2tok = {i: t for t, i in self.tok2id.items()}
        self.pad_id = self.tok2id["<PAD>"]

    @property
    def vocab_size(self):
        return len(self.tok2id)

    def encode(self, text):
        # Pre-tokenize: add spaces around punctuation so split() works
        for ch in '(),=>':
            text = text.replace(ch, f' {ch} ')
        return [self.tok2id.get(t, self.tok2id["<UNK>"]) for t in text.split()]

    def decode(self, ids):
        raw = " ".join(self.id2tok.get(i, "<UNK>") for i in ids)
        # Collapse spaces around punctuation back
        for ch in '(),=>':
            raw = raw.replace(f' {ch} ', ch)
        raw = raw.replace(f' {ch}', ch).replace(f'{ch} ', ch)
        return raw

    def tokenize(self, text):
        return [self.tok2id["<BOS>"]] + self.encode(text) + [self.tok2id["<EOS>"]]


class TranslationDataset(Dataset):
    def __init__(self, n_examples=80000, max_len=80):
        self.tok = TranslatorTokenizer()
        self.max_len = max_len
        self.examples = []
        for _ in range(n_examples):
            text = generate_pair()
            ids = self.tok.tokenize(text)[:max_len]
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
    ml = max(x.size(0) for x in xs)
    px = torch.full((len(batch), ml), 0, dtype=torch.long)
    py = torch.full((len(batch), ml), 0, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        px[i, :x.size(0)] = x
        py[i, :y.size(0)] = y
    return px, py


def train(n_examples=80000, epochs=30, batch_size=64, lr=3e-4):
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")

    dataset = TranslationDataset(n_examples=n_examples)
    tok = dataset.tok
    print(f"Vocab size: {tok.vocab_size}")

    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = BrainZip(
        vocab_size=tok.vocab_size,
        d_model=128, n_heads=4, n_layers=4, d_ff=512,
        max_len=80, dropout=0.1,
    ).to(device)
    print(f"Model params: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    save_dir = "exp2_checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    best_val = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss, n = 0, 0
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
            n += 1

        scheduler.step()
        avg_train = total_loss / n

        model.eval()
        vloss, vn = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=0)
                vloss += loss.item()
                vn += 1
        avg_val = vloss / vn

        print(f"Epoch {epoch+1:3d}/{epochs} | train={avg_train:.4f} | val={avg_val:.4f} | {time.time()-start:.1f}s")

        if avg_val < best_val:
            best_val = avg_val
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_loss": avg_val,
                "vocab_size": tok.vocab_size,
            }, os.path.join(save_dir, "best_model.pt"))
            print(f"  -> Saved best model (val={avg_val:.4f})")

    print(f"\nDone. Best val_loss: {best_val:.4f}")
    return model, tok


def translate(model, tok, text, device, max_new_tokens=40, temperature=0.2):
    """Translate NL → Logic using the trained model."""
    model.eval()
    ids = tok.encode(f"<NL> PREMISE: {text}")
    # Add BOS
    ids = [tok.tok2id["<BOS>"]] + ids
    idx = torch.tensor([ids], dtype=torch.long).to(device)

    with torch.no_grad():
        output = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature)

    generated = output[0].tolist()
    result = tok.decode(generated)

    # Extract logic part
    if "<LOGIC>" in result:
        logic_part = result.split("<LOGIC>")[-1].strip()
        # Clean up trailing tokens
        logic_part = logic_part.split("<EOS>")[0].strip() if "<EOS>" in logic_part else logic_part
        return logic_part
    return result


if __name__ == "__main__":
    train()