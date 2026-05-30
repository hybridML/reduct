"""
Dataset and dataloader for reduct training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from data.generator import generate_dataset, format_for_training
from data.tokenizer import BrainZipTokenizer


class LogicDataset(Dataset):
    def __init__(self, n_examples: int = 50000, max_len: int = 128):
        self.tokenizer = BrainZipTokenizer()
        self.max_len = max_len
        raw = generate_dataset(n_examples)
        self.examples = []
        for premise, conclusion in raw:
            full = format_for_training(premise, conclusion)
            ids = self.tokenizer.tokenize_with_special(full, add_bos=True, add_eos=True)
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
    pad_id = 0

    padded_x = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    padded_y = torch.full((len(batch), max_len), pad_id, dtype=torch.long)

    for i, (x, y) in enumerate(zip(xs, ys)):
        padded_x[i, : x.size(0)] = x
        padded_y[i, : y.size(0)] = y

    return padded_x, padded_y


def get_dataloader(n_examples: int = 50000, batch_size: int = 32, max_len: int = 128) -> DataLoader:
    dataset = LogicDataset(n_examples=n_examples, max_len=max_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )