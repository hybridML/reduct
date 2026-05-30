"""
Tokenizer for reduct synthetic language.

Byte-pair encoding is overkill for our controlled vocabulary.
We use a simple word-level tokenizer since our "language"
is fully synthetic with a known, small vocabulary.
"""

from data.generator import build_vocab


class BrainZipTokenizer:
    def __init__(self):
        self.tok2id, self.id2tok = build_vocab()
        self.pad_id = self.tok2id["<PAD>"]
        self.bos_id = self.tok2id["<BOS>"]
        self.eos_id = self.tok2id["<EOS>"]
        self.unk_id = self.tok2id["<UNK>"]

    @property
    def vocab_size(self) -> int:
        return len(self.tok2id)

    def encode(self, text: str) -> list[int]:
        tokens = text.split()
        ids = []
        for t in tokens:
            if t in self.tok2id:
                ids.append(self.tok2id[t])
            else:
                ids.append(self.unk_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.id2tok.get(i, "<UNK>") for i in ids)

    def tokenize_with_special(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        ids = self.encode(text)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids