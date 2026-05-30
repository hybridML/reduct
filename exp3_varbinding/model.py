"""
Approach 3: Variable Binding Architecture (Slot Attention)

The fundamental problem: standard Transformers bind meaning to token IDs.
When they see "cat_0", the embedding IS the meaning. Novel tokens get <UNK>,
which has no meaning, so reasoning breaks.

Solution: Add an explicit variable-binding mechanism that learns to map
positions in a template to abstract ROLE SLOTS:

  "all X are Y" → [QUANTIFIER=all, SUBJECT=X, PREDICATE=Y]

The model learns:
  - "all" → QUANTIFIER slot (fixed, always means "universal")
  - Position 1 → SUBJECT slot (variable, can be ANY entity)
  - Position 3 → PREDICATE slot (variable, can be ANY entity)

When it sees "all zorp are blarp", the slot attention binds:
  - SUBJECT=zorp, PREDICATE=blarp

The REASONING operates on slots, not tokens:
  - Template: [QUANTIFIER=all, SUBJECT=X, PREDICATE=Y] +
              [QUANTIFIER=all, SUBJECT=Y, PREDICATE=Z]
  - Rule:     → [QUANTIFIER=all, SUBJECT=X, PREDICATE=Z]

Novel entities are fine because the model never needs to "understand"
what zorp IS — it only needs to know that zorp fills the SUBJECT slot
and blarp fills the PREDICATE slot, and then apply the transitivity rule.

Architecture:
  1. Token embedder → standard embeddings
  2. Slot attention layer → maps token positions to role slots
  3. Slot transformer → reasons over slot representations (not token sequences)
  4. Output head → maps slots back to tokens
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotAttention(nn.Module):
    """
    Maps a sequence of token embeddings to a fixed number of semantic slots.

    Each slot represents a ROLE in the logical template:
    Slot 0: QUANTIFIER (all, some, not, if, etc.)
    Slot 1: SUBJECT (the entity being quantified over)
    Slot 2: PREDICATE (the property being ascribed)

    Key innovation: slots are POSITIONAL roles, not token-specific.
    The model learns WHICH tokens fill WHICH roles, not what the tokens mean.
    """

    def __init__(self, d_model: int, n_slots: int = 6, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_slots = n_slots
        self.d_model = d_model

        self.slot_queries = nn.Parameter(torch.randn(1, n_slots, d_model) * 0.02)
        self.slot_keys = nn.Linear(d_model, d_model)
        self.slot_values = nn.Linear(d_model, d_model)
        self.gru = nn.GRUCell(d_model, d_model)
        self.norm_slots = nn.LayerNorm(d_model)
        self.norm_input = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D) — token sequence
        returns: (B, n_slots, D) — slot representations
        """
        B, T, D = x.shape
        slots = self.slot_queries.expand(B, -1, -1)  # (B, n_slots, D)

        for _ in range(3):  # iterative attention rounds
            slots_norm = self.norm_slots(slots)
            x_norm = self.norm_input(x)

            # Compute attention from slots to input
            attn_logits = torch.bmm(
                slots_norm,  # (B, n_slots, D)
                x_norm.transpose(1, 2),  # (B, D, T)
            ) / math.sqrt(D)  # (B, n_slots, T)

            # Softmax over input dimension (each slot attends to relevant tokens)
            attn = F.softmax(attn_logits, dim=-1)  # (B, n_slots, T)

            # Weighted sum of input
            updates = torch.bmm(attn, x)  # (B, n_slots, D)

            # GRU update
            slots = self.gru(
                updates.reshape(B * self.n_slots, D),
                slots.reshape(B * self.n_slots, D),
            ).reshape(B, self.n_slots, D)

            slots = self.norm_slots(slots)

        return slots


class SlotReasoningBlock(nn.Module):
    """
    Applies Transformer self-attention over SLOT representations,
    not token sequences. This is the key architectural change.

    Slots represent roles (SUBJECT, PREDICATE), not specific entities.
    Self-attention over slots learns logical composition rules:
    "slot_SUBJECT of fact 1 matches slot_PREDICATE of fact 2" = transitivity
    """

    def __init__(self, d_model: int, n_heads: int = 4, d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """
        slots: (B, n_slots, D)
        returns: (B, n_slots, D)
        """
        residual = slots
        slots = self.ln1(slots)
        slots, _ = self.attn(slots, slots, slots)
        slots = residual + slots

        residual = slots
        slots = self.ln2(slots)
        slots = self.ff(slots)
        slots = residual + slots

        return slots


class SlotBindingTransformer(nn.Module):
    """
    Full architecture for variable-binding reasoning.

    1. Token Embedding → standard learned embeddings
    2. Slot Attention → bind tokens to semantic roles
    3. Slot Transformer → reason over role representations
    4. Dual output heads:
       a. Token head: generate next token (for training)
       b. Slot head: predict slot assignments (for interpretability)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_slots: int = 6,
        n_heads: int = 4,
        n_reasoning_layers: int = 4,
        d_ff: int = 256,
        max_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_slots = n_slots
        self.max_len = max_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

        # Token-level processing (shallow — just learns to bind)
        self.token_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True),
            num_layers=2,
        )

        # Slot attention: bind tokens to roles
        self.slot_attention = SlotAttention(d_model, n_slots, n_heads, dropout)

        # Slot-level reasoning: compose over roles
        self.slot_reasoning = nn.ModuleList(
            [SlotReasoningBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_reasoning_layers)]
        )

        # Output heads
        self.token_head = nn.Linear(d_model, vocab_size)
        self.slot_classifier = nn.Linear(d_model, 8)  # 8 role types

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        idx: (B, T) — token indices
        returns: (token_logits, slot_role_logits)
        """
        B, T = idx.shape
        assert T <= self.max_len

        # Step 1: Token embeddings
        x = self.token_emb(idx) + self.pos_enc[:, :T, :]
        x = self.dropout(x)

        # Step 2: Shallow token processing
        x = self.token_transformer(x)

        # Step 3: Bind tokens to slots
        slots = self.slot_attention(x)  # (B, n_slots, D)

        # Step 4: Reason over slots
        for block in self.slot_reasoning:
            slots = block(slots)

        # Step 5: Map slots back to token space
        # Use slot representations to enhance token predictions
        # Each token position attends to all slots
        slot_attn = torch.bmm(x, slots.transpose(1, 2)) / math.sqrt(self.d_model)  # (B, T, n_slots)
        slot_attn = F.softmax(slot_attn, dim=-1)
        slot_context = torch.bmm(slot_attn, slots)  # (B, T, D)

        # Combine original token representations with slot context
        enhanced = x + slot_context

        token_logits = self.token_head(enhanced)
        slot_role_logits = self.slot_classifier(slots)

        return token_logits, slot_role_logits

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 20, temperature: float = 0.3) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_len:]
            token_logits, _ = self(idx_cond)
            logits = token_logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==================== Training Data ====================

SLOT_ROLES = {
    "QUANTIFIER": 0,  # all, some, not, if, each, every
    "SUBJECT": 1,     # entity being talked about
    "PREDICATE": 2,   # property being ascribed
    "CONNECTIVE": 3,  # and, or, therefore, implies
    "NEGATION": 4,    # not, no
    "CONCLUSION": 5,  # CONCLUSION marker
    "VALIDITY": 6,    # VALID marker
    "OTHER": 7,       # padding, etc.
}

KEYWORD_ROLES = {
    "all": SLOT_ROLES["QUANTIFIER"],
    "every": SLOT_ROLES["QUANTIFIER"],
    "some": SLOT_ROLES["QUANTIFIER"],
    "each": SLOT_ROLES["QUANTIFIER"],
    "if": SLOT_ROLES["QUANTIFIER"],
    "not": SLOT_ROLES["NEGATION"],
    "no": SLOT_ROLES["NEGATION"],
    "and": SLOT_ROLES["CONNECTIVE"],
    "or": SLOT_ROLES["CONNECTIVE"],
    "therefore": SLOT_ROLES["CONNECTIVE"],
    "implies": SLOT_ROLES["CONNECTIVE"],
    "CONCLUSION": SLOT_ROLES["CONCLUSION"],
    "VALID": SLOT_ROLES["VALIDITY"],
    "is": SLOT_ROLES["CONNECTIVE"],
    "are": SLOT_ROLES["CONNECTIVE"],
}


def assign_slot_roles(tokens: list[str], template_type: str) -> list[int]:
    """Assign semantic role labels to each token in a sequence."""
    roles = []
    for tok in tokens:
        if tok in KEYWORD_ROLES:
            roles.append(KEYWORD_ROLES[tok])
        elif tok.startswith("ent_") or tok.startswith("var_"):
            # Determine subject vs predicate by position and template
            roles.append(SLOT_ROLES["SUBJECT"])  # simplified
        else:
            roles.append(SLOT_ROLES["OTHER"])
    return roles