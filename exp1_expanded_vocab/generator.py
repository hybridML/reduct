"""
Approach 1: Expanded Token Space

Core insight: In the original experiment, the model had only 20 entity tokens
(cat_0..cat_19). It could memorize specific token→token mappings rather than
learning the abstract positional structure of inference.

Solution: Train with 500+ entity tokens sampled randomly per example.
The model can't memorize all pairwise combinations — it MUST learn
the structural pattern "all X are Y, all Y are Z → all X are Z"
as a function of POSITIONS, not tokens.

Reserve ent_500..ent_599 as held-out novel tokens for evaluation.
The model has NEVER seen these during training. If it can still compose
valid inferences with them, it has learned compositional reasoning.
"""

import random
import itertools
from typing import List, Tuple

# Training entities: 500 tokens the model sees during training
TRAIN_ENTITIES = [f"ent_{i}" for i in range(500)]
# Novel entities: 100 tokens reserved for evaluation — NEVER in training
NOVEL_ENTITIES = [f"ent_{i}" for i in range(500, 600)]

VARS = [f"var_{c}" for c in "abcdefghij"]

UNIVERSAL_TEMPLATES = [
    "all {a} are {b}",
    "every {a} is {b}",
]

CHAIN_TEMPLATES = [
    "all {a} are {b} all {b} are {c} CONCLUSION: all {a} are {c}",
    "all {a} are {b} and all {b} are {c} therefore all {a} are {c}",
]

INSTANTIATION_TEMPLATES = [
    "all {a} are {b} {x} is {a} CONCLUSION: {x} is {b}",
    "every {a} is {b} {x} is {a} therefore {x} is {b}",
]

MODUS_PONENS_TEMPLATES = [
    "if {x} is {a} then {x} is {b} {x} is {a} CONCLUSION: {x} is {b}",
]

CONTRAPOSITIVE_TEMPLATES = [
    "all {a} are {b} CONCLUSION: all not {b} are not {a}",
]

NEGATION_INTRO_TEMPLATES = [
    "{x} is {a} {x} is not {a} CONCLUSION: contradiction",
]

DISJUNCTION_TEMPLATES = [
    "either {x} is {a} or {x} is {b} {x} is not {a} CONCLUSION: {x} is {b}",
]

DOUBLE_NEGATION_TEMPLATES = [
    "not not {x} is {a} CONCLUSION: {x} is {a}",
]

TRANSITIVE_TEMPLATES = [
    "all {a} are {b} all {b} are {c} all {c} are {d} CONCLUSION: all {a} are {d}",
]


def _ents(n: int, allow_repeat: bool = False) -> List[str]:
    if allow_repeat:
        return [random.choice(TRAIN_ENTITIES) for _ in range(n)]
    return random.sample(TRAIN_ENTITIES, n)


def generate_example() -> Tuple[str, str]:
    gen = random.choice([
        _gen_chain,
        _gen_instantiation,
        _gen_modus_ponens,
        _gen_contrapositive,
        _gen_negation,
        _gen_disjunction,
        _gen_double_negation,
        _gen_long_chain,
    ])
    return gen()


def _gen_chain() -> Tuple[str, str]:
    a, b, c = _ents(3)
    return random.choice(CHAIN_TEMPLATES).format(a=a, b=b, c=c), f"chain: {a}→{b}→{c}"


def _gen_instantiation() -> Tuple[str, str]:
    a, b = _ents(2)
    x = random.choice(VARS)
    return random.choice(INSTANTIATION_TEMPLATES).format(a=a, b=b, x=x), f"instantiation: {x}∈{a}→{x}∈{b}"


def _gen_modus_ponens() -> Tuple[str, str]:
    a, b = _ents(2)
    x = random.choice(VARS)
    return random.choice(MODUS_PONENS_TEMPLATES).format(a=a, b=b, x=x), f"mp: {x}∈{a}→{x}∈{b}"


def _gen_contrapositive() -> Tuple[str, str]:
    a, b = _ents(2)
    return random.choice(CONTRAPOSITIVE_TEMPLATES).format(a=a, b=b), f"contra: ¬{b}→¬{a}"


def _gen_negation() -> Tuple[str, str]:
    a = random.choice(TRAIN_ENTITIES)
    x = random.choice(VARS)
    return random.choice(NEGATION_INTRO_TEMPLATES).format(a=a, x=x), "negation: contradiction"


def _gen_disjunction() -> Tuple[str, str]:
    a, b = _ents(2)
    x = random.choice(VARS)
    return random.choice(DISJUNCTION_TEMPLATES).format(a=a, b=b, x=x), f"disj: {x}∈{b}"


def _gen_double_negation() -> Tuple[str, str]:
    a = random.choice(TRAIN_ENTITIES)
    x = random.choice(VARS)
    return random.choice(DOUBLE_NEGATION_TEMPLATES).format(a=a, x=x), f"dneg: {x}∈{a}"


def _gen_long_chain() -> Tuple[str, str]:
    a, b, c, d = _ents(4)
    return random.choice(TRANSITIVE_TEMPLATES).format(a=a, b=b, c=c, d=d), f"long_chain: {a}→{d}"


def generate_dataset(n: int) -> List[Tuple[str, str]]:
    return [generate_example() for _ in range(n)]


def build_vocab():
    special = [
        "<PAD>", "<BOS>", "<EOS>", "<UNK>",
        "PREMISE:", "CONCLUSION:", "VALID:",
        "all", "every", "some", "each", "is", "are",
        "not", "if", "then", "therefore", "and", "or",
        "either", "unless", "no", "both", "negation",
        "contradiction", "implies", "leads", "to", "so",
        "plus", "given", "it", "follows", "that", "when",
        "has", "does", "of", "the", "a", "at", "least",
        "one", "such", "case", "false", "true", "present",
        "exists", "means", "entails", "equivalent",
    ]
    tokens = set(special)
    tokens.update(VARS)
    tokens.update(TRAIN_ENTITIES)
    # Novel entities map to <UNK> during training
    tokens = sorted(tokens)
    tok2id = {t: i for i, t in enumerate(tokens)}
    id2tok = {i: t for t, i in tok2id.items()}
    return tok2id, id2tok


def build_expanded_vocab():
    """Vocab that INCLUDES novel entities for evaluation."""
    special = [
        "<PAD>", "<BOS>", "<EOS>", "<UNK>",
        "PREMISE:", "CONCLUSION:", "VALID:",
        "all", "every", "some", "each", "is", "are",
        "not", "if", "then", "therefore", "and", "or",
        "either", "unless", "no", "both", "negation",
        "contradiction", "implies", "leads", "to", "so",
        "plus", "given", "it", "follows", "that", "when",
        "has", "does", "of", "the", "a", "at", "least",
        "one", "such", "case", "false", "true", "present",
        "exists", "means", "entails", "equivalent",
    ]
    tokens = set(special)
    tokens.update(VARS)
    tokens.update(TRAIN_ENTITIES)
    tokens.update(NOVEL_ENTITIES)
    tokens = sorted(tokens)
    tok2id = {t: i for i, t in enumerate(tokens)}
    id2tok = {i: t for t, i in tok2id.items()}
    return tok2id, id2tok