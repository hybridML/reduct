"""
Synthetic data generator for reduct.

Generates training examples that contain ONLY logical structure —
implication, conjunction, disjunction, negation, quantification —
using synthetic placeholder tokens (var_X, cat_A) instead of
any real-world entities.

No factual content. No encyclopedic knowledge. Pure compositional reasoning.
"""

import random
import itertools
from typing import List, Tuple

VARS = [f"var_{c}" for c in "abcdefghij"]
CATS = [f"cat_{i}" for i in range(20)]
PROPS = [f"prop_{i}" for i in range(20)]

NEGATION_TEMPLATES = [
    "not {p}",
    "it is not the case that {p}",
    "{p} is false",
    "negation of {p}",
]

CONJUNCTION_TEMPLATES = [
    "{p} and {q}",
    "both {p} and {q}",
    "{p} plus {q}",
]

DISJUNCTION_TEMPLATES = [
    "{p} or {q}",
    "either {p} or {q}",
    "{p} unless {q}",
]

IMPLICATION_TEMPLATES = [
    "if {p} then {q}",
    "{p} implies {q}",
    "{p} leads to {q}",
    "when {p} therefore {q}",
    "given {p} it follows that {q}",
]

UNIVERSAL_TEMPLATES = [
    "all {x} are {y}",
    "every {x} is {y}",
    "for any {x} {x} is {y}",
    "each {x} is a {y}",
]

EXISTENTIAL_TEMPLATES = [
    "some {x} are {y}",
    "there exists {x} such that {x} is {y}",
    "at least one {x} is {y}",
]

CHAIN_TEMPLATES = [
    "all {x} are {y} and all {y} are {z} therefore all {x} are {z}",
    "all {x} are {y} all {y} are {z} so all {x} are {z}",
]

CONTRAPOSITIVE_TEMPLATES = [
    "all {x} are {y} therefore all not {y} are not {x}",
]

MODUS_PONENS_TEMPLATES = [
    "all {x} are {y} {x} is present therefore {x} is {y}",
]

MODUS_TOLLENS_TEMPLATES = [
    "all {x} are {y} no {y} exists therefore no {x} exists",
]

DISJUNCTIVE_SYLLOGISM_TEMPLATES = [
    "either {x} is {y} or {x} is {z} {x} is not {y} therefore {x} is {z}",
]


def _pick(seq, n, allow_repeat=True):
    if allow_repeat:
        return [random.choice(seq) for _ in range(n)]
    return random.sample(seq, min(n, len(seq)))


def generate_atomic_proposition(n=1) -> str:
    templates = [
        "{x} is {y}",
        "{x} has {z}",
        "{x} does {z}",
    ]
    results = []
    for _ in range(n):
        t = random.choice(templates)
        results.append(t.format(
            x=random.choice(VARS),
            y=random.choice(CATS),
            z=random.choice(PROPS),
        ))
    return results[0] if n == 1 else results


def generate_negation() -> Tuple[str, str]:
    p = generate_atomic_proposition()
    template = random.choice(NEGATION_TEMPLATES)
    negated = template.format(p=p)
    return negated, f"VALID: {p} and {negated} is a contradiction"


def generate_conjunction() -> Tuple[str, str]:
    p = generate_atomic_proposition()
    q = generate_atomic_proposition()
    while q == p:
        q = generate_atomic_proposition()
    template = random.choice(CONJUNCTION_TEMPLATES)
    conj = template.format(p=p, q=q)
    return conj, f"VALID: {conj} entails {p}"


def generate_implication() -> Tuple[str, str]:
    p = generate_atomic_proposition()
    q = generate_atomic_proposition()
    template = random.choice(IMPLICATION_TEMPLATES)
    impl = template.format(p=p, q=q)
    return impl, f"VALID: {impl} means {p} leads to {q}"


def generate_chain() -> Tuple[str, str]:
    x, y, z = random.sample(CATS, 3)
    template = random.choice(CHAIN_TEMPLATES)
    chain = template.format(x=x, y=y, z=z)
    conclusion = f"all {x} are {z}"
    return chain, f"CONCLUSION: {conclusion}"


def generate_modus_ponens() -> Tuple[str, str]:
    x = random.choice(VARS)
    a, b = random.sample(CATS, 2)
    template = random.choice(MODUS_PONENS_TEMPLATES)
    premise = template.format(x=a, y=b)
    conclusion = f"{x} is {b}"
    return premise, f"CONCLUSION: {conclusion}"


def generate_modus_tollens() -> Tuple[str, str]:
    x = random.choice(VARS)
    a, b = random.sample(CATS, 2)
    template = random.choice(MODUS_TOLLENS_TEMPLATES)
    premise = template.format(x=a, y=b)
    conclusion = f"no {a} exists"
    return premise, f"CONCLUSION: {conclusion}"


def generate_disjunctive_syllogism() -> Tuple[str, str]:
    x = random.choice(VARS)
    y, z = random.sample(CATS, 2)
    template = random.choice(DISJUNCTIVE_SYLLOGISM_TEMPLATES)
    premise = template.format(x=x, y=y, z=z)
    conclusion = f"{x} is {z}"
    return premise, f"CONCLUSION: {conclusion}"


def generate_transitive() -> Tuple[str, str]:
    x, y, z = random.sample(VARS, 3)
    a, b, c = random.sample(CATS, 3)
    return (
        f"all {a} are {b} all {b} are {c}",
        f"CONCLUSION: all {a} are {c}"
    )


def generate_quantifier_instantiation() -> Tuple[str, str]:
    x = random.choice(VARS)
    a, b = random.sample(CATS, 2)
    return (
        f"all {a} are {b} {x} is {a}",
        f"CONCLUSION: {x} is {b}"
    )


def generate_double_negation() -> Tuple[str, str]:
    p = generate_atomic_proposition()
    return (
        f"not not {p}",
        f"CONCLUSION: {p}"
    )


def generate_de_morgan() -> Tuple[str, str]:
    p = generate_atomic_proposition()
    q = generate_atomic_proposition()
    variant = random.choice(["and_or", "or_and"])
    if variant == "and_or":
        return (
            f"not ({p} and {q})",
            f"VALID: equivalent to not {p} or not {q}"
        )
    else:
        return (
            f"not ({p} or {q})",
            f"VALID: equivalent to not {p} and not {q}"
        )


GENERATORS = [
    generate_negation,
    generate_conjunction,
    generate_implication,
    generate_chain,
    generate_modus_ponens,
    generate_modus_tollens,
    generate_disjunctive_syllogism,
    generate_transitive,
    generate_quantifier_instantiation,
    generate_double_negation,
    generate_de_morgan,
]


def generate_example() -> Tuple[str, str]:
    gen = random.choice(GENERATORS)
    return gen()


def generate_dataset(n: int = 50000) -> List[Tuple[str, str]]:
    return [generate_example() for _ in range(n)]


def format_for_training(premise: str, conclusion: str) -> str:
    return f"PREMISE: {premise} {conclusion}"


def build_vocab():
    special = ["<PAD>", "<BOS>", "<EOS>", "<UNK>",
               "PREMISE:", "CONCLUSION:", "VALID:", "and", "or", "not",
               "if", "then", "therefore", "all", "every", "some",
               "each", "is", "are", "has", "does", "of",
               "the", "a", "implies", "leads", "to",
               "given", "it", "follows", "that", "when",
               "plus", "either", "unless", "at", "least", "one",
               "such", "case", "false", "true", "present",
               "exists", "no", "both", "negation", "equivalent",
               "contradiction", "entails", "means", "so"]
    tokens = set(special)
    for v in VARS:
        tokens.add(v)
    for c in CATS:
        tokens.add(c)
    for p in PROPS:
        tokens.add(p)
    tokens = sorted(tokens)
    tok2id = {t: i for i, t in enumerate(tokens)}
    id2tok = {i: t for t, i in tok2id.items()}
    return tok2id, id2tok