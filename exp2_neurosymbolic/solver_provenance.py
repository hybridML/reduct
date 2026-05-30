"""
reduct — Provenance-Tracked Logic Solver

Extends the base solver to track WHY every fact was derived.
Every conclusion includes a full derivation chain tracing back to source facts.

This is critical for compliance: an auditor can ask "why does Alice have
budget_portal_access?" and get the full chain:

  IS(Alice, budget_portal_access)
    ← universal_instantiation
    ← [IS(Alice, finance_employee), ALL(finance_employee, budget_portal_access)]
    ← source: HR database (query_employee)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum


class LogicOp(Enum):
    ALL = "ALL"
    SOME = "SOME"
    NOT = "NOT"
    AND = "AND"
    OR = "OR"
    IMPLIES = "IMPLIES"
    IFF = "IFF"


@dataclass
class LogicAtom:
    predicate: str
    subject: str
    obj: Optional[str] = None

    def __str__(self):
        if self.obj:
            return f"{self.predicate}({self.subject}, {self.obj})"
        return f"{self.predicate}({self.subject})"

    def __eq__(self, other):
        if not isinstance(other, LogicAtom):
            return False
        return (self.predicate == other.predicate and
                self.subject == other.subject and
                self.obj == other.obj)

    def __hash__(self):
        return hash((self.predicate, self.subject, self.obj))


@dataclass
class Provenance:
    """Why was this fact derived?"""
    atom: LogicAtom
    rule: str  # "universal_instantiation", "transitivity", "given", etc.
    premises: list  # List of LogicAtom or (LogicAtom, ...) that led to this
    source: str = ""  # Where the premises came from: "user_input", "query_employee", "query_policy", etc.

    def __str__(self):
        premise_strs = [str(p) for p in self.premises]
        return f"{self.atom} ← {self.rule} ← [{', '.join(premise_strs)}] ({self.source})"


class ProvenanceSolver:
    def __init__(self):
        self.facts: set = set()
        self.derived: set = set()
        self.contradictions: List[Tuple] = []
        self._seen_contradictions: set = set()
        self._provenance: dict = {}  # atom_str -> Provenance
        self._fact_sources: dict = {}  # atom_str -> source tag

    def add_fact(self, atom: LogicAtom, source: str = "user_input"):
        if atom in self.facts:
            return
        self.facts.add(atom)
        self._provenance[str(atom)] = Provenance(
            atom=atom, rule="given", premises=[], source=source
        )
        self._fact_sources[str(atom)] = source
        # Check for contradiction
        neg_pred = f"NOT_{atom.predicate}"
        neg = LogicAtom(predicate=neg_pred, subject=atom.subject, obj=atom.obj)
        if neg in self.facts or neg in self.derived:
            pair = tuple(sorted([str(atom), str(neg)]))
            if pair not in self._seen_contradictions:
                self.contradictions.append((atom, neg))
                self._seen_contradictions.add(pair)
            return
        if atom.predicate.startswith("NOT_"):
            positive = LogicAtom(predicate=atom.predicate[4:], subject=atom.subject, obj=atom.obj)
            if positive in self.facts or positive in self.derived:
                pair = tuple(sorted([str(atom), str(positive)]))
                if pair not in self._seen_contradictions:
                    self.contradictions.append((atom, positive))
                    self._seen_contradictions.add(pair)

    def _prove(self, atom: LogicAtom, rule: str, premises: list, source: str = "derived") -> Provenance:
        prov = Provenance(atom=atom, rule=rule, premises=premises, source=source)
        self._provenance[str(atom)] = prov
        return prov

    def _try_transitivity(self):
        all_facts = self.facts | self.derived
        universals = [f for f in all_facts if f.predicate == "ALL"]
        for a in universals:
            for b in universals:
                if a.obj and b.obj and a.obj == b.subject and a.subject != b.obj:
                    new_fact = LogicAtom(predicate="ALL", subject=a.subject, obj=b.obj)
                    if new_fact not in self.facts and new_fact not in self.derived:
                        self.derived.add(new_fact)
                        self._prove(new_fact, "transitivity", [a, b])
                        self._check_contradiction(new_fact)

    def _try_instantiation(self):
        all_facts = self.facts | self.derived
        universals = [f for f in all_facts if f.predicate == "ALL"]
        instances = [f for f in all_facts if f.predicate == "IS"]
        for u in universals:
            if not u.obj:
                continue
            for inst in instances:
                if inst.obj == u.subject:
                    new_fact = LogicAtom(predicate="IS", subject=inst.subject, obj=u.obj)
                    if new_fact not in self.facts and new_fact not in self.derived:
                        self.derived.add(new_fact)
                        self._prove(new_fact, "universal_instantiation", [inst, u])
                        self._check_contradiction(new_fact)

    def _try_disjunctive_syllogism(self):
        all_facts = self.facts | self.derived
        disjunctions = [f for f in all_facts if f.predicate == "OR"]
        negations = [f for f in all_facts if f.predicate.startswith("NOT_")]
        for d in disjunctions:
            for n in negations:
                neg_pred = n.predicate[4:]
                if neg_pred == d.subject:
                    new_fact = LogicAtom(predicate=d.obj, subject=d.subject)
                    if new_fact not in self.facts and new_fact not in self.derived:
                        self.derived.add(new_fact)
                        self._prove(new_fact, "disjunctive_syllogism", [d, n])

    def _check_contradiction(self, atom: LogicAtom):
        neg_pred = f"NOT_{atom.predicate}"
        neg = LogicAtom(predicate=neg_pred, subject=atom.subject, obj=atom.obj)
        if neg in self.facts or neg in self.derived:
            pair = tuple(sorted([str(atom), str(neg)]))
            if pair not in self._seen_contradictions:
                self.contradictions.append((atom, neg))
                self._seen_contradictions.add(pair)

    def forward_chain(self, max_iterations: int = 20) -> set:
        prev_size = 0
        for _ in range(max_iterations):
            self._try_transitivity()
            self._try_instantiation()
            self._try_disjunctive_syllogism()
            curr_size = len(self.facts | self.derived)
            if curr_size == prev_size:
                break
            prev_size = curr_size
        return self.facts | self.derived

    def get_provenance(self, atom_str: str) -> Optional[Provenance]:
        return self._provenance.get(atom_str)

    def explain(self, atom_str: str, depth: int = 0, seen: set = None) -> str:
        """Recursively explain why a fact holds, tracing back to sources."""
        if seen is None:
            seen = set()
        if atom_str in seen:
            return "  " * depth + f"↻ (circular reference)"
        seen.add(atom_str)
        prov = self._provenance.get(atom_str)
        if not prov:
            return "  " * depth + f"{atom_str} (unknown source)"
        indent = "  " * depth
        if prov.rule == "given":
            source_tag = f" [{prov.source}]" if prov.source else ""
            return f"{indent}{atom_str} (given{source_tag})"
        lines = [f"{indent}{atom_str}"]
        lines.append(f"{indent}  ← {prov.rule}")
        for p in prov.premises:
            lines.append(self.explain(str(p), depth + 2, seen.copy()))
        source_tag = f" [{prov.source}]" if prov.source else ""
        lines.append(f"{indent}  (derived{source_tag})")
        return "\n".join(lines)


def logic_to_text(atom: LogicAtom) -> str:
    if atom.predicate == "ALL" and atom.obj:
        return f"all {atom.subject} are {atom.obj}"
    elif atom.predicate == "IS":
        return f"{atom.subject} is {atom.obj}"
    elif atom.predicate.startswith("NOT_"):
        inner_pred = atom.predicate[4:]
        if inner_pred == "ALL":
            return f"not all {atom.subject} are {atom.obj}"
        elif inner_pred == "IS":
            return f"{atom.subject} is not {atom.obj}"
    return f"{atom.subject} {atom.predicate.lower()} {atom.obj or ''}"


def parse_to_logic(text: str) -> List[LogicAtom]:
    atoms = []
    text = text.strip()
    tokens = text.split()
    i = 0
    while i < len(tokens):
        if tokens[i] == "all" and i + 3 < len(tokens) and tokens[i + 2] == "are":
            atoms.append(LogicAtom(predicate="ALL", subject=tokens[i + 1], obj=tokens[i + 3]))
            i += 4
        elif tokens[i] == "every" and i + 3 < len(tokens) and tokens[i + 2] == "is":
            atoms.append(LogicAtom(predicate="ALL", subject=tokens[i + 1], obj=tokens[i + 3]))
            i += 4
        elif i + 3 < len(tokens) and tokens[i + 1] == "is" and tokens[i + 2] == "not":
            atoms.append(LogicAtom(predicate="NOT_IS", subject=tokens[i], obj=tokens[i + 3]))
            i += 4
        elif i + 2 < len(tokens) and tokens[i + 1] == "is":
            atoms.append(LogicAtom(predicate="IS", subject=tokens[i], obj=tokens[i + 2]))
            i += 3
        else:
            i += 1
    return atoms