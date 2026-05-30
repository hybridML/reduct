"""
reduct — Enhanced Provenance-Tracked Logic Solver

Adds four critical reasoning capabilities on top of the base solver:

1. CONDITIONALS (IF/THEN):  IF(A, B) + A => B
   "If Alice is on_call, then Alice has emergency_access"

2. NEGATION-AS-FAILURE (closed-world): If CANNOT prove X, then NOT_X
   "We have no record that Bob is finance_employee, so he is NOT finance_employee"
   Configurable per-domain. Essential for access control (default-deny).

3. TEMPORAL REASONING: AFTER(duration, fact) means fact becomes true after duration.
   AFTER(90_days, terminated) means "after 90 days, access is terminated"

4. CARDINALITY CONSTRAINTS: AT_MOST(N, role) — at most N entities can hold a role
   "At most 2 people can be primary_approver"

Provenance is tracked for every derivation, including NAF and conditional conclusions.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Set, Dict
from enum import Enum
import time


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
    atom: LogicAtom
    rule: str
    premises: list
    source: str = ""

    def __str__(self):
        premise_strs = [str(p) for p in self.premises]
        return f"{self.atom} <- {self.rule} <- [{', '.join(premise_strs)}] ({self.source})"


@dataclass
class SolverConfig:
    max_iterations: int = 20
    detect_contradictions: bool = True
    track_provenance: bool = True
    naf_enabled: bool = False
    naf_predicates: Set[str] = field(default_factory=set)
    temporal_reference_time: Optional[float] = None


class EnhancedProvenanceSolver:
    def __init__(self, config: Optional[SolverConfig] = None):
        self.config = config or SolverConfig()
        self.facts: Set[LogicAtom] = set()
        self.derived: Set[LogicAtom] = set()
        self.contradictions: List[Tuple] = []
        self._seen_contradictions: set = set()
        self._provenance: Dict[str, Provenance] = {}
        self._fact_sources: Dict[str, str] = {}
        self._naf_derived: Set[LogicAtom] = set()

    def add_fact(self, atom: LogicAtom, source: str = "user_input"):
        if atom in self.facts:
            return
        self.facts.add(atom)
        self._provenance[str(atom)] = Provenance(
            atom=atom, rule="given", premises=[], source=source
        )
        self._fact_sources[str(atom)] = source
        self._check_contradiction_on_add(atom)

    def _check_contradiction_on_add(self, atom: LogicAtom):
        neg_pred = f"NOT_{atom.predicate}"
        neg = LogicAtom(predicate=neg_pred, subject=atom.subject, obj=atom.obj)
        if neg in self.facts or neg in self.derived:
            pair = tuple(sorted([str(atom), str(neg)]))
            if pair not in self._seen_contradictions:
                self.contradictions.append((atom, neg))
                self._seen_contradictions.add(pair)
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

    def _check_contradiction(self, atom: LogicAtom):
        neg_pred = f"NOT_{atom.predicate}"
        neg = LogicAtom(predicate=neg_pred, subject=atom.subject, obj=atom.obj)
        if neg in self.facts or neg in self.derived:
            pair = tuple(sorted([str(atom), str(neg)]))
            if pair not in self._seen_contradictions:
                self.contradictions.append((atom, neg))
                self._seen_contradictions.add(pair)

    def _all_facts(self) -> Set[LogicAtom]:
        return self.facts | self.derived

    def _expand_role(self, role: str) -> Set[str]:
        """Given a role, find all super-roles through ALL chains.
        E.g., physician -> clinical_record_access -> phi_access -> audit_required
        """
        all_facts = self._all_facts()
        super_roles = set()
        queue = [role]
        visited = set()
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            for f in all_facts:
                if f.predicate == "ALL" and f.subject == current and f.obj:
                    super_roles.add(f.obj)
                    if f.obj not in visited:
                        queue.append(f.obj)
        return super_roles

    # ── Rule 1: Transitivity ──────────────────────────────
    def _try_transitivity(self):
        all_facts = self._all_facts()
        universals = [f for f in all_facts if f.predicate == "ALL"]
        for a in universals:
            for b in universals:
                if a.obj and b.obj and a.obj == b.subject and a.subject != b.obj:
                    new_fact = LogicAtom(predicate="ALL", subject=a.subject, obj=b.obj)
                    if new_fact not in self.facts and new_fact not in self.derived:
                        self.derived.add(new_fact)
                        self._prove(new_fact, "transitivity", [a, b])
                        self._check_contradiction(new_fact)

    # ── Rule 2: Universal Instantiation ──────────────────
    def _try_instantiation(self):
        all_facts = self._all_facts()
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

    # ── Rule 3: Conditionals (IF/THEN) ────────────────────
    # IF(A, B) means "if A holds, then B holds"
    # When we discover IS(x, A), derive IS(x, B)
    # When we discover ALL(A, C) and IF(A, B), and we know someone IS(x, A),
    #   we can also derive IS(x, B) via instantiation
    def _try_conditionals(self):
        all_facts = self._all_facts()
        conditionals = [f for f in all_facts if f.predicate == "IF"]
        for cond in conditionals:
            antecedent_role = cond.subject
            consequent_role = cond.obj
            for inst in all_facts:
                if inst.predicate == "IS" and inst.obj == antecedent_role:
                    new_fact = LogicAtom(predicate="IS", subject=inst.subject, obj=consequent_role)
                    if new_fact not in self.facts and new_fact not in self.derived:
                        self.derived.add(new_fact)
                        self._prove(new_fact, "conditional_instantiation", [inst, cond])
                        self._check_contradiction(new_fact)

    # ── Rule 4: Negation-as-Failure ───────────────────────
    # Closed-world assumption: if we cannot prove IS(x, role) and the domain
    # says role is subject to NAF, then derive NOT_IS(x, role)
    def _try_naf(self):
        if not self.config.naf_enabled:
            return
        all_facts = self._all_facts()
        known_roles = set()
        known_people = set()
        for f in all_facts:
            if f.predicate == "IS":
                known_people.add(f.subject)
                known_roles.add(f.obj)
            elif f.predicate == "ALL":
                known_roles.add(f.subject)
                known_roles.add(f.obj)

        naf_predicates = self.config.naf_predicates
        target_roles = naf_predicates if naf_predicates else known_roles

        for person in known_people:
            for role in target_roles:
                positive = LogicAtom(predicate="IS", subject=person, obj=role)
                negative = LogicAtom(predicate="NOT_IS", subject=person, obj=role)
                if positive not in self.facts and positive not in self.derived:
                    if negative not in self.facts and negative not in self.derived and negative not in self._naf_derived:
                        self._naf_derived.add(negative)
                        self._prove(negative, "negation_as_failure", [],
                                    source="closed_world_assumption")

    # ── Rule 5: Temporal Reasoning ────────────────────────
    # AFTER(duration_days, role) as a universal: anyone in a role gets
    # a time-labeled property after duration.
    # E.g., AFTER(90, terminated_employee) + IS(Charlie, terminated_employee)
    #   => HOLDS(Charlie, access_revoked_after_90_days)
    def _try_temporal(self):
        all_facts = self._all_facts()
        temporals = [f for f in all_facts if f.predicate == "AFTER"]
        for temp in temporals:
            duration = temp.subject
            role_or_prop = temp.obj
            for inst in all_facts:
                if inst.predicate == "IS" and inst.obj == role_or_prop:
                    derived_prop = f"expires_after_{duration}"
                    new_fact = LogicAtom(predicate="HOLDS", subject=inst.subject, obj=derived_prop)
                    if new_fact not in self.facts and new_fact not in self.derived:
                        self.derived.add(new_fact)
                        self._prove(new_fact, "temporal_reasoning", [inst, temp])

    # ── Rule 6: Cardinality Constraints ────────────────────
    # AT_MOST(N, role) means at most N entities can hold role.
    # If more than N IS(x, role) facts exist, flag as violation.
    # EXACTLY(N, role) means exactly N must hold role.
    def check_cardinality(self) -> List[dict]:
        violations = []
        all_facts = self._all_facts()
        constraints = [f for f in all_facts if f.predicate == "AT_MOST"]
        for c in constraints:
            try:
                max_count = int(c.subject)
            except (ValueError, TypeError):
                continue
            role = c.obj
            holders = [f for f in all_facts if f.predicate == "IS" and f.obj == role]
            if len(holders) > max_count:
                violation = {
                    "constraint": str(c),
                    "max": max_count,
                    "actual": len(holders),
                    "holders": [str(h) for h in holders],
                }
                violations.append(violation)

        exact_constraints = [f for f in all_facts if f.predicate == "EXACTLY"]
        for c in exact_constraints:
            try:
                count = int(c.subject)
            except (ValueError, TypeError):
                continue
            role = c.obj
            holders = [f for f in all_facts if f.predicate == "IS" and f.obj == role]
            if len(holders) != count:
                violation = {
                    "constraint": str(c),
                    "required": count,
                    "actual": len(holders),
                    "holders": [str(h) for h in holders],
                }
                violations.append(violation)

        return violations

    # ── Rule 7: Mutual Exclusivity ────────────────────────
    # MUTEX(role_a, role_b) means no entity can hold both roles.
    # If IS(x, role_a) and IS(x, role_b), flag violation.
    def check_mutex(self) -> List[dict]:
        violations = []
        all_facts = self._all_facts()
        mutex_pairs = [f for f in all_facts if f.predicate == "MUTEX"]
        for mp in mutex_pairs:
            role_a = mp.subject
            role_b = mp.obj
            holders_a = {f.subject for f in all_facts if f.predicate == "IS" and f.obj == role_a}
            holders_b = {f.subject for f in all_facts if f.predicate == "IS" and f.obj == role_b}
            conflicts = holders_a & holders_b
            if conflicts:
                violations.append({
                    "constraint": str(mp),
                    "conflicting_entities": list(conflicts),
                    "roles": [role_a, role_b],
                })
        # Also check derived roles through ALL chains
        for mp in mutex_pairs:
            role_a = mp.subject
            role_b = mp.obj
            # Expand role_a and role_b through ALL chains to find all super-roles
            all_roles_a = self._expand_role(role_a)
            all_roles_b = self._expand_role(role_b)
            # Check if any entity holds both role_a (or any super-role) and role_b (or any super-role)
            for ra in all_roles_a | {role_a}:
                for rb in all_roles_b | {role_b}:
                    if ra == rb:
                        continue
                    holders_a = {f.subject for f in all_facts if f.predicate == "IS" and f.obj == ra}
                    holders_b = {f.subject for f in all_facts if f.predicate == "IS" and f.obj == rb}
                    conflicts = holders_a & holders_b
                    for entity in conflicts:
                        already_found = any(
                            entity in v.get("conflicting_entities", []) and
                            set(v.get("roles", [])) == {role_a, role_b}
                            for v in violations
                        )
                        if not already_found:
                            violations.append({
                                "constraint": str(mp),
                                "conflicting_entities": [entity],
                                "roles": [role_a, role_b],
                            })
        return violations

    # ── Rule 8: Disjunctive Syllogism ─────────────────────
    def _try_disjunctive_syllogism(self):
        all_facts = self._all_facts()
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

    # ── Main inference loop ────────────────────────────────
    def forward_chain(self, max_iterations: int = None) -> Set[LogicAtom]:
        if max_iterations is None:
            max_iterations = self.config.max_iterations
        prev_size = 0
        for _ in range(max_iterations):
            self._try_transitivity()
            self._try_instantiation()
            self._try_conditionals()
            self._try_temporal()
            self._try_disjunctive_syllogism()
            curr_size = len(self.facts | self.derived)
            if curr_size == prev_size:
                break
            prev_size = curr_size
        if self.config.naf_enabled:
            self._try_naf()
        return self.facts | self.derived

    def get_provenance(self, atom_str: str) -> Optional[Provenance]:
        return self._provenance.get(atom_str)

    def explain(self, atom_str: str, depth: int = 0, seen: set = None) -> str:
        if seen is None:
            seen = set()
        if atom_str in seen:
            return "  " * depth + "↻ (circular reference)"
        seen.add(atom_str)
        prov = self._provenance.get(atom_str)
        if not prov:
            return "  " * depth + f"{atom_str} (unknown source)"
        indent = "  " * depth
        if prov.rule == "given":
            source_tag = f" [{prov.source}]" if prov.source else ""
            return f"{indent}{atom_str} (given{source_tag})"
        lines = [f"{indent}{atom_str}"]
        lines.append(f"{indent}  <- {prov.rule}")
        for p in prov.premises:
            lines.append(self.explain(str(p), depth + 2, seen.copy()))
        source_tag = f" [{prov.source}]" if prov.source else ""
        lines.append(f"{indent}  (derived{source_tag})")
        return "\n".join(lines)

    def get_all_derivations(self) -> List[dict]:
        results = []
        for atom in sorted(self.derived, key=str):
            prov = self._provenance.get(str(atom))
            results.append({
                "fact": str(atom),
                "rule": prov.rule if prov else "unknown",
                "premises": [str(p) for p in prov.premises] if prov else [],
                "source": prov.source if prov else "",
            })
        return results


def logic_to_text(atom: LogicAtom) -> str:
    if atom.predicate == "ALL" and atom.obj:
        return f"all {atom.subject} are {atom.obj}"
    elif atom.predicate == "IS":
        return f"{atom.subject} is {atom.obj}"
    elif atom.predicate == "IF" and atom.obj:
        return f"if {atom.subject} then {atom.obj}"
    elif atom.predicate == "AFTER" and atom.obj:
        return f"after {atom.subject} days, {atom.obj}"
    elif atom.predicate == "AT_MOST" and atom.obj:
        return f"at most {atom.subject} {atom.obj}"
    elif atom.predicate == "EXACTLY" and atom.obj:
        return f"exactly {atom.subject} {atom.obj}"
    elif atom.predicate == "MUTEX" and atom.obj:
        return f"{atom.subject} and {atom.obj} are mutually exclusive"
    elif atom.predicate == "HOLDS":
        return f"{atom.subject} holds {atom.obj}"
    elif atom.predicate.startswith("NOT_"):
        inner_pred = atom.predicate[4:]
        if inner_pred == "ALL":
            return f"not all {atom.subject} are {atom.obj}"
        elif inner_pred == "IS":
            return f"{atom.subject} is not {atom.obj}"
    return f"{atom.subject} {atom.predicate.lower()} {atom.obj or ''}"


def parse_to_logic(text: str) -> List[LogicAtom]:
    atoms = []
    import re
    # IF X THEN Y
    for m in re.finditer(r'if\s+(\S+)\s+then\s+(\S+)', text, re.IGNORECASE):
        atoms.append(LogicAtom(predicate="IF", subject=m.group(1), obj=m.group(2)))
    # AFTER N days, X
    for m in re.finditer(r'after\s+(\d+)\s+days?,\s*(\S+)', text, re.IGNORECASE):
        atoms.append(LogicAtom(predicate="AFTER", subject=m.group(1), obj=m.group(2)))
    # AT_MOST(N, role)
    for m in re.finditer(r'at\s+most\s+(\d+)\s+(\S+)', text, re.IGNORECASE):
        atoms.append(LogicAtom(predicate="AT_MOST", subject=m.group(1), obj=m.group(2)))
    # EXACTLY(N, role)
    for m in re.finditer(r'exactly\s+(\d+)\s+(\S+)', text, re.IGNORECASE):
        atoms.append(LogicAtom(predicate="EXACTLY", subject=m.group(1), obj=m.group(2)))
    # X and Y are mutually exclusive / MUTEX(X, Y)
    for m in re.finditer(r'(\S+)\s+and\s+(\S+)\s+are\s+mutually\s+exclusive', text, re.IGNORECASE):
        atoms.append(LogicAtom(predicate="MUTEX", subject=m.group(1), obj=m.group(2)))
    # All X are Y
    for m in re.finditer(r'all\s+(\S+)\s+are\s+(\S+)', text):
        atoms.append(LogicAtom(predicate="ALL", subject=m.group(1), obj=m.group(2)))
    # Every X is Y
    for m in re.finditer(r'every\s+(\S+)\s+is\s+(\S+)', text):
        atoms.append(LogicAtom(predicate="ALL", subject=m.group(1), obj=m.group(2)))
    # X is not Y
    for m in re.finditer(r'(\S+)\s+is\s+not\s+(\S+)', text):
        atoms.append(LogicAtom(predicate="NOT_IS", subject=m.group(1), obj=m.group(2)))
    # X is Y
    for m in re.finditer(r'(\S+)\s+is\s+an?\s+(\S+)', text):
        if m.group(1).lower() not in ("all", "every", "not"):
            atoms.append(LogicAtom(predicate="IS", subject=m.group(1), obj=m.group(2)))
    for m in re.finditer(r'(\S+)\s+is\s+(\S+)', text):
        existing = {(a.predicate, a.subject, a.obj) for a in atoms}
        candidate = ("IS", m.group(1), m.group(2))
        if candidate not in existing and m.group(1).lower() not in ("all", "every", "not") and m.group(2).lower() != "not":
            atoms.append(LogicAtom(predicate="IS", subject=m.group(1), obj=m.group(2)))
    return atoms