"""
reduct — API Server

FastAPI server that exposes the reasoning pipeline as a service.
Organizations configure their own policies and entities in config/domain.yaml.

Supports:
  - Universal rules (ALL), instantiation (IS), conditionals (IF/THEN)
  - Negation-as-failure (closed-world assumption)
  - Temporal reasoning (AFTER)
  - Cardinality constraints (AT_MOST, EXACTLY)
  - Mutual exclusivity (MUTEX)
  - Full provenance tracking for audit compliance

Run:
    uvicorn api.server:app --reload --port 8000

Endpoints:
    POST /reason          — Main reasoning endpoint
    POST /explain         — Explain why a conclusion was derived
    GET  /health          — Health check
    GET  /policies         — List loaded policies
    GET  /entities         — List loaded entities
    GET  /constraints      — List loaded constraints
"""

import json
import os
import re
from typing import Optional, List, Dict
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import (
    EntityRegistry, LogicAtom, logic_to_text,
    _preregister_known_entities, _detect_tool_needs,
    query_policy, query_employee, query_who_has_role,
)
from exp2_neurosymbolic.solver_enhanced import (
    EnhancedProvenanceSolver, SolverConfig, LogicAtom as EnhAtom,
    logic_to_text as enhanced_logic_to_text,
)


# ──────────────────────────────────────────────────────
#  Pydantic Schemas
# ──────────────────────────────────────────────────────

class ReasonRequest(BaseModel):
    query: str = Field(..., description="Natural language query to reason about")
    use_llm: bool = Field(False, description="Use local LLM (Ollama) for NL translation")
    config_path: Optional[str] = Field(None, description="Path to domain YAML config (default: config/domain.yaml)")
    naf_enabled: Optional[bool] = Field(None, description="Enable negation-as-failure (overrides config)")
    max_iterations: Optional[int] = Field(None, description="Max forward-chaining iterations (overrides config)")


class Explanation(BaseModel):
    fact: str
    rule: str
    premises: List[str]
    source: str


class ContradictionInfo(BaseModel):
    fact_a: str
    fact_b: str


class CardinalityViolation(BaseModel):
    constraint: str
    max_or_required: Optional[int] = None
    actual: int
    holders: List[str]


class MutexViolation(BaseModel):
    constraint: str
    conflicting_entities: List[str]
    roles: List[str]


class ReasonResponse(BaseModel):
    query: str
    conclusions: List[str]
    abstract_facts: List[str]
    derived_facts: List[str]
    naf_conclusions: List[str] = Field([], description="Facts derived by negation-as-failure")
    contradictions: List[ContradictionInfo]
    explanations: List[str]
    derivations: List[Explanation] = Field([], description="Structured derivation records")
    cardinality_violations: List[CardinalityViolation] = Field([], description="Cardinality constraint violations")
    mutex_violations: List[MutexViolation] = Field([], description="Mutual exclusivity violations")


class ExplainRequest(BaseModel):
    conclusion: str


class ExplainResponse(BaseModel):
    conclusion: str
    explanation: str


class HealthResponse(BaseModel):
    status: str
    llm_mode: str
    policies_loaded: int
    entities_loaded: int
    naf_enabled: bool
    config_file: str


class PolicyInfo(BaseModel):
    description: str
    rule_count: int
    rules: List[str]


# ──────────────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────────────

_config = None
_config_path = None


def load_config(path: str = None) -> dict:
    global _config, _config_path
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "config", "domain.yaml")
    _config_path = path
    with open(path) as f:
        _config = yaml.safe_load(f)
    return _config


def get_config() -> dict:
    global _config
    if _config is None:
        load_config()
    return _config


def get_active_policies() -> dict:
    return get_config().get("policies", {})


def get_active_employees() -> dict:
    return get_config().get("entities", {}).get("employees", {})


def get_active_roles() -> dict:
    return get_config().get("entities", {}).get("role_index", {})


def get_active_constraints() -> dict:
    return get_config().get("constraints", {})


def get_solver_config() -> dict:
    return get_config().get("solver", {})


# ──────────────────────────────────────────────────────
#  Rule Parsing Helpers
# ──────────────────────────────────────────────────────

_RULE_PATTERNS = [
    (r'all\s+(\S+)\s+are\s+(\S+)', "ALL"),
    (r'if\s+(\S+)\s+then\s+(\S+)', "IF"),
    (r'after\s+(\d+)\s+days?,\s*(\S+)', "AFTER"),
    (r'at\s+most\s+(\d+)\s+(\S+)', "AT_MOST"),
    (r'exactly\s+(\d+)\s+(\S+)', "EXACTLY"),
    (r'(\S+)\s+and\s+(\S+)\s+are\s+mutually\s+exclusive', "MUTEX"),
]


def parse_rule(rule_text: str):
    for pattern, predicate in _RULE_PATTERNS:
        m = re.match(pattern, rule_text.strip(), re.IGNORECASE)
        if m:
            return EnhAtom(predicate=predicate, subject=m.group(1), obj=m.group(2))
    return None


# ──────────────────────────────────────────────────────
#  App
# ──────────────────────────────────────────────────────

app = FastAPI(
    title="reduct",
    description="Privacy-preserving reasoning engine with provenance tracking, cardinality constraints, and negation-as-failure. Entities are abstracted before processing — the model never sees plaintext.",
    version="0.2.0",
)


@app.on_event("startup")
def startup():
    load_config()


@app.get("/health", response_model=HealthResponse)
def health():
    from pipeline import _check_ollama
    model = _check_ollama()
    config = get_config()
    policies = get_active_policies()
    employees = get_active_employees()
    solver_cfg = get_solver_config()
    naf = solver_cfg.get("naf_enabled", False)
    return HealthResponse(
        status="ok",
        llm_mode=model if model else "rule-based",
        policies_loaded=len(policies),
        entities_loaded=sum(len(v.get("roles", [])) for v in employees.values()),
        naf_enabled=naf,
        config_file=os.path.basename(_config_path) if _config_path else "unknown",
    )


@app.get("/policies")
def list_policies():
    policies = get_active_policies()
    result = {}
    for name, data in policies.items():
        result[name] = PolicyInfo(
            description=data.get("description", ""),
            rule_count=len(data.get("rules", [])),
            rules=data.get("rules", []),
        )
    return result


@app.get("/entities")
def list_entities():
    employees = get_active_employees()
    roles = get_active_roles()
    return {"employees": employees, "role_index": roles}


@app.get("/constraints")
def list_constraints():
    return get_active_constraints()


@app.post("/reason", response_model=ReasonResponse)
def reason(request: ReasonRequest):
    if request.config_path:
        load_config(request.config_path)

    config = get_config()
    policies = get_active_policies()
    employees = get_active_employees()
    roles = get_active_roles()
    constraints = get_active_constraints()
    solver_cfg = get_solver_config()

    naf = request.naf_enabled if request.naf_enabled is not None else solver_cfg.get("naf_enabled", False)
    max_iter = request.max_iterations or solver_cfg.get("max_iterations", 20)
    naf_predicates = set(solver_cfg.get("naf_predicates", []))

    solver_config = SolverConfig(
        max_iterations=max_iter,
        detect_contradictions=solver_cfg.get("detect_contradictions", True),
        track_provenance=solver_cfg.get("track_provenance", True),
        naf_enabled=naf,
        naf_predicates=naf_predicates,
    )

    registry = EntityRegistry()
    solver = EnhancedProvenanceSolver(config=solver_config)

    for name, data in employees.items():
        registry.register(name)
        for role in data.get("roles", []):
            registry.register(role)
    for role_name in roles:
        registry.register(role_name)
        role_vals = roles[role_name]
        if isinstance(role_vals, list):
            for n in role_vals:
                registry.register(n.strip() if isinstance(n, str) else str(n).strip())
        elif isinstance(role_vals, str):
            for n in role_vals.split(","):
                registry.register(n.strip())

    all_sources = {}

    for policy_name, policy_data in policies.items():
        for rule in policy_data.get("rules", []):
            atom = parse_rule(rule)
            if atom:
                abs_subject = registry.register(atom.subject)
                abs_obj = registry.register(atom.obj) if atom.obj else None
                abs_atom = EnhAtom(predicate=atom.predicate, subject=abs_subject, obj=abs_obj)
                solver.add_fact(abs_atom, source=f"policy:{policy_name}")
                all_sources[str(abs_atom)] = f"policy:{policy_name}"

    for constraint_type, constraint_list in constraints.items():
        for constraint_text in constraint_list:
            atom = parse_rule(constraint_text)
            if atom:
                abs_subject = registry.register(atom.subject)
                abs_obj = registry.register(atom.obj) if atom.obj else None
                abs_atom = EnhAtom(predicate=atom.predicate, subject=abs_subject, obj=abs_obj)
                solver.add_fact(abs_atom, source=f"constraint:{constraint_type}")
                all_sources[str(abs_atom)] = f"constraint:{constraint_type}"

    from pipeline import translate
    translation = translate(request.query, registry, use_llm=request.use_llm)

    all_facts = list(translation.get("logic", []))
    for atom_str in translation.get("logic", []):
        all_sources[atom_str] = "user_input"

    for tc in translation.get("tools", []):
        name = tc.get("name", "")
        args = tc.get("arguments", tc.get("args", {}))
        source = f"query:{name}"

        if name == "query_policy":
            policy_name = args.get("policy_name", args)
            policy = policies.get(policy_name, {"rules": []})
            for rule in policy.get("rules", []):
                atom = parse_rule(rule)
                if atom:
                    abs_s = registry.register(atom.subject)
                    abs_o = registry.register(atom.obj) if atom.obj else None
                    abs_atom = EnhAtom(predicate=atom.predicate, subject=abs_s, obj=abs_o)
                    solver.add_fact(abs_atom, source=source)

        elif name == "query_employee":
            emp_name = args.get("name", args)
            emp = employees.get(emp_name, {})
            name_id = registry.register(emp_name)
            for role in emp.get("roles", []):
                role_id = registry.register(role)
                atom = EnhAtom(predicate="IS", subject=name_id, obj=role_id)
                solver.add_fact(atom, source=source)

        elif name == "query_who_has_role":
            role = args.get("role", args)
            role_id = registry.register(role)
            role_vals = roles.get(role, [])
            if isinstance(role_vals, list):
                names_list = role_vals
            elif isinstance(role_vals, str):
                names_list = [n.strip() for n in role_vals.split(",")]
            else:
                names_list = []
            for n in names_list:
                name_id = registry.register(n.strip() if isinstance(n, str) else str(n).strip())
                atom = EnhAtom(predicate="IS", subject=name_id, obj=role_id)
                solver.add_fact(atom, source=source)

    all_facts_parsed = list(translation.get("logic", []))
    other_predicates = ["IF", "AFTER", "AT_MOST", "EXACTLY", "MUTEX", "NOT_IS"]
    for atom_str in all_facts_parsed:
        m = re.match(r'(' + '|'.join(other_predicates + ['ALL', 'IS']) + r')\((\S+),\s*(\S+)\)', atom_str.strip())
        if m:
            atom = EnhAtom(predicate=m.group(1), subject=m.group(2), obj=m.group(3))
            source = all_sources.get(atom_str, "user_input")
            if atom not in solver.facts:
                solver.add_fact(atom, source=source)

    solver.forward_chain()

    explanations = []
    for atom in sorted(solver.derived, key=str):
        explanation = solver.explain(str(atom))
        explanations.append(explanation)

    derivations = solver.get_all_derivations()

    derived_nl = []
    for atom in sorted(solver.derived, key=str):
        nl = registry.to_human(enhanced_logic_to_text(atom))
        derived_nl.append(nl)

    naf_nl = []
    for atom in sorted(solver._naf_derived, key=str):
        nl = registry.to_human(enhanced_logic_to_text(atom))
        naf_nl.append(nl)

    contradictions = []
    for a, b in solver.contradictions:
        contradictions.append(ContradictionInfo(
            fact_a=registry.to_human(enhanced_logic_to_text(a)),
            fact_b=registry.to_human(enhanced_logic_to_text(b)),
        ))

    cardinality_violations = []
    for v in solver.check_cardinality():
        holders_human = [registry.to_human(h) for h in v.get("holders", [])]
        cardinality_violations.append(CardinalityViolation(
            constraint=v["constraint"],
            max_or_required=v.get("max") or v.get("required"),
            actual=v["actual"],
            holders=holders_human,
        ))

    mutex_violations = []
    for v in solver.check_mutex():
        entities_human = [registry.to_human(e) for e in v.get("conflicting_entities", [])]
        roles_human = [registry.to_human(r) for r in v.get("roles", [])]
        mutex_violations.append(MutexViolation(
            constraint=v["constraint"],
            conflicting_entities=entities_human,
            roles=roles_human,
        ))

    return ReasonResponse(
        query=request.query,
        conclusions=derived_nl,
        abstract_facts=[str(a) for a in solver.facts],
        derived_facts=[str(a) for a in solver.derived],
        naf_conclusions=naf_nl,
        contradictions=contradictions,
        explanations=explanations,
        derivations=derivations,
        cardinality_violations=cardinality_violations,
        mutex_violations=mutex_violations,
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(request: ExplainRequest):
    return ExplainResponse(
        conclusion=request.conclusion,
        explanation="Provenance tracking requires a prior /reason call with the same query.",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)