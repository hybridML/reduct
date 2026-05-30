"""
Reduct — API Server

Practical, business-facing API endpoints for privacy-preserving reasoning.

Endpoints:
    POST /access    — What can this user access? (access review)
    POST /audit     — Find all violations across all users (compliance audit)
    POST /why       — Explain why a user has a specific access (provenance)
    POST /conflict  — Detect contradictions in policy set (policy review)
    POST /impact    — Simulate a policy change before applying it (change analysis)
    GET  /health    — Health check
    GET  /policies  — List loaded policies
    GET  /entities  — List loaded entities
"""

import json
import os
import re
from typing import Optional, List, Dict, Set
from pathlib import Path
from contextlib import contextmanager

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import (
    EntityRegistry, LogicAtom, logic_to_text,
)
from exp2_neurosymbolic.solver_enhanced import (
    EnhancedProvenanceSolver, SolverConfig, LogicAtom as EnhAtom,
    logic_to_text as enhanced_logic_to_text,
)


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
#  Core: build solver from config
# ──────────────────────────────────────────────────────

@contextmanager
def build_solver(config_path: str = None, naf_enabled: bool = None):
    """Build a fully-loaded solver from config. Yields (solver, registry, employees, roles, policies, constraints)."""
    if config_path:
        load_config(config_path)
    config = get_config()
    policies = get_active_policies()
    employees = get_active_employees()
    roles = get_active_roles()
    constraints = get_active_constraints()
    solver_cfg = get_solver_config()

    naf = naf_enabled if naf_enabled is not None else solver_cfg.get("naf_enabled", False)
    naf_predicates = set(solver_cfg.get("naf_predicates", []))

    solver_config = SolverConfig(
        max_iterations=solver_cfg.get("max_iterations", 30),
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

    for policy_name, policy_data in policies.items():
        for rule in policy_data.get("rules", []):
            atom = parse_rule(rule)
            if atom:
                abs_s = registry.register(atom.subject)
                abs_o = registry.register(atom.obj) if atom.obj else None
                abs_atom = EnhAtom(predicate=atom.predicate, subject=abs_s, obj=abs_o)
                solver.add_fact(abs_atom, source=f"policy:{policy_name}")

    for constraint_type, constraint_list in constraints.items():
        for constraint_text in constraint_list:
            atom = parse_rule(constraint_text)
            if atom:
                abs_s = registry.register(atom.subject)
                abs_o = registry.register(atom.obj) if atom.obj else None
                abs_atom = EnhAtom(predicate=atom.predicate, subject=abs_s, obj=abs_o)
                solver.add_fact(abs_atom, source=f"constraint:{constraint_type}")

    for name, data in employees.items():
        name_id = registry.register(name)
        for role in data.get("roles", []):
            role_id = registry.register(role)
            solver.add_fact(EnhAtom(predicate="IS", subject=name_id, obj=role_id), source="employee_directory")

    solver.forward_chain()
    yield solver, registry, employees, roles, policies, constraints


def _human(atom: EnhAtom, registry: EntityRegistry) -> str:
    return registry.to_human(enhanced_logic_to_text(atom))


def _get_all_roles_for(entity_id: str, solver: EnhancedProvenanceSolver, registry: EntityRegistry) -> List[Dict]:
    """Get all roles (direct + derived) for an entity, with provenance."""
    results = []
    for atom in solver.facts | solver.derived:
        if atom.predicate == "IS" and atom.subject == entity_id:
            prov = solver.get_provenance(str(atom))
            results.append({
                "role": registry.to_human(atom.obj) if atom.obj else str(atom.obj),
                "direct": atom in solver.facts,
                "source": prov.source if prov else "unknown",
                "rule": prov.rule if prov else "unknown",
            })
    # Also check NAF for roles they DON'T have
    naf_roles = []
    for atom in solver._naf_derived:
        if atom.predicate == "NOT_IS" and atom.subject == entity_id:
            naf_roles.append({
                "role": registry.to_human(atom.obj) if atom.obj else str(atom.obj),
                "access": False,
                "reason": "not assigned (closed-world assumption)",
            })
    return results, naf_roles


# ──────────────────────────────────────────────────────
#  Request/Response Models
# ──────────────────────────────────────────────────────

class AccessCheckRequest(BaseModel):
    entity: str = Field(..., description="Name or ID of the user/entity to check")
    config_path: Optional[str] = Field(None, description="Path to domain YAML config")
    naf_enabled: Optional[bool] = Field(None, description="Enable negation-as-failure")


class AccessItem(BaseModel):
    role: str
    access: bool
    direct: Optional[bool] = None
    source: Optional[str] = None
    rule: Optional[str] = None
    reason: Optional[str] = None


class AccessCheckResponse(BaseModel):
    entity: str
    access: List[AccessItem]
    denied: List[AccessItem] = Field([], description="Roles explicitly denied by NAF")
    violations: List[Dict] = Field([], description="Cardinality and mutex violations involving this user")
    summary: str


class AuditRequest(BaseModel):
    config_path: Optional[str] = Field(None, description="Path to domain YAML config")
    naf_enabled: Optional[bool] = Field(None, description="Enable negation-as-failure")


class AuditViolation(BaseModel):
    type: str
    description: str
    severity: str = "high"
    entities: List[str] = []
    roles: List[str] = []
    policy_source: Optional[str] = None


class AuditResponse(BaseModel):
    total_entities: int
    total_policies: int
    total_rules: int
    total_violations: int
    violations: List[AuditViolation]
    summary: str


class WhyRequest(BaseModel):
    entity: str = Field(..., description="Name of the user/entity")
    role: str = Field(..., description="The role/access to explain")
    config_path: Optional[str] = Field(None, description="Path to domain YAML config")
    naf_enabled: Optional[bool] = Field(None, description="Enable negation-as-failure")


class WhyResponse(BaseModel):
    entity: str
    role: str
    has_access: bool
    provenance: str
    derivation_chain: List[Dict] = Field([], description="Step-by-step chain from source rules to conclusion")


class ConflictRequest(BaseModel):
    config_path: Optional[str] = Field(None, description="Path to domain YAML config")
    naf_enabled: Optional[bool] = Field(None, description="Enable negation-as-failure")


class ConflictResponse(BaseModel):
    conflicts: List[Dict] = Field([], description="Contradictions found in the policy set")
    summary: str


class ImpactRequest(BaseModel):
    add_rules: List[str] = Field([], description="Rules to add (e.g. 'all contractor are limited_access')")
    remove_rules: List[str] = Field([], description="Rules to remove (exact match)")
    add_entities: Dict[str, List[str]] = Field({}, description="Entities to add: {'name': ['role1', 'role2']}")
    config_path: Optional[str] = Field(None, description="Base config path")
    naf_enabled: Optional[bool] = Field(None, description="Enable negation-as-failure")


class ImpactChange(BaseModel):
    entity: str
    role: str
    type: str = Field(..., description="gained, lost, or denied")


class ImpactResponse(BaseModel):
    rules_added: int
    rules_removed: int
    entities_added: int
    access_gained: List[ImpactChange] = Field([], description="New access granted by the change")
    access_lost: List[ImpactChange] = Field([], description="Access removed by the change")
    new_violations: List[Dict] = Field([], description="New violations introduced by the change")
    resolved_violations: List[Dict] = Field([], description="Violations resolved by the change")
    summary: str


class HealthResponse(BaseModel):
    status: str
    version: str
    policies_loaded: int
    entities_loaded: int
    naf_enabled: bool
    config_file: str


class PolicyInfo(BaseModel):
    description: str
    rule_count: int
    rules: List[str]


# ──────────────────────────────────────────────────────
#  App
# ──────────────────────────────────────────────────────

app = FastAPI(
    title="Reduct",
    description="Privacy-preserving reasoning API. Ask business questions — get actionable answers with audit-grade provenance.",
    version="1.0.0",
)


@app.on_event("startup")
def startup():
    load_config()


@app.get("/health", response_model=HealthResponse)
def health():
    config = get_config()
    policies = get_active_policies()
    employees = get_active_employees()
    solver_cfg = get_solver_config()
    return HealthResponse(
        status="ok",
        version="1.0.0",
        policies_loaded=len(policies),
        entities_loaded=len(employees),
        naf_enabled=solver_cfg.get("naf_enabled", False),
        config_file=os.path.basename(_config_path) if _config_path else "unknown",
    )


@app.get("/policies")
def list_policies():
    policies = get_active_policies()
    return {name: PolicyInfo(
        description=data.get("description", ""),
        rule_count=len(data.get("rules", [])),
        rules=data.get("rules", []),
    ) for name, data in policies.items()}


@app.get("/entities")
def list_entities():
    return {"employees": get_active_employees(), "role_index": get_active_roles()}


# ──────────────────────────────────────────────────────
#  POST /access — What can this user access?
# ──────────────────────────────────────────────────────

@app.post("/access", response_model=AccessCheckResponse)
def access_check(request: AccessCheckRequest):
    with build_solver(request.config_path, request.naf_enabled) as (solver, registry, employees, roles, policies, constraints):
        entity_id = registry.register(request.entity)
        if request.entity not in employees and entity_id.startswith("id_"):
            # Entity not in directory — check if it was even registered
            # Still add it so derivations work off user-supplied info
            pass

        roles_info, naf_info = _get_all_roles_for(entity_id, solver, registry)

        access_list = []
        for r in roles_info:
            access_list.append(AccessItem(
                role=r["role"],
                access=True,
                direct=r["direct"],
                source=r["source"],
                rule=r["rule"],
            ))
        for n in naf_info:
            access_list.append(AccessItem(
                role=n["role"],
                access=False,
                reason=n["reason"],
            ))

        # Check violations involving this user
        user_violations = []
        for v in solver.check_cardinality():
            if entity_id in [h for h in v.get("holders", [])]:
                user_violations.append({"type": "cardinality", "description": f"Cardinality constraint violated: {v['constraint']}", "actual": v["actual"], "required": v.get("max") or v.get("required")})
        for v in solver.check_mutex():
            if entity_id in v.get("conflicting_entities", []):
                user_violations.append({"type": "mutex", "description": f"Mutually exclusive roles: {v['roles']}", "conflicting_roles": v["roles"]})

        granted = [a for a in access_list if a.access]
        denied = [a for a in access_list if not a.access]

        return AccessCheckResponse(
            entity=request.entity,
            access=access_list,
            denied=denied,
            violations=user_violations,
            summary=f"{request.entity} has {len(granted)} access grants and {len(denied)} explicit denials" + (f" with {len(user_violations)} violation(s)" if user_violations else ""),
        )


# ──────────────────────────────────────────────────────
#  POST /audit — Find all violations across all users
# ──────────────────────────────────────────────────────

def _extract_role_from_constraint(constraint: str) -> List[str]:
    if "(" in constraint and ")" in constraint:
        inner = constraint.split("(")[1].split(")")[0]
        parts = [p.strip() for p in inner.split(",")]
        return parts[1:] if len(parts) > 1 else parts
    return []


@app.post("/audit", response_model=AuditResponse)
def audit(request: AuditRequest):
    with build_solver(request.config_path, request.naf_enabled) as (solver, registry, employees, roles, policies, constraints):
        violations = []

        for a, b in solver.contradictions:
            violations.append(AuditViolation(
                type="contradiction",
                description=f"{_human(a, registry)} contradicts {_human(b, registry)}",
                severity="critical",
                entities=list({a.subject, b.subject} & set(registry._id_to_label.keys())),
                roles=[],
            ))

        for v in solver.check_cardinality():
            holders_human = [registry.to_human(h) for h in v.get("holders", [])]
            violations.append(AuditViolation(
                type="cardinality",
                description=f"Cardinality constraint violated: {v['constraint']} (found {v['actual']}, expected {v.get('max') or v.get('required')})",
                severity="high",
                entities=holders_human,
                roles=_extract_role_from_constraint(v.get("constraint", "")),
            ))

        for v in solver.check_mutex():
            entities_human = [registry.to_human(e) for e in v.get("conflicting_entities", [])]
            roles_human = [registry.to_human(r) for r in v.get("roles", [])]
            violations.append(AuditViolation(
                type="segregation_of_duties",
                description=f"Segregation of duties violation: {', '.join(entities_human)} holds mutually exclusive roles {', '.join(roles_human)}",
                severity="high",
                entities=entities_human,
                roles=roles_human,
                policy_source="constraint:mutex",
            ))

        total_rules = sum(len(p.get("rules", [])) for p in policies.values())
        return AuditResponse(
            total_entities=len(employees),
            total_policies=len(policies),
            total_rules=total_rules,
            total_violations=len(violations),
            violations=violations,
            summary=f"Audited {len(employees)} entities across {len(policies)} policies ({total_rules} rules). Found {len(violations)} violation(s).",
        )


# ──────────────────────────────────────────────────────
#  POST /why — Explain why a user has specific access
# ──────────────────────────────────────────────────────

@app.post("/why", response_model=WhyResponse)
def why(request: WhyRequest):
    with build_solver(request.config_path, request.naf_enabled) as (solver, registry, employees, roles, policies, constraints):
        entity_id = registry.register(request.entity)
        role_id = registry.register(request.role)

        target_atom = EnhAtom(predicate="IS", subject=entity_id, obj=role_id)
        all_facts = solver.facts | solver.derived
        has_access = target_atom in all_facts

        provenance_str = ""
        derivation_chain = []
        if has_access:
            prov = solver.get_provenance(str(target_atom))
            if prov:
                provenance_str = solver.explain(str(target_atom))
                chain = [str(target_atom)]
                current = prov
                depth = 0
                seen = set()
                while current and depth < 10:
                    if str(current.atom) in seen:
                        break
                    seen.add(str(current.atom))
                    derivation_chain.append({
                        "step": str(current.atom),
                        "rule": current.rule,
                        "source": current.source,
                        "human_readable": _human(current.atom, registry) if current.atom else "",
                    })
                    if current.premises:
                        current = solver.get_provenance(str(current.premises[0]))
                    else:
                        break
                    depth += 1
        else:
            # Check NAF
            naf_atom = EnhAtom(predicate="NOT_IS", subject=entity_id, obj=role_id)
            if naf_atom in solver._naf_derived:
                provenance_str = f"{request.entity} does NOT have {request.role} (closed-world assumption: no evidence of assignment)"
            else:
                provenance_str = f"No record of {request.entity} having {request.role}"

        return WhyResponse(
            entity=request.entity,
            role=request.role,
            has_access=has_access,
            provenance=provenance_str,
            derivation_chain=derivation_chain,
        )


# ──────────────────────────────────────────────────────
#  POST /conflict — Detect policy contradictions
# ──────────────────────────────────────────────────────

@app.post("/conflict", response_model=ConflictResponse)
def conflict(request: ConflictRequest):
    with build_solver(request.config_path, request.naf_enabled) as (solver, registry, employees, roles, policies, constraints):
        conflicts = []
        for a, b in solver.contradictions:
            conflicts.append({
                "type": "contradiction",
                "fact_a": _human(a, registry),
                "fact_b": _human(b, registry),
                "severity": "critical",
                "explanation": f"{_human(a, registry)} and {_human(b, registry)} cannot both be true",
            })

        # Check for rules that contradict each other
        rule_pairs = []
        all_rules = []
        for pn, pd in policies.items():
            for rule in pd.get("rules", []):
                all_rules.append((pn, rule))

        for i, (pn1, r1) in enumerate(all_rules):
            atom1 = parse_rule(r1)
            if atom1 and atom1.predicate == "ALL":
                for pn2, r2 in all_rules[i+1:]:
                    atom2 = parse_rule(r2)
                    if atom2 and atom2.predicate == "ALL":
                        # ALL(A, B) and ALL(A, NOT_B) would conflict
                        if atom1.subject == atom2.subject and atom1.obj != atom2.obj:
                            pass  # Different conclusions from same premise = expected, not conflict

        if not conflicts:
            summary = "No contradictions found in the policy set."
        else:
            summary = f"Found {len(conflicts)} contradiction(s) in the policy set."

        return ConflictResponse(conflicts=conflicts, summary=summary)


# ──────────────────────────────────────────────────────
#  POST /impact — Simulate a policy change
# ──────────────────────────────────────────────────────

@app.post("/impact", response_model=ImpactResponse)
def impact(request: ImpactRequest):
    # First: run without changes to get baseline
    with build_solver(request.config_path, request.naf_enabled) as (solver_before, reg_before, emp_before, roles_before, pol_before, con_before):
        before_access = set()
        for atom in solver_before.facts | solver_before.derived:
            if atom.predicate == "IS":
                before_access.add((atom.subject, atom.obj))
        before_violations = len(solver_before.check_cardinality()) + len(solver_before.check_mutex()) + len(solver_before.contradictions)
        before_violation_details = {
            "cardinality": [{"constraint": v["constraint"], "actual": v["actual"]} for v in solver_before.check_cardinality()],
            "mutex": [{"roles": v["roles"], "entities": v.get("conflicting_entities", [])} for v in solver_before.check_mutex()],
            "contradictions": len(solver_before.contradictions),
        }

    # Now: modify config and run again
    import copy
    config_copy = copy.deepcopy(get_config())
    if request.config_path:
        load_config(request.config_path)

    # Apply changes to a copy
    modified_policies = config_copy.get("policies", {})
    modified_employees = config_copy.get("entities", {}).get("employees", {})
    modified_constraints = config_copy.get("constraints", {})

    # Add rules to a new policy called "simulated"
    if request.add_rules:
        if "simulated" not in modified_policies:
            modified_policies["simulated"] = {"description": "Simulated policy changes", "rules": []}
        for rule in request.add_rules:
            modified_policies["simulated"]["rules"].append(rule)

    # Remove rules
    if request.remove_rules:
        for rule in request.remove_rules:
            for pn in modified_policies:
                if rule in modified_policies[pn].get("rules", []):
                    modified_policies[pn]["rules"].remove(rule)

    # Add entities
    for name, role_list in request.add_entities.items():
        modified_employees[name] = {"roles": role_list}

    # Build solver with modified config
    global _config
    _config = config_copy
    _config["policies"] = modified_policies
    _config["entities"]["employees"] = modified_employees
    _config["constraints"] = modified_constraints

    with build_solver(None, request.naf_enabled) as (solver_after, reg_after, emp_after, roles_after, pol_after, con_after):
        after_access = set()
        for atom in solver_after.facts | solver_after.derived:
            if atom.predicate == "IS":
                after_access.add((atom.subject, atom.obj))
        after_violations = len(solver_after.check_cardinality()) + len(solver_after.check_mutex()) + len(solver_after.contradictions)
        after_violation_details = {
            "cardinality": [{"constraint": v["constraint"], "actual": v["actual"]} for v in solver_after.check_cardinality()],
            "mutex": [{"roles": v["roles"], "entities": v.get("conflicting_entities", [])} for v in solver_after.check_mutex()],
            "contradictions": len(solver_after.contradictions),
        }

    # Compute diff
    gained = after_access - before_access
    lost = before_access - after_access

    access_gained = [
        ImpactChange(entity=reg_after.to_human(s) if reg_after.lookup_id(reg_after.to_human(s) if s.startswith("id_") else s) == s else s,
                     role=reg_after.to_human(r) if reg_after.lookup_id(reg_after.to_human(r) if r.startswith("id_") else r) == r else r,
                     type="gained")
        for s, r in gained
    ]
    access_lost = [
        ImpactChange(entity=reg_after.to_human(s) if reg_after.lookup_id(reg_after.to_human(s) if s.startswith("id_") else s) == s else s,
                     role=reg_after.to_human(r) if reg_after.lookup_id(reg_after.to_human(r) if r.startswith("id_") else r) == r else r,
                     type="lost")
        for s, r in lost
    ]

    # Try to humanize the entity/role names
    def humanize(registry, abstract_id):
        for label, aid in registry._label_to_id.items():
            if aid == abstract_id:
                return label
        return abstract_id

    access_gained_clean = []
    for s, r in gained:
        access_gained_clean.append(ImpactChange(
            entity=humanize(reg_after, s),
            role=humanize(reg_after, r),
            type="gained",
        ))
    access_lost_clean = []
    for s, r in lost:
        access_lost_clean.append(ImpactChange(
            entity=humanize(reg_before, s),
            role=humanize(reg_before, r),
            type="lost",
        ))

    new_violations = []
    if after_violations > before_violations:
        new_violations.append({"type": "increased_violations", "before": before_violations, "after": after_violations})

    resolved_violations = []
    if after_violations < before_violations:
        resolved_violations.append({"type": "decreased_violations", "before": before_violations, "after": after_violations})

    total_gain = len(access_gained_clean)
    total_lost = len(access_lost_clean)

    return ImpactResponse(
        rules_added=len(request.add_rules),
        rules_removed=len(request.remove_rules),
        entities_added=len(request.add_entities),
        access_gained=access_gained_clean,
        access_lost=access_lost_clean,
        new_violations=new_violations,
        resolved_violations=resolved_violations,
        summary=f"+{len(request.add_rules)} rules, -{len(request.remove_rules)} rules, +{len(request.add_entities)} entities. {total_gain} access grants gained, {total_lost} lost. Violations: {before_violations} → {after_violations}.",
    )


# ──────────────────────────────────────────────────────
#  POST /chat — LLM privacy proxy
# ──────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message in plain English (will be redacted before LLM)")
    context: Optional[Dict[str, str]] = Field(None, description="Entity name → category mapping, e.g. {'John Smith': 'person', 'metformin': 'drug'}")
    domain: Optional[str] = Field(None, description="Industry domain: healthcare, finance, legal, defense")
    llm_backend: Optional[str] = Field(None, description="LLM backend: ollama, openai, anthropic")
    llm_model: Optional[str] = Field(None, description="Model name (e.g. qwen3:4b, gpt-4o, claude-sonnet-4-20250514)")


class ChatAuditItem(BaseModel):
    entity: str
    alias: str
    category: str


class ChatResponse(BaseModel):
    response: str = Field(..., description="LLM response with real names restored")
    alias_input: str = Field(..., description="What the LLM saw (synthetic aliases)")
    alias_output: str = Field(..., description="What the LLM returned (synthetic aliases)")
    audit: Dict = Field(..., description="Audit trail: mapping, backend, PII status")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    from proxy.reduct_proxy import AliasMapper, ProxyConfig, _scrub_pii, DOMAIN_SYSTEM_PROMPTS
    from proxy.backends import BACKENDS

    domain = request.domain or "healthcare"
    backend_name = request.llm_backend or "ollama"

    config = ProxyConfig(
        domain=domain,
        llm_backend=backend_name,
        llm_model=request.llm_model or "qwen3:4b",
    )

    mapper = AliasMapper()

    # Register context entities
    if request.context:
        for name, category in request.context.items():
            mapper.register(name, category)

    # Auto-detect and register remaining entities
    alias_input = mapper.auto_redact(request.message, domain)
    alias_input = _scrub_pii(alias_input, config)

    # Build system prompt
    system_prompt = DOMAIN_SYSTEM_PROMPTS.get(domain, DOMAIN_SYSTEM_PROMPTS["healthcare"])

    # Call LLM
    backend_cls = BACKENDS.get(backend_name)
    if not backend_cls:
        raise HTTPException(status_code=400, detail=f"Unknown LLM backend: {backend_name}. Choose from: {list(BACKENDS.keys())}")

    try:
        backend = backend_cls()
        llm_response = backend.complete(alias_input, system_prompt, config)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM backend error: {str(e)}")

    # Restore real names
    restored_response = mapper.restore(llm_response)

    return ChatResponse(
        response=restored_response,
        alias_input=alias_input,
        alias_output=llm_response,
        audit={
            "entities_redacted": len(mapper.mapping),
            "entity_mapping": mapper.mapping,
            "llm_backend": backend_name,
            "llm_model": config.llm_model,
            "pii_sent_to_llm": False,
            "domain": domain,
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)