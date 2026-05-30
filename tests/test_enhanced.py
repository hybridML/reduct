"""
reduct — Enhanced Reasoning Tests

Tests conditionals, NAF, temporal, cardinality, mutual exclusivity,
and the full API with healthcare config.
"""

import os
import sys
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from exp2_neurosymbolic.solver_enhanced import (
    EnhancedProvenanceSolver, SolverConfig, LogicAtom,
    logic_to_text, parse_to_logic,
)
from pipeline import EntityRegistry


# ──────────────────────────────────────────────────────
#  Conditional Reasoning (IF/THEN)
# ──────────────────────────────────────────────────────

class TestConditionals:
    def test_basic_conditional(self):
        solver = EnhancedProvenanceSolver()
        solver.add_fact(LogicAtom("IF", "on_call", "emergency_access"))
        solver.add_fact(LogicAtom("IS", "Dr_Kelly", "on_call"))
        solver.forward_chain()
        derived = solver.facts | solver.derived
        assert LogicAtom("IS", "Dr_Kelly", "emergency_access") in derived

    def test_conditional_with_universal_chain(self):
        solver = EnhancedProvenanceSolver()
        solver.add_fact(LogicAtom("IF", "on_call", "emergency_access"))
        solver.add_fact(LogicAtom("ALL", "emergency_access", "audit_required"))
        solver.add_fact(LogicAtom("IS", "Dr_Kelly", "on_call"))
        solver.forward_chain()
        derived = solver.facts | solver.derived
        assert LogicAtom("IS", "Dr_Kelly", "emergency_access") in derived
        assert LogicAtom("IS", "Dr_Kelly", "audit_required") in derived

    def test_conditional_provenance(self):
        solver = EnhancedProvenanceSolver()
        solver.add_fact(LogicAtom("IF", "on_call", "emergency_access"), source="policy")
        solver.add_fact(LogicAtom("IS", "Dr_Kelly", "on_call"), source="hr")
        solver.forward_chain()
        result = LogicAtom("IS", "Dr_Kelly", "emergency_access")
        prov = solver.get_provenance(str(result))
        assert prov is not None
        assert prov.rule == "conditional_instantiation"

    def test_conditional_no_match(self):
        solver = EnhancedProvenanceSolver()
        solver.add_fact(LogicAtom("IF", "on_call", "emergency_access"))
        solver.add_fact(LogicAtom("IS", "Alice", "finance_employee"))
        solver.forward_chain()
        derived = solver.derived
        assert LogicAtom("IS", "Alice", "emergency_access") not in derived


# ──────────────────────────────────────────────────────
#  Negation-as-Failure (Closed-World)
# ──────────────────────────────────────────────────────

class TestNAF:
    def test_naf_basic(self):
        config = SolverConfig(naf_enabled=True, naf_predicates={"clinical_access"})
        solver = EnhancedProvenanceSolver(config=config)
        solver.add_fact(LogicAtom("IS", "Bob", "billing_staff"))
        solver.add_fact(LogicAtom("IS", "Alice", "physician"))
        solver.forward_chain()
        naf_facts = solver._naf_derived
        assert LogicAtom("NOT_IS", "Bob", "clinical_access") in naf_facts

    def test_naf_does_not_derive_positive(self):
        config = SolverConfig(naf_enabled=True, naf_predicates={"clinical_access"})
        solver = EnhancedProvenanceSolver(config=config)
        solver.add_fact(LogicAtom("IS", "Alice", "physician"))
        solver.add_fact(LogicAtom("ALL", "physician", "clinical_access"))
        solver.forward_chain()
        derived = solver.facts | solver.derived
        assert LogicAtom("IS", "Alice", "clinical_access") in derived
        naf_facts = solver._naf_derived
        assert LogicAtom("NOT_IS", "Alice", "clinical_access") not in naf_facts

    def test_naf_disabled_by_default(self):
        solver = EnhancedProvenanceSolver()
        solver.add_fact(LogicAtom("IS", "Bob", "billing_staff"))
        solver.forward_chain()
        assert len(solver._naf_derived) == 0

    def test_naf_provenance(self):
        config = SolverConfig(naf_enabled=True, naf_predicates={"phi_access"})
        solver = EnhancedProvenanceSolver(config=config)
        solver.add_fact(LogicAtom("IS", "Bob", "billing_staff"))
        solver.forward_chain()
        naf_atom = LogicAtom("NOT_IS", "Bob", "phi_access")
        prov = solver.get_provenance(str(naf_atom))
        assert prov is not None
        assert prov.rule == "negation_as_failure"
        assert prov.source == "closed_world_assumption"


# ──────────────────────────────────────────────────────
#  Temporal Reasoning (AFTER)
# ──────────────────────────────────────────────────────

class TestTemporal:
    def test_after_basic(self):
        solver = EnhancedProvenanceSolver()
        solver.add_fact(LogicAtom("AFTER", "90", "terminated_employee"))
        solver.add_fact(LogicAtom("IS", "Alice", "terminated_employee"))
        solver.forward_chain()
        derived = solver.facts | solver.derived
        assert LogicAtom("HOLDS", "Alice", "expires_after_90") in derived

    def test_after_provenance(self):
        solver = EnhancedProvenanceSolver()
        solver.add_fact(LogicAtom("AFTER", "90", "terminated_employee"), source="policy")
        solver.add_fact(LogicAtom("IS", "Alice", "terminated_employee"), source="hr")
        solver.forward_chain()
        result = LogicAtom("HOLDS", "Alice", "expires_after_90")
        prov = solver.get_provenance(str(result))
        assert prov is not None
        assert prov.rule == "temporal_reasoning"

    def test_after_different_durations(self):
        solver = EnhancedProvenanceSolver()
        solver.add_fact(LogicAtom("AFTER", "30", "inactive_employee"))
        solver.add_fact(LogicAtom("IS", "Bob", "inactive_employee"))
        solver.forward_chain()
        derived = solver.facts | solver.derived
        assert LogicAtom("HOLDS", "Bob", "expires_after_30") in derived


# ──────────────────────────────────────────────────────
#  Cardinality Constraints
# ──────────────────────────────────────────────────────

class TestCardinality:
    def test_at_most_violation(self):
        solver = EnhancedProvenanceSolver()
        solver.add_fact(LogicAtom("AT_MOST", "2", "approver"))
        solver.add_fact(LogicAtom("IS", "Alice", "approver"))
        solver.add_fact(LogicAtom("IS", "Bob", "approver"))
        solver.add_fact(LogicAtom("IS", "Charlie", "approver"))
        violations = solver.check_cardinality()
        assert len(violations) == 1
        assert violations[0]["max"] == 2
        assert violations[0]["actual"] == 3

    def test_at_most_satisfied(self):
        solver = EnhancedProvenanceSolver()
        solver.add_fact(LogicAtom("AT_MOST", "2", "approver"))
        solver.add_fact(LogicAtom("IS", "Alice", "approver"))
        solver.add_fact(LogicAtom("IS", "Bob", "approver"))
        violations = solver.check_cardinality()
        assert len(violations) == 0

    def test_exactly_violation(self):
        solver = EnhancedProvenanceSolver()
        solver.add_fact(LogicAtom("EXACTLY", "1", "primary_physician"))
        solver.add_fact(LogicAtom("IS", "Dr_Evans", "primary_physician"))
        solver.add_fact(LogicAtom("IS", "Dr_Park", "primary_physician"))
        violations = solver.check_cardinality()
        assert len(violations) == 1
        assert violations[0]["required"] == 1
        assert violations[0]["actual"] == 2

    def test_exactly_satisfied(self):
        solver = EnhancedProvenanceSolver()
        solver.add_fact(LogicAtom("EXACTLY", "1", "primary_physician"))
        solver.add_fact(LogicAtom("IS", "Dr_Evans", "primary_physician"))
        violations = solver.check_cardinality()
        assert len(violations) == 0


# ──────────────────────────────────────────────────────
#  Mutual Exclusivity
# ──────────────────────────────────────────────────────

class TestMutex:
    def test_mutex_violation(self):
        solver = EnhancedProvenanceSolver()
        solver.add_fact(LogicAtom("MUTEX", "billing_staff", "physician"))
        solver.add_fact(LogicAtom("IS", "Alice", "billing_staff"))
        solver.add_fact(LogicAtom("IS", "Alice", "physician"))
        violations = solver.check_mutex()
        assert len(violations) == 1
        assert "Alice" in violations[0]["conflicting_entities"]

    def test_mutex_satisfied(self):
        solver = EnhancedProvenanceSolver()
        solver.add_fact(LogicAtom("MUTEX", "billing_staff", "physician"))
        solver.add_fact(LogicAtom("IS", "Alice", "billing_staff"))
        solver.add_fact(LogicAtom("IS", "Bob", "physician"))
        violations = solver.check_mutex()
        assert len(violations) == 0

    def test_mutex_with_derived_roles(self):
        solver = EnhancedProvenanceSolver()
        solver.add_fact(LogicAtom("MUTEX", "clinical_access", "billing_access"))
        solver.add_fact(LogicAtom("IS", "Alice", "physician"))
        solver.add_fact(LogicAtom("ALL", "physician", "clinical_access"))
        solver.add_fact(LogicAtom("IS", "Alice", "billing_staff"))
        solver.add_fact(LogicAtom("ALL", "billing_staff", "billing_access"))
        solver.forward_chain()
        violations = solver.check_mutex()
        all_facts = solver.facts | solver.derived
        assert LogicAtom("IS", "Alice", "clinical_access") in all_facts
        assert LogicAtom("IS", "Alice", "billing_access") in all_facts
        assert len(violations) >= 1


# ──────────────────────────────────────────────────────
#  Combined Reasoning
# ──────────────────────────────────────────────────────

class TestCombinedReasoning:
    def test_hipaa_scenario(self):
        """Full HIPAA scenario: physician derives clinical access, PHI access, audit."""
        solver = EnhancedProvenanceSolver()
        solver.add_fact(LogicAtom("ALL", "physician", "clinical_record_access"))
        solver.add_fact(LogicAtom("ALL", "clinical_record_access", "phi_access"))
        solver.add_fact(LogicAtom("ALL", "phi_access", "audit_required"))
        solver.add_fact(LogicAtom("IS", "Dr_Evans", "physician"))
        solver.forward_chain()
        derived = solver.facts | solver.derived
        assert LogicAtom("IS", "Dr_Evans", "clinical_record_access") in derived
        assert LogicAtom("IS", "Dr_Evans", "phi_access") in derived
        assert LogicAtom("IS", "Dr_Evans", "audit_required") in derived

    def test_conditional_plus_temporal(self):
        """If on_call then emergency_access, after 90 days terminated triggers expiry."""
        solver = EnhancedProvenanceSolver()
        solver.add_fact(LogicAtom("IF", "on_call", "emergency_access"))
        solver.add_fact(LogicAtom("IS", "Dr_Kelly", "on_call"))
        solver.add_fact(LogicAtom("AFTER", "90", "terminated_employee"))
        solver.add_fact(LogicAtom("IS", "Alice", "terminated_employee"))
        solver.forward_chain()
        derived = solver.facts | solver.derived
        assert LogicAtom("IS", "Dr_Kelly", "emergency_access") in derived
        assert LogicAtom("HOLDS", "Alice", "expires_after_90") in derived

    def test_naf_with_cardinality(self):
        """NAF + cardinality: Bob isn't clinical, and only 2 approvers allowed."""
        config = SolverConfig(naf_enabled=True, naf_predicates={"clinical_record_access"})
        solver = EnhancedProvenanceSolver(config=config)
        solver.add_fact(LogicAtom("IS", "Dr_Evans", "physician"))
        solver.add_fact(LogicAtom("ALL", "physician", "clinical_record_access"))
        solver.add_fact(LogicAtom("IS", "Bob", "billing_staff"))
        solver.add_fact(LogicAtom("AT_MOST", "1", "primary_physician"))
        solver.add_fact(LogicAtom("IS", "Dr_Evans", "primary_physician"))
        solver.add_fact(LogicAtom("IS", "Dr_Park", "primary_physician"))
        solver.forward_chain()
        violations = solver.check_cardinality()
        assert len(violations) == 1
        naf_facts = solver._naf_derived
        assert LogicAtom("NOT_IS", "Bob", "clinical_record_access") in naf_facts


# ──────────────────────────────────────────────────────
#  Logic-to-Text and Parser
# ──────────────────────────────────────────────────────

class TestEnhancedParser:
    def test_parse_conditional(self):
        atoms = parse_to_logic("if physician then clinical_access")
        assert len(atoms) == 1
        assert atoms[0].predicate == "IF"
        assert atoms[0].subject == "physician"
        assert atoms[0].obj == "clinical_access"

    def test_parse_temporal(self):
        atoms = parse_to_logic("after 90 days, terminated_employee")
        assert len(atoms) == 1
        assert atoms[0].predicate == "AFTER"
        assert atoms[0].subject == "90"

    def test_parse_at_most(self):
        atoms = parse_to_logic("at most 2 approver")
        assert len(atoms) == 1
        assert atoms[0].predicate == "AT_MOST"
        assert atoms[0].subject == "2"

    def test_parse_exactly(self):
        atoms = parse_to_logic("exactly 1 primary_physician")
        assert len(atoms) == 1
        assert atoms[0].predicate == "EXACTLY"

    def test_parse_mutex(self):
        atoms = parse_to_logic("billing and clinical are mutually exclusive")
        assert len(atoms) == 1
        assert atoms[0].predicate == "MUTEX"

    def test_logic_to_text_conditional(self):
        atom = LogicAtom("IF", "on_call", "emergency_access")
        assert logic_to_text(atom) == "if on_call then emergency_access"

    def test_logic_to_text_temporal(self):
        atom = LogicAtom("AFTER", "90", "terminated_employee")
        assert logic_to_text(atom) == "after 90 days, terminated_employee"

    def test_logic_to_text_at_most(self):
        atom = LogicAtom("AT_MOST", "2", "approver")
        assert logic_to_text(atom) == "at most 2 approver"

    def test_logic_to_text_mutex(self):
        atom = LogicAtom("MUTEX", "billing", "clinical")
        assert logic_to_text(atom) == "billing and clinical are mutually exclusive"

    def test_logic_to_text_holds(self):
        atom = LogicAtom("HOLDS", "Alice", "expires_after_90")
        assert logic_to_text(atom) == "Alice holds expires_after_90"


# ──────────────────────────────────────────────────────
#  Healthcare Config Tests
# ──────────────────────────────────────────────────────

class TestHealthcareConfig:
    def test_load_healthcare_config(self):
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "domain_healthcare.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert "policies" in config
        assert "hipaa_minimum_necessary" in config["policies"]
        assert "constraints" in config
        assert "cardinality" in config["constraints"]
        assert "mutex" in config["constraints"]

    def test_healthcare_policies_parseable(self):
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "domain_healthcare.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        count = 0
        for policy_name, policy in config["policies"].items():
            for rule in policy.get("rules", []):
                atom = parse_to_logic(rule)
                if atom or rule.startswith("all ") or rule.startswith("if ") or rule.startswith("after "):
                    count += 1
        assert count > 0

    def test_healthcare_solver_scenario(self):
        """Simulate the full HIPAA scenario with the enhanced solver."""
        solver = EnhancedProvenanceSolver()
        # Minimum necessary
        solver.add_fact(LogicAtom("ALL", "physician", "clinical_record_access"))
        solver.add_fact(LogicAtom("ALL", "clinical_record_access", "phi_access"))
        solver.add_fact(LogicAtom("ALL", "phi_access", "audit_required"))
        solver.add_fact(LogicAtom("ALL", "billing_staff", "billing_record_access"))
        solver.add_fact(LogicAtom("ALL", "billing_record_access", "phi_access"))
        # Role assignments
        solver.add_fact(LogicAtom("IS", "Dr_Evans", "physician"))
        solver.add_fact(LogicAtom("IS", "Bob", "billing_staff"))
        # Conditional
        solver.add_fact(LogicAtom("IF", "on_call", "emergency_access"))
        solver.add_fact(LogicAtom("IS", "Nurse_Kelly", "on_call"))
        # Temporal
        solver.add_fact(LogicAtom("AFTER", "90", "terminated_employee"))
        solver.add_fact(LogicAtom("IS", "Alice", "terminated_employee"))
        # Cardinality
        solver.add_fact(LogicAtom("EXACTLY", "1", "primary_physician"))
        solver.add_fact(LogicAtom("IS", "Dr_Evans", "primary_physician"))
        solver.add_fact(LogicAtom("IS", "Dr_Park", "primary_physician"))
        # Mutex
        solver.add_fact(LogicAtom("MUTEX", "clinical_record_access", "billing_record_access"))

        solver.forward_chain()
        derived = solver.facts | solver.derived

        # Dr. Evans should have clinical access, PHI access, and audit
        assert LogicAtom("IS", "Dr_Evans", "clinical_record_access") in derived
        assert LogicAtom("IS", "Dr_Evans", "phi_access") in derived
        assert LogicAtom("IS", "Dr_Evans", "audit_required") in derived

        # Bob should have billing access and PHI access
        assert LogicAtom("IS", "Bob", "billing_record_access") in derived
        assert LogicAtom("IS", "Bob", "phi_access") in derived

        # Nurse Kelly gets emergency access via conditional
        assert LogicAtom("IS", "Nurse_Kelly", "emergency_access") in derived

        # Alice's access expires after 90 days
        assert LogicAtom("HOLDS", "Alice", "expires_after_90") in derived

        # Cardinality violation: 2 primary physicians
        violations = solver.check_cardinality()
        assert len(violations) >= 1

        # Mutex violation: anyone with both clinical and billing
        mutex_violations = solver.check_mutex()
        assert len(mutex_violations) >= 1