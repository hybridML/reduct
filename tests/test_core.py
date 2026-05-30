"""
reduct — Tests

Tests the provenance solver, pipeline, config loading, and API.
Run with: pytest tests/ -v
"""

import re
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from exp2_neurosymbolic.solver_provenance import (
    ProvenanceSolver, LogicAtom as ProvAtom,
    logic_to_text, parse_to_logic,
)
from pipeline import EntityRegistry, LogicAtom, logic_to_text as pipe_logic_to_text


# ──────────────────────────────────────────────────────
#  Provenance Solver Tests
# ──────────────────────────────────────────────────────

class TestProvenanceSolver:
    def test_basic_instantiation(self):
        solver = ProvenanceSolver()
        solver.add_fact(ProvAtom("ALL", "id_1", "id_2"), source="policy:finance_access")
        solver.add_fact(ProvAtom("IS", "Alice", "id_1"), source="hr_database")
        solver.forward_chain()
        derived = solver.facts | solver.derived
        result = ProvAtom("IS", "Alice", "id_2")
        assert result in derived, f"Expected {result} in derived, got {derived}"

    def test_transitivity(self):
        solver = ProvenanceSolver()
        solver.add_fact(ProvAtom("ALL", "id_1", "id_2"), source="rule_1")
        solver.add_fact(ProvAtom("ALL", "id_2", "id_3"), source="rule_2")
        solver.forward_chain()
        derived = solver.facts | solver.derived
        result = ProvAtom("ALL", "id_1", "id_3")
        assert result in derived

    def test_multi_hop_reasoning(self):
        solver = ProvenanceSolver()
        solver.add_fact(ProvAtom("ALL", "finance_employee", "budget_portal_access"))
        solver.add_fact(ProvAtom("ALL", "budget_portal_access", "expense_system"))
        solver.add_fact(ProvAtom("IS", "Alice", "finance_employee"))
        solver.forward_chain()
        derived = solver.facts | solver.derived
        assert ProvAtom("IS", "Alice", "budget_portal_access") in derived
        assert ProvAtom("IS", "Alice", "expense_system") in derived

    def test_contradiction_detection(self):
        solver = ProvenanceSolver()
        solver.add_fact(ProvAtom("IS", "Diana", "full_time_employee"))
        solver.add_fact(ProvAtom("NOT_IS", "Diana", "full_time_employee"))
        assert len(solver.contradictions) > 0

    def test_provenance_tracking(self):
        solver = ProvenanceSolver()
        solver.add_fact(ProvAtom("ALL", "finance_employee", "budget_portal_access"), source="policy:finance_access")
        solver.add_fact(ProvAtom("IS", "Alice", "finance_employee"), source="hr_database")
        solver.forward_chain()
        result_atom = ProvAtom("IS", "Alice", "budget_portal_access")
        prov = solver.get_provenance(str(result_atom))
        assert prov is not None
        assert prov.rule == "universal_instantiation"
        assert len(prov.premises) == 2

    def test_explain_chain(self):
        solver = ProvenanceSolver()
        solver.add_fact(ProvAtom("ALL", "A", "B"), source="policy_rule")
        solver.add_fact(ProvAtom("ALL", "B", "C"), source="policy_rule")
        solver.add_fact(ProvAtom("IS", "Alice", "A"), source="hr")
        solver.forward_chain()
        # Alice -> A -> B (instantiation)
        prov_ab = solver.get_provenance(str(ProvAtom("IS", "Alice", "B")))
        assert prov_ab is not None
        assert prov_ab.source == "derived"
        # Check that the provenance of A->B references Alice IS A and ALL A B
        premise_strs = [str(p) for p in prov_ab.premises]
        assert "IS(Alice, A)" in premise_strs
        assert "ALL(A, B)" in premise_strs

    def test_explain_recursive(self):
        solver = ProvenanceSolver()
        solver.add_fact(ProvAtom("ALL", "A", "B"), source="policy")
        solver.add_fact(ProvAtom("ALL", "B", "C"), source="policy")
        solver.forward_chain()
        trans_result = ProvAtom("ALL", "A", "C")
        prov = solver.get_provenance(str(trans_result))
        assert prov is not None
        assert prov.rule == "transitivity"
        explanation = solver.explain(str(trans_result))
        assert "transitivity" in explanation

    def test_no_derivation_without_rules(self):
        solver = ProvenanceSolver()
        solver.add_fact(ProvAtom("IS", "Alice", "manager"), source="hr")
        solver.forward_chain()
        assert len(solver.derived) == 0

    def test_contradiction_in_derived(self):
        solver = ProvenanceSolver()
        solver.add_fact(ProvAtom("ALL", "A", "B"), source="rule")
        solver.add_fact(ProvAtom("IS", "X", "A"), source="fact")
        solver.add_fact(ProvAtom("NOT_IS", "X", "B"), source="fact")
        # NOT_IS should be flagged as contradictory once IS(X,B) is derived
        solver.forward_chain()
        assert len(solver.contradictions) > 0

    def test_deduplication_in_provenance(self):
        solver = ProvenanceSolver()
        solver.add_fact(ProvAtom("IS", "Alice", "manager"))
        solver.add_fact(ProvAtom("IS", "Alice", "manager"))  # duplicate
        assert len([f for f in solver.facts if str(f) == "IS(Alice, manager)"]) == 1


# ──────────────────────────────────────────────────────
#  Entity Registry Tests
# ──────────────────────────────────────────────────────

class TestEntityRegistry:
    def test_register_and_lookup(self):
        reg = EntityRegistry()
        abs_id = reg.register("Alice")
        assert abs_id == "id_1"
        assert reg.lookup_id("Alice") == "id_1"

    def test_abstract_and_restore(self):
        reg = EntityRegistry()
        reg.register("Alice")
        reg.register("manager")
        abs_text = reg.to_abstract("Alice is a manager")
        assert "Alice" not in abs_text
        human_text = reg.to_human(abs_text)
        assert "Alice" in human_text

    def test_id_passthrough(self):
        reg = EntityRegistry()
        result = reg.register("id_42")
        assert result == "id_42"  # abstract IDs pass through unchanged

    def test_multi_word_entity(self):
        reg = EntityRegistry()
        reg.register("budget_portal_access")
        abs_text = reg.to_abstract("budget_portal_access is required")
        assert "budget_portal_access" not in abs_text

    def test_consistent_mapping(self):
        reg = EntityRegistry()
        id1 = reg.register("Alice")
        id2 = reg.register("Alice")
        assert id1 == id2


# ──────────────────────────────────────────────────────
#  Logic Parser Tests
# ──────────────────────────────────────────────────────

class TestLogicParser:
    def test_parse_all_statement(self):
        atoms = parse_to_logic("all finance_employee are budget_portal_access")
        assert len(atoms) == 1
        assert atoms[0].predicate == "ALL"
        assert atoms[0].subject == "finance_employee"
        assert atoms[0].obj == "budget_portal_access"

    def test_parse_is_statement(self):
        atoms = parse_to_logic("Alice is manager")
        assert len(atoms) == 1
        assert atoms[0].predicate == "IS"
        assert atoms[0].subject == "Alice"
        assert atoms[0].obj == "manager"

    def test_parse_not_statement(self):
        atoms = parse_to_logic("Bob is not contractor")
        assert len(atoms) == 1
        assert atoms[0].predicate == "NOT_IS"

    def test_logic_to_text(self):
        from exp2_neurosymbolic.solver_provenance import logic_to_text as l2t
        atom = ProvAtom("ALL", "finance_employee", "budget_portal_access")
        assert l2t(atom) == "all finance_employee are budget_portal_access"
        atom = ProvAtom("IS", "Alice", "manager")
        assert l2t(atom) == "Alice is manager"


# ──────────────────────────────────────────────────────
#  Pipeline Integration Tests (rule-based)
# ──────────────────────────────────────────────────────

class TestPipelineIntegration:
    def test_basic_reasoning(self):
        from pipeline import run_agent
        result = run_agent(
            "Alice is finance_employee. All finance_employee are budget_portal_access.",
            use_llm=False,
            verbose=False,
        )
        assert any("budget_portal_access" in c for c in result["derived_nl"])

    def test_multi_hop(self):
        from pipeline import run_agent
        result = run_agent(
            "All finance_employee are budget_portal_access. All budget_portal_access are expense_system. Alice is finance_employee.",
            use_llm=False,
            verbose=False,
        )
        assert any("expense_system" in c for c in result["derived_nl"])

    def test_contradiction_detection_pipeline(self):
        from pipeline import run_agent
        result = run_agent(
            "Diana is full_time_employee. Diana is not full_time_employee.",
            use_llm=False,
            verbose=False,
        )
        assert len(result["contradictions"]) > 0

    def test_abstract_entity_passthrough(self):
        from pipeline import run_agent
        result = run_agent(
            "All id_99 are id_100. All id_100 are id_101.",
            use_llm=False,
            verbose=False,
        )
        assert any("id_99" in c or "id_101" in c for c in result["derived_nl"])


# ──────────────────────────────────────────────────────
#  Config Loading Tests
# ──────────────────────────────────────────────────────

class TestConfig:
    def test_load_domain_yaml(self):
        import yaml
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "domain.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert "policies" in config
        assert "entities" in config
        assert "finance_access" in config["policies"]
        assert "Alice" in config["entities"]["employees"]

    def test_config_policies_parseable(self):
        import yaml
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "domain.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        for policy_name, policy in config["policies"].items():
            for rule in policy.get("rules", []):
                m = re.match(r'all\s+(\S+)\s+are\s+(\S+)', rule)
                assert m is not None, f"Rule '{rule}' in '{policy_name}' doesn't match expected pattern"


# ──────────────────────────────────────────────────────
#  API Tests (using TestClient)
# ──────────────────────────────────────────────────────

class TestAPI:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from api.server import app
        return TestClient(app)

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["policies_loaded"] > 0

    def test_list_policies(self, client):
        resp = client.get("/policies")
        assert resp.status_code == 200
        data = resp.json()
        assert "finance_access" in data
        assert data["finance_access"]["rule_count"] > 0

    def test_reason_basic(self, client):
        resp = client.post("/reason", json={"query": "Alice is finance_employee", "use_llm": False})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["abstract_facts"]) > 0
        assert len(data["explanations"]) >= 0

    def test_reason_derives_conclusions(self, client):
        resp = client.post("/reason", json={
            "query": "Alice is finance_employee. All finance_employee are budget_portal_access.",
            "use_llm": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["conclusions"]) > 0

    def test_entities_endpoint(self, client):
        resp = client.get("/entities")
        assert resp.status_code == 200
        data = resp.json()
        assert "employees" in data
        assert "Alice" in data["employees"]