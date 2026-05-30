"""
Reduct — Practical API Tests

Tests for the business-facing endpoints: /access, /audit, /why, /conflict, /impact.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from api.server import app, load_config

client = TestClient(app)

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "..", "config", "domain.yaml")
HEALTHCARE_CONFIG = os.path.join(os.path.dirname(__file__), "..", "config", "domain_healthcare.yaml")


@pytest.fixture(autouse=True)
def reset_config():
    """Reset config to default before each test."""
    load_config(DEFAULT_CONFIG)


class TestAccessCheck:
    def test_access_for_known_employee(self):
        resp = client.post("/access", json={"entity": "Alice"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["entity"] == "Alice"
        assert len(data["access"]) > 0
        assert any(a["access"] is True for a in data["access"])

    def test_access_grants_roles(self):
        resp = client.post("/access", json={"entity": "Alice"})
        data = resp.json()
        roles = [a["role"] for a in data["access"] if a["access"]]
        assert "budget_portal_access" in roles or any("budget" in r for r in roles)

    def test_access_unknown_entity(self):
        resp = client.post("/access", json={"entity": "UnknownPerson"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["entity"] == "UnknownPerson"

    def test_access_with_naf(self):
        resp = client.post("/access", json={"entity": "Bob", "naf_enabled": True})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["access"]) > 0

    def test_access_summary(self):
        resp = client.post("/access", json={"entity": "Alice"})
        data = resp.json()
        assert "summary" in data
        assert "Alice" in data["summary"]


class TestAudit:
    def test_audit_returns_counts(self):
        resp = client.post("/audit", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "total_entities" in data
        assert "total_policies" in data
        assert "total_rules" in data
        assert "total_violations" in data
        assert "violations" in data
        assert "summary" in data

    def test_audit_with_healthcare_config(self):
        resp = client.post("/audit", json={"config_path": HEALTHCARE_CONFIG})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_policies"] > 0
        assert data["total_entities"] > 0


class TestWhy:
    def test_why_entity_has_access(self):
        resp = client.post("/why", json={"entity": "Alice", "role": "budget_portal_access"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["entity"] == "Alice"
        assert data["role"] == "budget_portal_access"
        assert "has_access" in data
        assert "provenance" in data

    def test_why_entity_has_derived_access(self):
        resp = client.post("/why", json={"entity": "Alice", "role": "expense_system"})
        data = resp.json()
        assert resp.status_code == 200
        assert "has_access" in data
        assert "provenance" in data

    def test_why_entity_lacks_access(self):
        resp = client.post("/why", json={"entity": "Bob", "role": "budget_portal_access"})
        data = resp.json()
        assert resp.status_code == 200
        assert "provenance" in data


class TestConflict:
    def test_conflict_returns_structure(self):
        resp = client.post("/conflict", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "conflicts" in data
        assert "summary" in data


class TestImpact:
    def test_impact_add_rule(self):
        resp = client.post("/impact", json={
            "add_rules": ["all contractor are reporting_portal"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["rules_added"] == 1
        assert data["rules_removed"] == 0
        assert "summary" in data

    def test_impact_add_entity(self):
        resp = client.post("/impact", json={
            "add_entities": {"Zara": ["finance_employee", "contractor"]},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["entities_added"] == 1

    def test_impact_no_changes(self):
        resp = client.post("/impact", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["rules_added"] == 0
        assert data["rules_removed"] == 0
        assert data["entities_added"] == 0


class TestHealthAndInfo:
    def test_health(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "policies_loaded" in data

    def test_policies(self):
        resp = client.get("/policies")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) > 0

    def test_entities(self):
        resp = client.get("/entities")
        assert resp.status_code == 200
        data = resp.json()
        assert "employees" in data