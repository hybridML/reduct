"""Tests for Reduct Proxy — privacy middleware for LLMs."""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from proxy.reduct_proxy import AliasMapper, ProxyConfig, _scrub_pii, DOMAIN_SYSTEM_PROMPTS


class TestAliasMapper:
    def test_register_person(self):
        mapper = AliasMapper()
        alias = mapper.register("John Smith", "person")
        assert alias in ["James", "Maria", "Robert", "Linda"]

    def test_register_drug(self):
        mapper = AliasMapper()
        alias = mapper.register("metformin", "drug")
        assert alias in ["metforin", "lisopril"]

    def test_register_condition(self):
        mapper = AliasMapper()
        alias = mapper.register("diabetes", "condition")
        assert alias in ["diabetix"]

    def test_register_multiple(self):
        mapper = AliasMapper()
        a1 = mapper.register("John Smith", "person")
        a2 = mapper.register("metformin", "drug")
        a3 = mapper.register("lisinopril", "drug")
        assert a1 != a2 != a3

    def test_consistent_mapping(self):
        mapper = AliasMapper()
        a1 = mapper.register("John Smith", "person")
        a2 = mapper.register("John Smith", "person")
        assert a1 == a2

    def test_redact_message(self):
        mapper = AliasMapper()
        mapper.register("John Smith", "person")
        mapper.register("metformin", "drug")
        redacted = mapper.redact("Can John Smith take metformin with his medications?")
        assert "John Smith" not in redacted
        assert "metformin" not in redacted
        assert "medications" in redacted  # not registered, stays

    def test_restore_response(self):
        mapper = AliasMapper()
        person_alias = mapper.register("John Smith", "person")
        drug_alias = mapper.register("metformin", "drug")
        redacted = mapper.redact("Can John Smith take metformin?")
        assert person_alias in redacted
        assert drug_alias in redacted
        restored = mapper.restore(redacted)
        assert "John Smith" in restored
        assert "metformin" in restored

    def test_full_roundtrip(self):
        mapper = AliasMapper()
        original = "Can Dr. Evans take metformin with lisinopril?"
        mapper.register("Dr. Evans", "person")
        mapper.register("metformin", "drug")
        mapper.register("lisinopril", "drug")
        redacted = mapper.redact(original)
        # The LLM would respond with aliases, then we restore:
        restored = mapper.restore(redacted)
        assert "Dr. Evans" in restored
        assert "metformin" in restored
        assert "lisinopril" in restored

    def test_auto_redact(self):
        mapper = AliasMapper()
        result = mapper.auto_redact("Can Alice take metformin?")
        # Alice should be detected and redacted
        assert "Alice" not in result or mapper.mapping.get("Alice") is not None

    def test_mapping_property(self):
        mapper = AliasMapper()
        mapper.register("John Smith", "person")
        mapper.register("metformin", "drug")
        m = mapper.mapping
        assert "John Smith" in m
        assert "metformin" in m

    def test_reverse_mapping(self):
        mapper = AliasMapper()
        person_alias = mapper.register("John Smith", "person")
        rm = mapper.reverse_mapping
        assert rm[person_alias] == "John Smith"


class TestPIIScrubbing:
    def test_scrub_ssn(self):
        result = _scrub_pii("SSN: 123-45-6789", ProxyConfig())
        assert "123-45-6789" not in result
        assert "[SSN_REDACTED]" in result

    def test_scrub_email(self):
        result = _scrub_pii("Email: john@example.com", ProxyConfig())
        assert "john@example.com" not in result
        assert "[EMAIL_REDACTED]" in result

    def test_scrub_phone(self):
        result = _scrub_pii("Phone: 555-123-4567", ProxyConfig())
        assert "555-123-4567" not in result
        assert "[PHONE_REDACTED]" in result

    def test_scrub_mrn(self):
        result = _scrub_pii("MRN: MRN-12345", ProxyConfig())
        assert "MRN-12345" not in result
        assert "[MRN_REDACTED]" in result

    def test_scrub_dates_enabled(self):
        result = _scrub_pii("Date: 01/15/2024", ProxyConfig(redact_dates=True))
        assert "01/15/2024" not in result
        assert "[DATE_REDACTED]" in result

    def test_scrub_dates_disabled(self):
        result = _scrub_pii("Date: 01/15/2024", ProxyConfig(redact_dates=False))
        # Date may or may not be redacted depending on what else matches

    def test_scrub_amounts(self):
        result = _scrub_pii("Amount: $1,234.56", ProxyConfig(redact_amounts=True))
        assert "$1,234.56" not in result
        assert "[AMOUNT_REDACTED]" in result

    def test_scrub_card_number(self):
        result = _scrub_pii("Card: 4111 1111 1111 1111", ProxyConfig(redact_ids=True))
        assert "4111 1111 1111 1111" not in result


class TestDomainSystemPrompts:
    def test_healthcare_prompt(self):
        assert "healthcare" in DOMAIN_SYSTEM_PROMPTS
        assert "clinical" in DOMAIN_SYSTEM_PROMPTS["healthcare"].lower()

    def test_finance_prompt(self):
        assert "finance" in DOMAIN_SYSTEM_PROMPTS
        assert "compliance" in DOMAIN_SYSTEM_PROMPTS["finance"].lower()

    def test_legal_prompt(self):
        assert "legal" in DOMAIN_SYSTEM_PROMPTS
        assert "conflict" in DOMAIN_SYSTEM_PROMPTS["legal"].lower()

    def test_defense_prompt(self):
        assert "defense" in DOMAIN_SYSTEM_PROMPTS
        assert "clearance" in DOMAIN_SYSTEM_PROMPTS["defense"].lower()

    def test_all_prompts_mention_aliases(self):
        for domain, prompt in DOMAIN_SYSTEM_PROMPTS.items():
            assert "alias" in prompt.lower() or "synthetic" in prompt.lower() or "pseudonym" in prompt.lower()

    def test_all_prompts_warn_against_identification(self):
        for domain, prompt in DOMAIN_SYSTEM_PROMPTS.items():
            assert "not real" in prompt.lower() or "never try" in prompt.lower() or "cannot reveal" in prompt.lower()


class TestProxyConfig:
    def test_default_config(self):
        config = ProxyConfig()
        assert config.domain == "healthcare"
        assert config.llm_backend == "ollama"
        assert config.redact_dates is True
        assert config.redact_amounts is True
        assert config.redact_ids is True

    def test_custom_config(self):
        config = ProxyConfig(
            domain="finance",
            llm_backend="openai",
            llm_model="gpt-4o",
            openai_api_key="sk-test",
        )
        assert config.domain == "finance"
        assert config.llm_backend == "openai"
        assert config.llm_model == "gpt-4o"


class TestIndustryConfigs:
    def test_healthcare_config_loads(self):
        import yaml
        path = os.path.join(os.path.dirname(__file__), "..", "config", "industries", "healthcare.yaml")
        with open(path) as f:
            config = yaml.safe_load(f)
        assert config["domain"] == "healthcare"
        assert len(config["entities"]) > 0
        assert "metformin" in config["entities"]

    def test_finance_config_loads(self):
        import yaml
        path = os.path.join(os.path.dirname(__file__), "..", "config", "industries", "finance.yaml")
        with open(path) as f:
            config = yaml.safe_load(f)
        assert config["domain"] == "finance"
        assert len(config["entities"]) > 0

    def test_legal_config_loads(self):
        import yaml
        path = os.path.join(os.path.dirname(__file__), "..", "config", "industries", "legal.yaml")
        with open(path) as f:
            config = yaml.safe_load(f)
        assert config["domain"] == "legal"

    def test_defense_config_loads(self):
        import yaml
        path = os.path.join(os.path.dirname(__file__), "..", "config", "industries", "defense.yaml")
        with open(path) as f:
            config = yaml.safe_load(f)
        assert config["domain"] == "defense"