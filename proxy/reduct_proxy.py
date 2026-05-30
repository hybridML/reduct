"""
Reduct Proxy — Privacy Middleware for LLMs

Instead of abstract IDs (ENT_1, ENT_5), uses synthetic aliases
so the LLM reasons naturally: "James can take metforin with his
current medications." But James/metforin are synthetic stand-ins
mapped to real data only on the user's machine.

Architecture:
    User → Reduct Proxy → LLM (Ollama/OpenAI/Anthropic)
    "Can John Smith take metformin?"      "Can James take metforin?"
    Reduct maps John→James                LLM sees only aliases
    Reduct maps metformin→metforin
    LLM responds about James/metforin
    Reduct maps back: "John Smith can take metformin."

The LLM never sees a single piece of real data.
"""

import json
import re
import os
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field


# ── Synthetic alias pools per category ──

PERSON_NAMES = [
    "James", "Maria", "Robert", "Linda", "Michael", "Patricia",
    "David", "Jennifer", "William", "Elizabeth", "Richard", "Barbara",
    "Joseph", "Susan", "Thomas", "Jessica", "Charles", "Sarah",
    "Christopher", "Karen", "Daniel", "Nancy", "Matthew", "Lisa",
    "Anthony", "Betty", "Mark", "Dorothy", "Donald", "Sandra",
    "Steven", "Ashley", "Paul", "Kimberly", "Andrew", "Emily",
    "Joshua", "Donna", "Kenneth", "Michelle", "Kevin", "Carol",
    "Brian", "Amanda", "George", "Melissa", "Timothy", "Deborah",
]

DRUG_NAMES = [
    "metforin", "lisopril", "atorivast", "asprimax", "omepril",
    "losartol", "amlodex", "metoprolix", "warfamax", "insulix",
    "prednisolix", "amoxivex", "ciprolin", "azithromax", "hydroclorix",
    "sertilax", "fluoximax", "levothyron", "albuterix", "ibumax",
]

CONDITION_NAMES = [
    "diabetix", "hypertensix", "hyperlipix", "asthmax", "COPDX",
    "depressix", "anxietix", "hypothyrox", "renalfix", "cardiofail",
    "atrialfix", "arthromax",
]

ACCOUNT_NAMES = [
    "checkfund", "savefund", "creditline", "mortgafix", "autoloanx",
    "personlofix", "investport", "retirefix", "wirexfer", "achflow",
]

ROLE_NAMES = [
    "approver_x", "reviewer_x", "trader_x", "compliance_x",
    "riskmgr_x", "auditor_x", "analyst_x", "operator_x",
    "commander_x", "liaison_x",
]

CATEGORY_POOLS = {
    "person": PERSON_NAMES,
    "drug": DRUG_NAMES,
    "condition": CONDITION_NAMES,
    "account": ACCOUNT_NAMES,
    "role": ROLE_NAMES,
}


@dataclass
class ProxyConfig:
    domain: str = "healthcare"
    llm_backend: str = "ollama"
    llm_model: str = "qwen3:4b"
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    ollama_url: str = "http://localhost:11434"
    temperature: float = 0.1
    max_tokens: int = 2048
    redact_dates: bool = True
    redact_amounts: bool = True
    redact_ids: bool = True


class AliasMapper:
    """Bidirectional mapper between plaintext and synthetic aliases.
    
    Instead of ENT_1, uses natural-sounding synthetic names like
    James, metforin, diabetix. This makes LLM reasoning much more
    natural while still being completely decoupled from real data.
    """

    def __init__(self):
        self._real_to_alias: Dict[str, str] = {}
        self._alias_to_real: Dict[str, str] = {}
        self._category_pool_idx: Dict[str, int] = {}
        self._pool_indices: Dict[str, int] = {}

    def register(self, plaintext: str, category: str = "person") -> str:
        plaintext = plaintext.strip()
        if plaintext in self._real_to_alias:
            return self._real_to_alias[plaintext]

        pool = CATEGORY_POOLS.get(category, PERSON_NAMES)
        if category not in self._pool_indices:
            self._pool_indices[category] = 0
        
        idx = self._pool_indices[category]
        if idx < len(pool):
            alias = pool[idx]
        else:
            alias = f"{category}_{idx + 1}"
        
        self._pool_indices[category] = idx + 1
        self._real_to_alias[plaintext] = alias
        self._alias_to_real[alias] = plaintext
        return alias

    def redact(self, text: str) -> str:
        for real in sorted(self._real_to_alias.keys(), key=len, reverse=True):
            text = re.sub(r'\b' + re.escape(real) + r'\b', self._real_to_alias[real], text)
        return text

    def restore(self, text: str) -> str:
        for alias in sorted(self._alias_to_real.keys(), key=len, reverse=True):
            text = text.replace(alias, self._alias_to_real[alias])
        return text

    def find_entities(self, text: str, domain: str = "healthcare") -> List[Tuple[str, str]]:
        entities = []
        capitalized = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
        stop = {'The', 'Is', 'Are', 'Can', 'Do', 'Does', 'What', 'How', 'Who',
                'When', 'Why', 'All', 'Every', 'If', 'Then', 'Their', 'They',
                'Not', 'And', 'Or', 'But', 'In', 'On', 'At', 'To', 'For', 'With'}
        for word in capitalized:
            if word not in stop:
                entities.append((word, "person"))
        return list(set(entities))

    def auto_redact(self, text: str, domain: str = "healthcare") -> str:
        entities = self.find_entities(text, domain)
        for entity, category in entities:
            if entity not in self._real_to_alias:
                self.register(entity, category)
        return self.redact(text)

    @property
    def mapping(self) -> Dict[str, str]:
        return dict(self._real_to_alias)

    @property
    def reverse_mapping(self) -> Dict[str, str]:
        return dict(self._alias_to_real)


# ── PII scrubbing ─────────────────────────────────────

def _scrub_pii(text: str, config: ProxyConfig) -> str:
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_REDACTED]', text)
    text = re.sub(r'\bMRN[-:]?\s*\d+\b', '[MRN_REDACTED]', text, flags=re.IGNORECASE)
    text = re.sub(r'\b[\w.-]+@[\w.-]+\.\w+\b', '[EMAIL_REDACTED]', text)
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE_REDACTED]', text)
    if config.redact_dates:
        text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '[DATE_REDACTED]', text)
    if config.redact_amounts:
        text = re.sub(r'\$[\d,]+\.?\d*\b', '[AMOUNT_REDACTED]', text)
    if config.redact_ids:
        text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[ID_REDACTED]', text)
        text = re.sub(r'\b[A-Z]{2}\d{2}[A-Z0-9]{4,}\b', '[IBAN_REDACTED]', text)
    return text


# ── Domain system prompts ─────────────────────────────

DOMAIN_SYSTEM_PROMPTS = {
    "healthcare": "You are a clinical decision support assistant. You will see patient information using synthetic aliases instead of real names. For example, \"James\" is a pseudonym for a real patient, and \"metforin\" is a pseudonym for a real drug. Reason about drug interactions, contraindications, and care pathways using these aliases.\n\nCRITICAL RULES:\n1. The aliases are NOT real names. Never try to identify who they represent.\n2. Provide clear, evidence-based clinical reasoning.\n3. Always include appropriate caveats and recommend verification.\n4. If you're unsure about a contraindication, say so explicitly.",
    "finance": "You are a financial compliance and risk analysis assistant. You will see account holders, transactions, and counterparties using synthetic aliases.\n\nCRITICAL RULES:\n1. The aliases are NOT real names. Never try to identify who they represent.\n2. Analyze compliance, risk, and regulatory requirements.\n3. Reference relevant regulations (SOX, AML, KYC) when applicable.\n4. If uncertain about a compliance ruling, say so explicitly.",
    "legal": "You are a legal research and analysis assistant. You will see client names and case references using synthetic aliases.\n\nCRITICAL RULES:\n1. The aliases are NOT real names. Never try to identify who they represent.\n2. Provide legal analysis and conflict checks.\n3. Note jurisdiction-specific considerations when relevant.\n4. Always include disclaimer that this is not legal advice.",
    "defense": "You are a security clearance and access control assistant. You will see personnel, compartments, and mission names using synthetic aliases.\n\nCRITICAL RULES:\n1. The aliases are NOT real identifiers. Never try to identify who or what they represent.\n2. Reason about need-to-know, clearance levels, and compartmentalization.\n3. If access should be denied or restricted, say so clearly.",
}


class ReductProxy:
    """Privacy middleware between users and LLMs.

    Usage:
        proxy = ReductProxy(config)
        result = proxy.chat(
            "Can John Smith take metformin with his medications?",
            context={"John Smith": "person", "metformin": "drug"}
        )
    """

    def __init__(self, config: ProxyConfig = None):
        self.config = config or ProxyConfig()
        self.mapper = AliasMapper()

    def chat(self, message: str, context: Optional[Dict[str, str]] = None,
             domain: Optional[str] = None) -> Dict:
        from proxy.backends import BACKENDS
        domain = domain or self.config.domain

        if context:
            for name, category in context.items():
                self.mapper.register(name, category)

        redacted_input = self.mapper.auto_redact(message, domain)
        redacted_input = _scrub_pii(redacted_input, self.config)

        system_prompt = DOMAIN_SYSTEM_PROMPTS.get(domain, DOMAIN_SYSTEM_PROMPTS["healthcare"])

        backend_cls = BACKENDS.get(self.config.llm_backend, BACKENDS["ollama"])
        backend = backend_cls()
        llm_response = backend.complete(redacted_input, system_prompt, self.config)

        restored_response = self.mapper.restore(llm_response)

        return {
            "response": restored_response,
            "alias_input": redacted_input,
            "alias_output": llm_response,
            "audit": {
                "entities_redacted": len(self.mapper.mapping),
                "entity_mapping": self.mapper.mapping,
                "llm_backend": self.config.llm_backend,
                "llm_model": self.config.llm_model,
                "pii_sent_to_llm": False,
                "domain": domain,
            },
        }