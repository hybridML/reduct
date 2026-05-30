# reduct

**Privacy-preserving reasoning engine for regulated industries.**

Reason about access control, compliance, and policy — without ever exposing entity names to the model.

## The Problem

Healthcare (HIPAA), finance (SOX), legal, and defense organizations can't send employee names, patient records, or financial data to third-party AI services. Current AI requires sending plaintext to a remote server — visible in logs, vulnerable to breaches, and non-compliant.

## The Insight

Reasoning and knowledge are different things. A symbolic logic solver operates on **variables**, not **names**. It doesn't need to know that "Alice" is a "finance_employee" — it only needs `IS(id_1, id_6)`. From `ALL(id_6, id_12)`, it derives `IS(id_1, id_12)`.

The mapping between abstract IDs and real names **never leaves your machine**. Even a compromised reasoning engine learns nothing about your entities.

This is structural abstraction — not redaction, not masking. The names are gone before the model sees them.

## Architecture

```
Your infrastructure (trust boundary)     Reasoning engine (no trust needed)
──────────────────────────────     ──────────────────────────────────────
"Alice is finance_employee"
        ↓
  EntityRegistry:
    Alice → id_1
    finance_employee → id_6
        ↓
  IS(id_1, id_6)  ──────────→   Solver: IS(id_1, id_6) + ALL(id_6, id_12)
                                 derives: IS(id_1, id_12)
  id_12 → budget_portal  ←──────
        ↓
"Alice has budget_portal_access"
```

## What It Does

1. **Parses** natural language into formal logic (rule-based or local LLM via Ollama)
2. **Queries** local data sources (HR policies, medical rules, access control)
3. **Derives** new conclusions via forward-chaining with provable correctness
4. **Provenance-tracked** — every derivation traces back to source facts
5. **Translates** conclusions back to natural language

All of this runs locally. No data leaves the machine.

## Reasoning Capabilities

| Rule | Pattern | Example |
|------|---------|---------|
| Universal instantiation | `ALL(X, Y) + IS(a, X) → IS(a, Y)` | All physicians → clinical access |
| Transitivity | `ALL(X, Y) + ALL(Y, Z) → ALL(X, Z)` | Clinical access → PHI → audit |
| Conditionals | `IF(X, Y) + IS(a, X) → IS(a, Y)` | If on_call → emergency access |
| Negation-as-failure | Closed-world: can't prove IS(a,X) → NOT_IS(a,X) | Bob isn't clinical staff |
| Temporal | `AFTER(N, X) + IS(a, X) → HOLDS(a, expires_after_N)` | After 90 days, terminated |
| Cardinality | `AT_MOST(N, role)` — violation if >N holders | At most 2 approvers |
| Exact cardinality | `EXACTLY(N, role)` — violation if ≠N holders | Exactly 1 primary physician |
| Mutual exclusivity | `MUTEX(A, B)` — no entity can hold both | Clinical and billing access are exclusive |
| Contradiction detection | `IS(a, X) + NOT_IS(a, X) → CONTRADICTION` | Conflicting access grants |

## Quick Start

```bash
pip install -r requirements.txt

# Run the CLI demo
python pipeline.py

# Start the API server
uvicorn api.server:app --reload --port 8000
```

### API Endpoints

```bash
# Health check
curl http://localhost:8000/health

# List loaded policies
curl http://localhost:8000/policies

# Reason about a query (default config)
curl -X POST http://localhost:8000/reason \
  -H "Content-Type: application/json" \
  -d '{"query": "Alice is finance_employee. What access does she have?"}'

# Reason about a query (healthcare config with NAF)
curl -X POST http://localhost:8000/reason \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Dr. Evans is physician. What access does she have?",
    "config_path": "config/domain_healthcare.yaml",
    "naf_enabled": true
  }'

# List constraints
curl http://localhost:8000/constraints

# Response includes:
# - conclusions: derived facts in plain English
# - abstract_facts: what the solver saw (no real names)
# - derived_facts: what the solver derived (no real names)
# - naf_conclusions: facts derived by negation-as-failure
# - explanations: full provenance chain for each derivation
# - derivations: structured derivation records
# - cardinality_violations: any AT_MOST/EXACTLY violations
# - mutex_violations: any mutual exclusivity violations
# - contradictions: any detected conflicts
```

### Example API Response

```json
{
  "query": "Alice is finance_employee. All finance_employee are budget_portal_access.",
  "conclusions": ["Alice is budget_portal_access"],
  "abstract_facts": ["ALL(id_2, id_1)", "IS(id_7, id_2)"],
  "derived_facts": ["IS(id_7, id_1)"],
  "naf_conclusions": [],
  "contradictions": [],
  "explanations": [
    "IS(id_7, id_1)\n  <- universal_instantiation\n    IS(id_7, id_2) (given [user_input])\n    ALL(id_2, id_1) (given [user_input])\n  (derived)"
  ],
  "derivations": [
    {
      "fact": "IS(id_7, id_1)",
      "rule": "universal_instantiation",
      "premises": ["IS(id_7, id_2)", "ALL(id_2, id_1)"],
      "source": "derived"
    }
  ],
  "cardinality_violations": [],
  "mutex_violations": []
}
```

## Configuration

Policies and entities are defined in `config/domain.yaml`. Organizations bring their own policies:

```yaml
policies:
  finance_access:
    description: "Financial system access control"
    rules:
      - "all finance_employee are budget_portal_access"
      - "all budget_portal_access are expense_system"

  medical:
    description: "HIPAA and medical data access"
    rules:
      - "all patient_record are phi"
      - "all phi are encrypted_at_rest"
      - "all encrypted_at_rest are hipaa_compliant"

entities:
  employees:
    Alice:
      roles: [finance_employee, full_time_employee]
    Bob:
      roles: [contractor]

constraints:
  cardinality:
    - "at most 2 approver"
    - "exactly 1 primary_physician"
  mutex:
    - "clinical_record_access and billing_record_access are mutually exclusive"

solver:
  max_iterations: 30
  detect_contradictions: true
  track_provenance: true
  naf_enabled: true
  naf_predicates:
    - clinical_record_access
    - billing_record_access
    - phi_access
```

### Healthcare/HIPAA Vertical

A complete HIPAA configuration is included at `config/domain_healthcare.yaml`:

- Minimum Necessary Standard (45 CFR 164.502(b))
- Role-Based Access (45 CFR 164.312(a))
- Segregation of Duties (clinical vs billing)
- Emergency break-glass access with audit
- Time-limited access for terminated employees
- Cardinality constraints (exactly 1 primary physician)

```bash
curl -X POST http://localhost:8000/reason \
  -H "Content-Type: application/json" \
  -d '{"query": "What access does Dr. Evans have?", "config_path": "config/domain_healthcare.yaml", "naf_enabled": true}'
```

## Provenance Tracking

Every derived fact traces back to its source — critical for audits:

```
IS(Alice, hipaa_compliant)
  <- universal_instantiation
    IS(Alice, encrypted_at_rest) (derived)
      <- universal_instantiation
        IS(Alice, phi) (derived)
          <- universal_instantiation
            IS(Alice, patient_record) [query:query_employee]
            ALL(patient_record, phi) [policy:hipaa_minimum_necessary]
        ALL(phi, encrypted_at_rest) [policy:hipaa_minimum_necessary]
    ALL(encrypted_at_rest, hipaa_compliant) [policy:hipaa_minimum_necessary]
```

Source tags include:
- `user_input` — facts from the user's query
- `policy:<name>` — facts loaded from domain config
- `constraint:<type>` — constraint facts from config
- `query:<tool>` — facts retrieved by tool calls
- `derived` — facts inferred by the solver
- `closed_world_assumption` — facts derived by NAF

## Results

| Approach | Novel Entity Accuracy | Notes |
|----------|----------------------|-------|
| Standard Transformer | 0% | Memorizes surface patterns, can't generalize |
| Expanded Vocab (500 entities) | 0% | More tokens = harder to memorize |
| **Neuro-Symbolic (this)** | **100%** | Solver operates on variables, not tokens |

## Tests

```bash
pytest tests/ -v
```

64 tests covering:
- Provenance solver (instantiation, transitivity, contradiction, deduplication)
- Enhanced solver (conditionals, NAF, temporal, cardinality, mutex, combined)
- Entity registry (lookup, abstraction, restoration, passthrough)
- Logic parser (ALL, IS, NOT_IS, IF, AFTER, AT_MOST, EXACTLY, MUTEX)
- Pipeline integration (basic, multi-hop, contradiction, abstract entities)
- Config loading (domain YAML, healthcare YAML, constraint parsing)
- API endpoints (health, policies, entities, reason, explain)

## Optional: Local LLM

If you have [Ollama](https://ollama.ai) running, the pipeline uses it for NL→Logic translation for richer natural language input. Without it, the rule-based parser handles the core patterns.

Priority order: `qwen2.5-coder:1.5b` > `qwen3:1.7b` > `qwen3:4b` > `qwen3:8b`

```bash
ollama pull qwen3:1.7b
python pipeline.py    # Auto-detects Ollama
```

## Key Files

- `api/server.py` — FastAPI server with full reasoning API
- `config/domain.yaml` — Default pluggable policy/entity config
- `config/domain_healthcare.yaml` — HIPAA/healthcare vertical config
- `pipeline.py` — Full agentic pipeline (NL → Logic → Tools → Solver → NL)
- `exp2_neurosymbolic/solver_enhanced.py` — Enhanced solver with conditionals, NAF, temporal, cardinality, mutex
- `exp2_neurosymbolic/solver_provenance.py` — Provenance-tracked logic solver
- `exp2_neurosymbolic/solver.py` — Base logic solver
- `tests/test_core.py` — Core test suite
- `tests/test_enhanced.py` — Enhanced reasoning test suite

## Target Market

Regulated industries where sending data to third-party AI is the blocker:
- **Healthcare/HIPAA** — patient data access reasoning without exposing patient names
- **Finance/SOX** — access control and segregation of duties audits
- **Legal** — privilege and conflict-of-interest checks
- **Defense/Classified** — clearance-based access reasoning

## License

Proprietary. Contact for enterprise licensing.