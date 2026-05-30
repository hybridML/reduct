"""
reduct — Live Examples

Demonstrates the neuro-symbolic architecture processing real-world scenarios
where the agent NEVER sees plaintext data. All entities are abstract variables.
The solver reasons purely on logical structure.

Run: python examples.py
"""

from exp2_neurosymbolic.solver import LogicSolver, parse_to_logic, logic_to_text, LogicAtom


def demo(title, description, premises, expected_conclusions):
    """Run a single demo with formatted output."""
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}")
    print(f"\n  {description}\n")

    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │  WHAT THE USER SEES (plaintext, never leaves device)    │")
    print("  └─────────────────────────────────────────────────────────┘\n")

    print("  Premises given by user:")
    for p in premises.split("  "):
        p = p.strip()
        if p:
            print(f"    • {p}")

    print("\n  ┌─────────────────────────────────────────────────────────┐")
    print("  │  WHAT THE CLOUD AGENT SEES (encrypted / abstracted)    │")
    print("  └─────────────────────────────────────────────────────────┘\n")

    solver = LogicSolver()
    atoms = parse_to_logic(premises)

    print("  Encrypted premises sent to cloud:")
    for a in atoms:
        solver.add_fact(a)
        print(f"    • {a}")

    solver.forward_chain()

    print("\n  ┌─────────────────────────────────────────────────────────┐")
    print("  │  SOLVER REASONING (on abstract variables only)          │")
    print("  └─────────────────────────────────────────────────────────┘\n")

    if solver.derived:
        print("  NEW conclusions derived by solver:")
        for a in sorted(solver.derived, key=str):
            print(f"    ★ {a}")
    else:
        print("  (no additional derivations needed)")

    if solver.contradictions:
        print(f"\n  ⚠️  CONTRADICTION DETECTED!")

    print("\n  ┌─────────────────────────────────────────────────────────┐")
    print("  │  DECRYPTED BACK FOR USER (local runtime translates)    │")
    print("  └─────────────────────────────────────────────────────────┘\n")

    if solver.derived:
        print("  New conclusions revealed to user:")
        for a in sorted(solver.derived, key=str):
            print(f"    → {logic_to_text(a)}")
    elif solver.contradictions:
        print("  → ACCESS DENIED: Policy contradiction detected")
    else:
        print("  → All premises confirmed (no new inferences needed)")

    for exp in expected_conclusions:
        derived_text = " ".join(logic_to_text(a).lower() for a in solver.derived)
        found = exp.lower() in derived_text
        if found:
            print(f"\n  ✅ Verified: \"{exp}\"")
        elif solver.contradictions and "contradiction" in exp.lower():
            print(f"\n  ✅ Contradiction detected as expected")


# ═══════════════════════════════════════════════════════════
# EXAMPLE 1: HR Access Control
# ═══════════════════════════════════════════════════════════

demo(
    "EXAMPLE 1: HR Access Control",
    "Alice works in Finance. All Finance employees have access to Budget Portal. "
    "The cloud agent never sees 'Alice' or 'Budget Portal' — only abstract variables.",
    "all ent_finance are ent_budget_portal  var_alice is ent_finance",
    ["var_alice is ent_budget_portal"],
)

# ═══════════════════════════════════════════════════════════
# EXAMPLE 2: Multi-hop Permission Chain
# ═══════════════════════════════════════════════════════════

demo(
    "EXAMPLE 2: Multi-hop Permission Chain",
    "Bob is a contractor → contractors are vendors → vendors have limited access → "
    "limited access includes Reporting Portal. 3-hop transitive derivation without "
    "the agent knowing what Bob, contractors, or portals are.",
    "all ent_contractor are ent_vendor  all ent_vendor are ent_limited_access  "
    "all ent_limited_access are ent_reporting_portal  ent_bob is ent_contractor",
    ["ent_bob is ent_reporting_portal", "ent_bob is ent_limited_access"],
)

# ═══════════════════════════════════════════════════════════
# EXAMPLE 3: Policy Violation / Contradiction
# ═══════════════════════════════════════════════════════════

demo(
    "EXAMPLE 3: Policy Violation Detection",
    "Charlie is marked as both terminated AND not terminated. The agent flags "
    "the contradiction without knowing what 'terminated' means or who Charlie is.",
    "var_charlie is ent_terminated  var_charlie is not ent_terminated",
    ["contradiction"],
)

# ═══════════════════════════════════════════════════════════
# EXAMPLE 4: Finance Approval Chain
# ═══════════════════════════════════════════════════════════

demo(
    "EXAMPLE 4: Finance Approval Chain",
    "Expenses over $10K → need VP approval → VP approval → goes to CFO review. "
    "Diana's expense qualifies. The agent derives CFO review without knowing "
    "what dollars, VPs, or CFOs are.",
    "all ent_over_10k are ent_vp_approval  all ent_vp_approval are ent_cfo_review  "
    "ent_diana_expense is ent_over_10k  ent_diana_expense is ent_vp_approval",
    ["ent_diana_expense is ent_cfo_review"],
)

# ═══════════════════════════════════════════════════════════
# EXAMPLE 5: HIPAA-like Medical Privacy
# ═══════════════════════════════════════════════════════════

demo(
    "EXAMPLE 5: Medical Privacy (HIPAA-like)",
    "All patient records are PHI. All PHI must be encrypted at rest. "
    "Dr. Evans accessed patient records. The agent derives that Dr. Evans' "
    "data must be encrypted — without knowing what PHI or encryption mean.",
    "all ent_patient_records are ent_phi  all ent_phi are ent_encrypted_at_rest  "
    "var_dr_evans is ent_patient_records",
    ["var_dr_evans is ent_encrypted_at_rest", "var_dr_evans is ent_phi"],
)

# ═══════════════════════════════════════════════════════════
# EXAMPLE 6: Cross-Org Trust Chain
# ═══════════════════════════════════════════════════════════

demo(
    "EXAMPLE 6: Cross-Organizational Trust",
    "Acme trusts Beta. Beta trusts Gamma. Through transitive trust, "
    "Acme users access Gamma resources — the agent has no idea what "
    "Acme, Beta, or Gamma are.",
    "all ent_acme_user are ent_beta_trusted  all ent_beta_trusted are ent_gamma_resource",
    ["all ent_acme_user are ent_gamma_resource"],
)

# ═══════════════════════════════════════════════════════════
# EXAMPLE 7: What The Agent CANNOT See
# ═══════════════════════════════════════════════════════════

print(f"\n{'─' * 70}")
print("  EXAMPLE 7: What the Cloud Agent CANNOT See")
print(f"{'─' * 70}")
print("""
  The cloud agent received these premises:
    ALL(ent_patient_records, ent_phi)
    ALL(ent_phi, ent_encrypted_at_rest)
    IS(var_dr_evans, ent_patient_records)

  And derived:
    IS(var_dr_evans, ent_encrypted_at_rest)

  But the agent DOES NOT KNOW:
    • That var_dr_evans is a person named "Dr. Evans"
    • That ent_patient_records means medical records
    • That ent_phi means Protected Health Information
    • That ent_encrypted_at_rest means AES-256 encryption

  It only knows: if ALL(X,Y) and ALL(Y,Z), then ALL(X,Z).
  And if ALL(X,Y) and IS(A,X), then IS(A,Y).

  These are STRUCTURAL rules. The data is MEANINGLESS to the agent.
  The encryption is inherent — not applied, but architectural.
""")

print(f"{'═' * 70}")
print("  BRAIN.ZIP — ARCHITECTURE SUMMARY")
print(f"{'═' * 70}")
print("""
  ┌──────────────────────────────────────────────────────────────┐
  │                                                             │
  │   User Device                  Cloud Agent                  │
  │   ───────────                  ───────────                   │
  │                                                             │
  │   Alice → var_alice            ALL(ent_finance, ent_x)     │
  │   Finance → ent_finance        IS(var_alice, ent_finance)  │
  │   Budget Portal → ent_x        ↓                            │
  │                                 DERIVED:                    │
  │                                 IS(var_alice, ent_x)         │
  │                                                             │
  │   var_alice=Alice              (doesn't know this)          │
  │   ent_x=Budget Portal          (doesn't know this)         │
  │                                                             │
  │   ──────── decryption layer ────────                         │
  │                                                             │
  │   IS(var_alice, ent_x)         → "Alice has Budget Portal"  │
  │                                                             │
  └──────────────────────────────────────────────────────────────┘

  The agent processes data correctly without ever seeing it.
  Reasoning is structural. Knowledge stays local. Encryption is inherent.
""")