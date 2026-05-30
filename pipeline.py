"""
reduct — Fully Local Agentic Reasoning Pipeline

Architecture:
  1. Parse natural English → logic atoms (rule-based or local LLM via Ollama)
  2. Query local data sources (HR, Finance, Medical policies)
  3. Solver reasons on abstract variables (provably correct forward-chaining)
  4. Translate conclusions back → natural English

Privacy model:
  - Entity names are abstracted BEFORE reaching the reasoning engine
  - The solver only sees abstract IDs (id_1, id_2, etc.)
  - The label↔ID mapping never leaves the EntityRegistry
  - Even a compromised reasoning engine would learn nothing about the entities
  - The LLM (if used) also only sees abstract IDs — it never sees plaintext

This runs entirely locally. No data leaves the machine.
If Ollama is running locally with a compatible model, it is used for NL translation.
Otherwise, the rule-based parser handles it.
"""

import json
import re
from typing import Optional

from exp2_neurosymbolic.solver import LogicSolver, LogicAtom, logic_to_text


class EntityRegistry:
    def __init__(self):
        self._label_to_id = {}
        self._id_to_label = {}
        self._counter = 0

    def register(self, label):
        label = label.strip()
        if label in self._label_to_id:
            return self._label_to_id[label]
        if re.match(r'^(id|ent|var)_\d+$', label):
            self._label_to_id[label] = label
            self._id_to_label[label] = label
            return label
        self._counter += 1
        abs_id = f"id_{self._counter}"
        self._label_to_id[label] = abs_id
        self._id_to_label[abs_id] = label
        return abs_id

    def lookup_id(self, label):
        return self._label_to_id.get(label.strip())

    def to_abstract(self, text):
        result = text
        for label, abs_id in sorted(self._label_to_id.items(), key=lambda x: -len(x[0])):
            result = re.sub(r'\b' + re.escape(label) + r'\b', abs_id, result)
        return result

    def to_human(self, text):
        def replace_id(match):
            return self._id_to_label.get(match.group(0), match.group(0))
        return re.sub(r'(id|ent|var)_\d+', replace_id, text)


POLICIES = {
    "finance_access": {"rules": ["all finance_employee are budget_portal_access", "all budget_portal_access are expense_system", "all vp_approved are cfo_review", "all submitted_expense are audit_trail"]},
    "hr_policies": {"rules": ["all full_time_employee are health_benefits", "all contractor are limited_access", "all limited_access are reporting_portal", "all terminated_employee are revoked_access"]},
    "medical": {"rules": ["all patient_record are phi", "all phi are encrypted_at_rest", "all phi are access_audit", "all encrypted_at_rest are hipaa_compliant"]},
}

EMPLOYEES = {"Alice": ["finance_employee", "full_time_employee"], "Bob": ["contractor"], "Charlie": ["finance_employee", "terminated_employee"], "Diana": ["vp_approved", "full_time_employee"], "Dr. Evans": ["physician"]}

ROLES = {"finance_employee": "Alice, Diana", "contractor": "Bob", "vp_approved": "Diana", "physician": "Dr. Evans", "full_time_employee": "Alice, Diana", "terminated_employee": "Charlie"}


def query_policy(policy_name, registry):
    policy = POLICIES.get(policy_name, {"rules": []})
    results = []
    for rule_text in policy.get("rules", []):
        m = re.match(r'all\s+(\S+)\s+are\s+(\S+)', rule_text.strip())
        if m:
            s, o = registry.register(m.group(1)), registry.register(m.group(2))
            results.append(f"ALL({s}, {o})")
    return results

def query_employee(name, registry):
    name_id = registry.register(name)
    return [f"IS({name_id}, {registry.register(a)})" for a in EMPLOYEES.get(name, [])]

def query_who_has_role(role, registry):
    role_id = registry.register(role)
    return [f"IS({registry.register(n.strip())}, {role_id})" for n in ROLES.get(role, "").split(",")]


SYSTEM_PROMPT = """You are a logic translation system. Convert text into JSON with two fields.
Output format: {"logic": ["ALL(x,y)", "IS(x,y)", "NOT_IS(x,y)"], "tools": [{"name": "query_policy", "arguments": {"policy_name": "finance_access"}}]}
Rules: Use EXACT entity IDs from the input. ALL(subject,object) for "all X are Y". IS(subject,object) for "X is Y". NOT_IS(subject,object) for "X is not Y". Available tools: query_policy (finance_access|hr_policies|medical), query_employee (name), query_who_has_role (role). Output valid JSON only.

Examples:
Input: "id_1 is id_6. All id_6 are id_12."
Output: {"logic": ["IS(id_1, id_6)", "ALL(id_6, id_12)"], "tools": [{"name": "query_policy", "arguments": {"policy_name": "finance_access"}}]}

Input: "What access does id_2 have?"
Output: {"logic": [], "tools": [{"name": "query_employee", "arguments": {"name": "id_2"}}, {"name": "query_policy", "arguments": {"policy_name": "hr_policies"}}]}"""

_OLLAMA_MODELS = ["qwen2.5-coder:1.5b", "qwen3:1.7b", "qwen3:4b", "qwen3:8b"]
_OLLAMA_TIMEOUT = 90


def _check_ollama():
    try:
        import urllib.request
        resp = urllib.request.urlopen(urllib.request.Request("http://localhost:11434/api/tags"), timeout=3)
        data = json.loads(resp.read().decode())
        available = {m["name"] for m in data.get("models", [])}
        for model in _OLLAMA_MODELS:
            if model in available:
                return model
    except:
        pass
    return None


def _normalize_llm_output(raw):
    logic, tools = [], []
    for atom in raw.get("logic", raw.get("logic_atoms", raw.get("atoms", []))):
        if isinstance(atom, str):
            logic.append(atom.strip())
        elif isinstance(atom, dict):
            for pred in ["ALL", "IS", "NOT_IS"]:
                if pred in atom:
                    args = atom[pred]
                    if isinstance(args, list) and len(args) >= 2:
                        logic.append(f"{pred}({args[0]}, {args[1]})")
                    break
    for tc in raw.get("tools", raw.get("tool_calls", [])):
        if isinstance(tc, dict):
            name = tc.get("name", "")
            args = tc.get("arguments", tc.get("args", tc.get("parameters", {})))
            if isinstance(args, str):
                args = {"policy_name": args} if name == "query_policy" else {"name": args}
            if name:
                tools.append({"name": name, "arguments": args if isinstance(args, dict) else {}})
        elif isinstance(tc, str):
            tools.append({"name": tc, "arguments": {}})
    return {"logic": logic, "tools": tools}


def _validate_atoms(atoms):
    pattern = re.compile(r'^(ALL|IS|NOT_IS)\((\S+),\s*(\S+)\)$')
    return [a for a in atoms if pattern.match(a)]


def _call_ollama(abstracted, model):
    import urllib.request
    try:
        payload = json.dumps({"model": model, "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": abstracted}], "stream": False, "format": "json", "options": {"temperature": 0.0}}).encode()
        resp = urllib.request.urlopen(urllib.request.Request("http://localhost:11434/api/chat", data=payload, headers={"Content-Type": "application/json"}), timeout=_OLLAMA_TIMEOUT)
        data = json.loads(resp.read().decode())
        content = data.get("message", {}).get("content", "")
        # Handlethinking tags from qwen3
        import re as _re
        if "" in content:
            content = content.split("")[-1].strip()
        json_match = _re.search(r'\{[^{}]*\}', content, _re.DOTALL)
        if json_match:
            content = json_match.group(0)
        return _normalize_llm_output(json.loads(content))
    except:
        return None


def _detect_tool_needs(text):
    tools = []
    pk = {"finance_access": ["finance", "budget", "expense", "access", "vp_approved", "cfo"], "hr_policies": ["hr", "benefits", "employee", "contractor", "terminated", "approved"], "medical": ["patient", "phi", "medical", "hipaa", "encrypted", "physician", "record"]}
    for name in EMPLOYEES:
        if name.lower() in text.lower():
            if not any(t["name"] == "query_employee" and t.get("arguments", {}).get("name") == name for t in tools):
                tools.append({"name": "query_employee", "arguments": {"name": name}})
                for role in EMPLOYEES.get(name, []):
                    for pn, kw in pk.items():
                        if role in kw or any(k in role.lower() for k in ["finance", "contractor", "terminated", "full_time", "vp"]):
                            if not any(t["name"] == "query_policy" and t.get("arguments", {}).get("policy_name") == pn for t in tools):
                                tools.append({"name": "query_policy", "arguments": {"policy_name": pn}})
    lower = text.lower()
    for pn, kw in pk.items():
        if any(k in lower for k in kw):
            if not any(t["name"] == "query_policy" and t.get("arguments", {}).get("policy_name") == pn for t in tools):
                tools.append({"name": "query_policy", "arguments": {"policy_name": pn}})
    return tools


_STOP_WORDS = {"all", "every", "the", "a", "an", "is", "are", "not", "what", "how", "who", "where", "when", "why", "can", "do", "does", "has", "have", "and", "or", "but", "in", "on", "at", "to", "of", "for", "with"}

def _preregister_known_entities(registry):
    for name in EMPLOYEES:
        registry.register(name)
    for rn, rl in ROLES.items():
        registry.register(rn)
        for n in rl.split(","):
            registry.register(n.strip())
    for pn in POLICIES:
        for rule in POLICIES[pn]["rules"]:
            m = re.match(r'all\s+(\S+)\s+are\s+(\S+)', rule)
            if m:
                registry.register(m.group(1))
                registry.register(m.group(2))

def _extract_and_register_entities(text, registry):
    for abs_id in re.findall(r'(id_\d+|ent_\d+|var_\d+)', text):
        registry.register(abs_id)
    for word in re.findall(r'\b([a-zA-Z](?:[a-zA-Z_]*[a-zA-Z])?)\b', text):
        if word.lower() in _STOP_WORDS or len(word) <= 2:
            continue
        if '_' in word or word[0].isupper() or word.lower() in {'phi', 'hipaa', 'alice', 'bob', 'charlie', 'diana'}:
            registry.register(word)
    for phrase in ["budget portal access", "expense system", "finance employee", "full time employee", "VP approval", "VP approved", "CFO review", "health benefits", "limited access", "reporting portal", "revoked access", "access audit", "encrypted at rest", "HIPAA compliant", "patient record", "budget portal", "patient records"]:
        if phrase.lower() in text.lower():
            registry.register(phrase.replace(" ", "_"))
    for name in re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', text):
        if name not in ("All", "The", "Is", "Not", "What", "How"):
            registry.register(name)

def _translate_rulebased(text, registry):
    logic_atoms = []
    _preregister_known_entities(registry)
    _extract_and_register_entities(text, registry)
    abstracted = registry.to_abstract(text)
    al = abstracted.lower()
    for m in re.finditer(r'all\s+(id_\d+)\s+are\s+(id_\d+)', al):
        logic_atoms.append(f"ALL({m.group(1)}, {m.group(2)})")
    for m in re.finditer(r'all\s+(\S+)\s+are\s+(\S+)', abstracted):
        s_id, o_id = registry.lookup_id(m.group(1)), registry.lookup_id(m.group(2))
        if s_id and o_id:
            atom = f"ALL({s_id}, {o_id})"
            if atom not in logic_atoms:
                logic_atoms.append(atom)
    for m in re.finditer(r'(id_\d+)\s+is\s+not\s+(id_\d+)', al):
        logic_atoms.append(f"NOT_IS({m.group(1)}, {m.group(2)})")
    for m in re.finditer(r'(id_\d+)\s+is\s+an?\s+(id_\d+)', al):
        logic_atoms.append(f"IS({m.group(1)}, {m.group(2)})")
    for m in re.finditer(r'(id_\d+)\s+is\s+(id_\d+)', al):
        atom = f"IS({m.group(1)}, {m.group(2)})"
        if atom not in logic_atoms:
            logic_atoms.append(atom)
    for m in re.finditer(r'every\s+(id_\d+)\s+is\s+(id_\d+)', al):
        logic_atoms.append(f"ALL({m.group(1)}, {m.group(2)})")
    if not logic_atoms:
        for m in re.finditer(r'all\s+(\S+)\s+are\s+(\S+)', text, re.IGNORECASE):
            s, o = registry.register(m.group(1)), registry.register(m.group(2))
            logic_atoms.append(f"ALL({s}, {o})")
        for m in re.finditer(r'(\S+)\s+is\s+not\s+(\S+)', text, re.IGNORECASE):
            s, o = registry.register(m.group(1)), registry.register(m.group(2))
            logic_atoms.append(f"NOT_IS({s}, {o})")
        for m in re.finditer(r'(\S+)\s+is\s+an?\s+(\S+)', text, re.IGNORECASE):
            if m.group(1).lower() not in ("all", "every", "not"):
                s, o = registry.register(m.group(1)), registry.register(m.group(2))
                logic_atoms.append(f"IS({s}, {o})")
    return {"logic": logic_atoms, "tools": _detect_tool_needs(text)}

def translate(nl_input, registry, use_llm=False, model=None):
    rule_result = _translate_rulebased(nl_input, registry)
    if not use_llm:
        return rule_result
    model = model or _check_ollama()
    if not model:
        return rule_result
    _preregister_known_entities(registry)
    _extract_and_register_entities(nl_input, registry)
    abstracted = registry.to_abstract(nl_input)
    llm_result = _call_ollama(abstracted, model)
    if not llm_result:
        return rule_result
    llm_result["logic"] = _validate_atoms(llm_result.get("logic", []))
    if not llm_result["logic"] and not llm_result["tools"]:
        return rule_result
    existing_atoms = set(rule_result["logic"])
    merged_logic = list(rule_result["logic"])
    for atom in llm_result.get("logic", []):
        if atom not in existing_atoms:
            merged_logic.append(atom)
    existing_tools = {(t["name"], json.dumps(t.get("arguments", {}), sort_keys=True)) for t in rule_result["tools"]}
    merged_tools = list(rule_result["tools"])
    for tc in llm_result.get("tools", []):
        tc_key = (tc.get("name", ""), json.dumps(tc.get("arguments", {}), sort_keys=True))
        if tc_key not in existing_tools:
            merged_tools.append(tc)
    return {"logic": merged_logic, "tools": merged_tools}

def execute_tool_call(tc, registry):
    name = tc["name"]
    args = tc.get("arguments", tc.get("args", {}))
    if name == "query_policy":
        return query_policy(args.get("policy_name", args), registry)
    elif name == "query_employee":
        return query_employee(args.get("name", args), registry)
    elif name == "query_who_has_role":
        return query_who_has_role(args.get("role", args), registry)
    return []

def run_agent(user_query, use_llm=False, model=None, verbose=True):
    registry = EntityRegistry()
    solver = LogicSolver()
    if verbose:
        print(f"\n{'='*70}\n  USER QUERY: \"{user_query}\"\n{'='*70}\n")
        print("  +----------------------------------------------------+")
        print("  |  STEP 1: Parse user query                         |")
        print("  +----------------------------------------------------+")
    translation = translate(user_query, registry, use_llm=use_llm, model=model)
    logic_strings = translation.get("logic", [])
    if verbose:
        print("\n  Direct logic atoms:")
        for a in logic_strings: print(f"    * {a}")
        if translation.get("tools"):
            print("\n  Tool calls:")
            for tc in translation["tools"]: print(f"    * {tc['name']}({', '.join(f'{k}={v}' for k,v in tc.get('arguments',{}).items())})")
    tool_facts = []
    if translation.get("tools"):
        if verbose: print("\n  +----------------------------------------------------+\n  |  STEP 2: Query local data sources                 |\n  +----------------------------------------------------+")
        for tc in translation["tools"]:
            results = execute_tool_call(tc, registry)
            if verbose: print(f"\n  -> {tc['name']}({', '.join(f'{k}={v}' for k,v in tc.get('arguments',{}).items())}):")
            for r in results: print(f"      {r}")
            tool_facts.extend(results)
    if verbose: print("\n  +----------------------------------------------------+\n  |  STEP 3: Forward-chain reasoning                  |\n  +----------------------------------------------------+")
    all_atom_strings = list(logic_strings) + tool_facts
    all_atoms = []
    for atom_str in all_atom_strings:
        m = re.match(r'(ALL|IS|NOT_IS)\((\S+),\s*(\S+)\)', atom_str.strip())
        if m:
            atom = LogicAtom(predicate=m.group(1), subject=m.group(2), obj=m.group(3))
            all_atoms.append(atom)
            solver.add_fact(atom)
    if verbose:
        print("\n  Facts in solver:")
        for a in sorted(solver.facts, key=str): print(f"    {a}\n      = {registry.to_human(str(a))}")
    solver.forward_chain()
    if verbose:
        if solver.contradictions:
            print("\n  WARNING: CONTRADICTION:")
            for a, b in solver.contradictions: print(f"    {a}  <->  {b}\n      = {registry.to_human(logic_to_text(a))}  <->  {registry.to_human(logic_to_text(b))}")
        if solver.derived:
            print("\n  Derived conclusions:")
            for a in sorted(solver.derived, key=str): print(f"    {a}\n      = {registry.to_human(logic_to_text(a))}")
        if not solver.contradictions and not solver.derived: print("\n  (No new derivations)")
    if verbose: print("\n  +----------------------------------------------------+\n  |  STEP 4: Translate back to human language          |\n  +----------------------------------------------------+\n")
    derived_nl = []
    if solver.contradictions:
        if verbose:
            for a, b in solver.contradictions: print(f"    X CONTRADICTION: {registry.to_human(logic_to_text(a))} <-> {registry.to_human(logic_to_text(b))}")
        derived_nl = ["CONTRADICTION"]
    if solver.derived:
        for a in sorted(solver.derived, key=str): nl = registry.to_human(logic_to_text(a)); derived_nl.append(nl); print(f"    >> {nl}")
    if not solver.contradictions and not solver.derived:
        if verbose: print("    (All given facts confirmed)")
    if verbose: print()
    return {"query": user_query, "logic_input": [str(a) for a in all_atoms], "tool_calls": translation.get("tools", []), "tool_results": tool_facts, "derived_logic": [str(a) for a in solver.derived], "derived_nl": derived_nl, "contradictions": [(str(a), str(b)) for a, b in solver.contradictions]}

if __name__ == "__main__":
    model = _check_ollama()
    mode = f"LLM ({model})" if model else "Rule-Based"
    print(f"{'='*70}\n  reduct -- Local Agentic Reasoning Pipeline ({mode} Mode)\n  All data stays on this machine. No cloud calls.\n{'='*70}")
    use_llm = model is not None
    run_agent("Alice is a finance_employee. All finance_employee are budget_portal_access.", use_llm=use_llm, model=model)
    run_agent("What access does Bob have?", use_llm=use_llm, model=model)
    run_agent("All patient_record are phi. All phi are encrypted_at_rest.", use_llm=use_llm, model=model)
    run_agent("Diana is full_time_employee. Diana is not full_time_employee.", use_llm=use_llm, model=model)
    run_agent("Diana is vp_approved. What else can we derive?", use_llm=use_llm, model=model)
    run_agent("All finance_employee are budget_portal_access. All budget_portal_access are expense_system. Alice is finance_employee.", use_llm=use_llm, model=model)
    run_agent("All id_99 are id_100. All id_100 are id_101.", use_llm=use_llm, model=model)