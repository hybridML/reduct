#!/usr/bin/env python3
"""CLI entry point for the Electron desktop app. Reads JSON from stdin, returns JSON to stdout."""

import sys
import json
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from proxy.reduct_proxy import ReductProxy, ProxyConfig

def main():
    args = json.loads(sys.argv[1])
    message = args.get('message', '')
    context = args.get('context', {})
    domain = args.get('domain', 'healthcare')
    backend = args.get('backend', 'ollama')
    model = args.get('model', 'qwen3:4b')

    config = ProxyConfig(
        domain=domain,
        llm_backend=backend,
        llm_model=model,
    )

    proxy = ReductProxy(config)

    try:
        result = proxy.chat(message=message, context=context, domain=domain)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e), "response": f"Error: {e}", "alias_input": message, "alias_output": "", "audit": {"pii_sent_to_llm": False, "error": True}}))

if __name__ == '__main__':
    main()