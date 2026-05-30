"""LLM backends for Reduct Proxy — separated for clarity."""

import json


class LLMBackend:
    def complete(self, prompt: str, system: str, config) -> str:
        raise NotImplementedError


class OllamaBackend(LLMBackend):
    def complete(self, prompt: str, system: str, config) -> str:
        import urllib.request
        payload = json.dumps({
            "model": config.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": config.temperature},
        }).encode()
        resp = urllib.request.urlopen(
            urllib.request.Request(
                f"{config.ollama_url}/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
            ),
            timeout=120,
        )
        data = json.loads(resp.read().decode())
        content = data.get("message", {}).get("content", "")
        import re as _re
        if "" in content:
            content = content.split("")[-1].strip()
        json_match = _re.search(r'\{[^{}]*\}', content, _re.DOTALL)
        if json_match:
            pass  # Not JSON mode, just return text
        return content


class OpenAIBackend(LLMBackend):
    def complete(self, prompt: str, system: str, config) -> str:
        import urllib.request
        payload = json.dumps({
            "model": config.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }).encode()
        resp = urllib.request.urlopen(
            urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {config.openai_api_key}",
                },
            ),
            timeout=120,
        )
        data = json.loads(resp.read().decode())
        return data["choices"][0]["message"]["content"]


class AnthropicBackend(LLMBackend):
    def complete(self, prompt: str, system: str, config) -> str:
        import urllib.request
        payload = json.dumps({
            "model": config.llm_model,
            "max_tokens": config.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        resp = urllib.request.urlopen(
            urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": config.anthropic_api_key,
                    "anthropic-version": "2023-06-01",
                },
            ),
            timeout=120,
        )
        data = json.loads(resp.read().decode())
        return data["content"][0]["text"]


BACKENDS = {
    "ollama": OllamaBackend,
    "openai": OpenAIBackend,
    "anthropic": AnthropicBackend,
}