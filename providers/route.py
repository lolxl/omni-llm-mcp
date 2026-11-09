# no-opencc
"""Table-driven model → provider resolution (single source of truth)."""
from providers.config import ROUTE_RULES


def resolve_route(model):
    """Return {provider, model, default_api_base?} for a single model name."""
    raw = str(model)
    low = raw.lower()
    for rule in ROUTE_RULES:
        kind = rule["kind"]
        match = rule["match"]
        hit = (low.startswith(match) if kind == "prefix" else match in low)
        if not hit:
            continue
        strip = rule.get("strip") or 0
        out = {"provider": rule["provider"], "model": raw[strip:] if strip else raw}
        if "default_api_base" in rule:
            out["default_api_base"] = rule["default_api_base"]
        return out
    return {"provider": "agy", "model": raw}


def provider_for_model(model):
    return resolve_route(model)["provider"]


def provider_supports_files(provider):
    """Providers that accept `files=`, each with different semantics:

    agy   — path/URL strings are appended into the prompt text; agy resolves them.
    codex — each entry must be an EXISTING local file, passed to `codex exec` as
            repeated `-i <path>` args (native vision). Codex reads it itself.
    grok  — each entry must be a local IMAGE file under OMNI_LLM_SOURCE_ROOTS; the
            server reads + base64-encodes it into one inline `--prompt-json` payload.

    Everything else (openai/ollama/lmstudio/claude/pi/cursor) LOUD-rejects with unsupported_files.
    """
    return provider in ("agy", "codex", "grok")
