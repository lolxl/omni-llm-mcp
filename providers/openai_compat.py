# no-opencc
"""OpenAI-compatible HTTP provider (OpenAI cloud, Ollama, LM Studio)."""
import asyncio
import json
import os
import urllib.error
import urllib.request

from providers import known_secrets
from providers.egress_gate import refuse_payload
from providers.errors import reason_from_http_status, redact_secrets
from providers.usage import normalize_usage


async def run_openai_compatible(prompt, model, timeout_s, account_profile,
                                api_base="https://api.openai.com/v1"):
    """Zero-dependency HTTP call to an OpenAI-compatible REST API."""
    profile_suffix = str(account_profile).upper()
    api_key = os.environ.get("OPENAI_API_KEY_%s" % profile_suffix)
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY", "dummy-key-for-local")
    if api_key != "dummy-key-for-local":
        known_secrets.register(api_key, authoritative=True)

    data = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")

    req = urllib.request.Request(
        "%s/chat/completions" % api_base.rstrip("/"),
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer %s" % api_key,
        },
    )

    def fetch():
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode()
            except Exception:
                body = str(e)
            return {"error": True, "code": e.code, "body": body}
        except Exception as e:
            return {"error": True, "detail": str(e)}

    denied = refuse_payload(prompt)
    if denied:
        return denied

    res = await asyncio.to_thread(fetch)

    if res.get("error"):
        reason = (reason_from_http_status(res.get("code"))
                  if res.get("code") is not None else "runtime_error")
        detail = res.get("body", res.get("detail"))
        if isinstance(detail, str):
            detail = redact_secrets(detail)
        return {"ok": False, "reason": reason, "detail": detail}

    text = res.get("choices", [{}])[0].get("message", {}).get("content", "")
    out = {"ok": True, "text": text, "model": model}
    out["usage"] = normalize_usage("openai", res.get("usage"))
    return out
