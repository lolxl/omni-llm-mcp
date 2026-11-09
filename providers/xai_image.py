# no-opencc
"""xAI image generation HTTP provider."""
import asyncio
import base64
import json
import os
import tempfile
import urllib.error
import urllib.request
from providers import known_secrets
from providers.config import DEFAULT_TIMEOUT_S
from providers.egress_gate import refuse_payload
from providers.errors import reason_from_http_status, redact_secrets


async def run_xai_image(prompt, model="grok-2-image", n=1, response_format="b64_json", timeout_s=DEFAULT_TIMEOUT_S, account_profile="default", out_dir=None):
    api_base = os.environ.get("XAI_API_BASE", "https://api.x.ai/v1")
    api_key = os.environ.get("XAI_API_KEY_%s" % str(account_profile).upper()) or os.environ.get("XAI_API_KEY")
    if not api_key:
        return {"ok": False, "reason": "auth_error", "detail": "XAI_API_KEY not set (no local xAI fallback)"}
    known_secrets.register(api_key, authoritative=True)
    try:
        timeout_s = int(timeout_s)
    except (TypeError, ValueError):
        timeout_s = DEFAULT_TIMEOUT_S
    try:
        n = int(n)
        n = max(1, min(10, n))
    except (TypeError, ValueError):
        n = 1
    if response_format not in ("url", "b64_json"):
        response_format = "b64_json"
    data = json.dumps({"model": model, "prompt": prompt, "n": n, "response_format": response_format}).encode("utf-8")
    req = urllib.request.Request("%s/images/generations" % api_base.rstrip("/"), data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer %s" % api_key})
    denied = refuse_payload(prompt)
    if denied:
        return denied
    def fetch():
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as e:
            body = ""
            try: body = e.read().decode()
            except Exception: body = str(e)
            return {"error": True, "code": e.code, "body": body}
        except Exception as e:
            return {"error": True, "detail": str(e)}
    res = await asyncio.to_thread(fetch)
    if res.get("error"):
        code = res.get("code")
        reason = reason_from_http_status(code) if code is not None else "runtime_error"
        detail = res.get("body", res.get("detail"))
        if isinstance(detail, str):
            detail = redact_secrets(detail)
        return {"ok": False, "reason": reason, "detail": detail}
    data = res.get("data") or []
    if not data:
        return {"ok": False, "reason": "empty_response", "detail": "xAI returned no image data"}
    images = []
    if response_format == "url":
        for item in data:
            entry = {"url": item.get("url")}
            if item.get("revised_prompt"):
                entry["revised_prompt"] = item["revised_prompt"]
            images.append(entry)
    else:
        base_out = out_dir or tempfile.mkdtemp(prefix="xai_img_")
        os.makedirs(base_out, exist_ok=True)
        for i, item in enumerate(data):
            b64 = item.get("b64_json")
            if not b64:
                continue
            try:
                raw = base64.b64decode(b64)
                path = os.path.join(base_out, "img_%d.png" % i)
                with open(path, "wb") as f:
                    f.write(raw)
            except Exception as e:
                return {"ok": False, "reason": "runtime_error", "detail": redact_secrets(str(e))}
            entry = {"path": path}
            if item.get("revised_prompt"):
                entry["revised_prompt"] = item["revised_prompt"]
            images.append(entry)
    if not images:
        return {"ok": False, "reason": "empty_response", "detail": "no decodable images"}
    return {"ok": True, "model": model, "images": images, "n": len(images)}
