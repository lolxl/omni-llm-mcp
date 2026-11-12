# no-opencc
"""Grok image_edit provider with host-fenced local source/output access."""
import asyncio
import os
import shutil
import subprocess
import tempfile

from providers import admission, cli_run, known_secrets
from providers.config import DEFAULT_TIMEOUT_S, GROK_EXTRA_AUTH_MARKERS
from providers.egress_gate import refuse_payload
from providers.host_fence import HostFenceError, prepare_host_fence
from providers.errors import classify_cli_error, exception_detail, safe_detail
from providers import vendor_bin

# R11 default. The request's model argument is not used by this backend.
GROK_IMAGE_MODEL = "grok-4.7"


def _image_argv(grok, cli_prompt, decision):
    """Today's gate keeps ``grok -p``. A catalog admission adds ``-m`` and effort."""
    argv = [grok, "-p", cli_prompt]
    if not isinstance(decision, dict) or decision.get("gate") != "catalog":
        return argv
    argv.extend(["-m", decision.get("model") or GROK_IMAGE_MODEL])
    effort = decision.get("effort")
    if effort:
        argv.extend(["--reasoning-effort", str(effort)])
    return argv


async def run_grok_image_edit(prompt, source_image, timeout_s=DEFAULT_TIMEOUT_S, out_dir=None, account_profile="default",
                             apply_aliases=False):
    if not str(source_image or "").strip():
        return {"ok": False, "reason": "runtime_error", "detail": "source_image is required for grok image_edit (it edits a reference image)"}
    grok = vendor_bin.which("grok")
    if not grok:
        return {"ok": False, "reason": "cli_missing", "detail": "grok not on PATH - install Grok CLI, then `grok login`"}
    try:
        timeout_s = int(timeout_s)
    except (TypeError, ValueError):
        timeout_s = DEFAULT_TIMEOUT_S
    base_out = out_dir or tempfile.mkdtemp(prefix="grok_img_edit_")
    os.makedirs(base_out, exist_ok=True)
    out_path = os.path.join(base_out, "image_edit.png")
    cli_prompt = "Use the image_edit tool. Source reference image (filesystem path): %s\n\n%s\n\nSave the resulting image as a PNG to %s then print the absolute saved path." % (source_image, prompt, out_path)
    home = os.path.expanduser("~")
    env = dict(os.environ, PATH="%s/.local/bin:%s" % (home, os.environ.get("PATH", "")))

    denied = refuse_payload(prompt, source_image, out_dir)
    if denied:
        return denied

    # The adapter runs on the process environment, whatever account_profile is.
    # Alias admission is opt-in. Dispatch's image path passes the flag.
    # The egress canary calls this seam without it and stays on the built-in.
    decision = admission.consider(
        "grok", "default", GROK_IMAGE_MODEL, None, host_env=os.environ,
        apply_aliases=apply_aliases, alias_image=True)
    if decision.get("refusal"):
        return admission.refusal_envelope("grok", "default", decision)

    source_path = os.path.expanduser(str(source_image))
    if not os.path.isabs(source_path):
        source_path = os.path.join(base_out, source_path)
    source_path = os.path.abspath(source_path)
    argv = _image_argv(grok, cli_prompt, decision)
    try:
        argv, child_env, _fence_meta = prepare_host_fence(
            argv, provider="grok", workdir=base_out,
            read_only=False, env=env,
            extra_read_paths=(source_path,),
        )
    except HostFenceError as exc:
        return exc.envelope

    def _run():
        return subprocess.run(argv, cwd=base_out, env=child_env, timeout=timeout_s, capture_output=True)

    # Spawned outside run_cli: the child's environment is registered here.
    known_secrets.register_child_env(child_env)
    try:
        proc = await asyncio.to_thread(_run)
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": "timeout", "detail": "grok image_edit exceeded %ds" % timeout_s}
    except FileNotFoundError:
        return {"ok": False, "reason": "cli_missing", "detail": "grok not on PATH - install Grok CLI, then `grok login`"}
    except Exception as e:
        return {"ok": False, "reason": "runtime_error", "detail": exception_detail(e)}
    known_secrets.register_child_env(child_env)  # a login file it refreshed
    # Bytes, decoded in one place: a credential with non-UTF-8 bytes is
    # registered as the decoder spelled it before any failure text is built.
    out = cli_run.decode_output(proc.stdout).strip()
    err = cli_run.decode_output(proc.stderr).strip()
    if proc.returncode != 0:
        reason = classify_cli_error(out, err, proc.returncode, extra_auth_markers=GROK_EXTRA_AUTH_MARKERS)
        return {"ok": False, "reason": reason, "detail": safe_detail(err or out or "exit %d" % proc.returncode)}
    if not os.path.exists(out_path):
        return {"ok": False, "reason": "empty_response", "detail": safe_detail(err or out or "grok produced no image at %s" % out_path)}
    return admission.stamp(
        {"ok": True, "model": "grok-image-edit", "images": [{"path": out_path}], "n": 1},
        decision)
