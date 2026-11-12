# no-opencc
"""Typed failure classification for CLI text, exceptions, and HTTP status.

Also holds the Codex JSONL error-event extractor, because every classify
call site already imports this module.
"""
import contextvars
import functools
import json
import re
import signal

from providers import known_secrets
from providers.config import (
    AUTH_MARKERS,
    RATE_LIMIT_MARKERS,
    HTTP_STATUS_REASONS,
)


def signal_name(rc):
    """Map a wait-status returncode to ``SIGKILL(9)``, else ``signal N``.

    ``None`` and non-negative values return ``signal ?`` (not ``signal 0``).
    """
    try:
        rc = int(rc)
    except (TypeError, ValueError):
        return "signal ?"
    if rc >= 0:
        return "signal ?"
    signo = -rc
    try:
        name = signal.Signals(signo).name
    except ValueError:
        return "signal %d" % signo
    return "%s(%d)" % (name, signo)


def signal_exit_detail(cli_name, returncode, stdout="", stderr=""):
    """Typed detail for a child that died by signal; output is never the reason."""
    last = stderr or stdout or ""
    return safe_detail("%s killed by %s; last output: %s" % (
        cli_name, signal_name(returncode), last))


# Distinctive key prefixes: none of these occurs inside an ordinary word, so
# they are masked wherever they appear, glued to other text or not (a JSON
# "\n", an ANSI colour code, "%3D" and a letter all end in a word character).
SECRET_KEY_RE = re.compile(
    r"sk-ant-[A-Za-z0-9_-]{16,}"
    r"|sk-(?:proj|svcacct|admin)-[A-Za-z0-9_-]{16,}"
    r"|xai-(?=[A-Za-z0-9]{20})[A-Za-z0-9_-]+"
    r"|gsk_[A-Za-z0-9]{20,}"
    r"|gh[pousr]_[A-Za-z0-9]{20,}"
    r"|github_pat_[A-Za-z0-9_]{20,}"
    r"|hf_[A-Za-z0-9]{20,}"
    r"|AIza[A-Za-z0-9_-]{30,}"
    r"|ya29\.[A-Za-z0-9_-]{20,}"
    r"|1//[A-Za-z0-9_-]{20,}")
_KEY_HINTS = ("sk-ant-", "sk-proj-", "sk-svcacct-", "sk-admin-", "xai-", "gsk_",
              "ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_", "hf_",
              "AIza", "ya29.", "1//")

# Generic "sk-" also occurs inside ordinary words ("desk-notes-archive-2026",
# "task-runner-config-file"). The run is split at the next "sk-" so a real key
# after a word is judged on its own. A match at a word boundary is a key when
# its body is 16+ characters. A match glued to a letter or digit is a key when
# its body (the maximal run of letters, digits, hyphens and underscores) is 24+
# characters or holds a 20+ character alphanumeric segment, whatever the case
# (lowercase hex keys, "sk-lf-<uuid>"); ordinary hyphenated words are shorter.
# The WHOLE body is masked either way.
_SK_RUN_RE = re.compile(r"sk-[A-Za-z0-9_-]*?(?=sk-|[^A-Za-z0-9_-]|\Z)")
_SK_SEGMENT_SPLIT_RE = re.compile(r"[-_]")
_ASCII_ALNUM = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


def _sk_run(match):
    run = match.group(0)
    body = run[3:]
    start = match.start()
    glued = start > 0 and match.string[start - 1] in _ASCII_ALNUM
    if not glued:
        return "<redacted>" if len(body) >= 16 else run
    if len(body) >= 24:
        return "<redacted>"
    for segment in _SK_SEGMENT_SPLIT_RE.split(body):
        if len(segment) >= 20 and all(c in _ASCII_ALNUM for c in segment):
            return "<redacted>"
    return run


BEARER_RE = re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE)
BASIC_RE = re.compile(
    r"(Authorization\s*:\s*Basic\s+)[A-Za-z0-9._~+/=-]+", re.IGNORECASE)
# A JWT is three dot-joined base64url segments, header and payload both
# starting "eyJ". Left-maximal runs of three segments are matched (linear:
# a run is entered once, at its start) and the callback masks from the first
# "eyJ" of the first segment, so a token glued behind "\\n", an ANSI code or
# "%3D" is covered and a word such as "keyJson.data.value" is not.
_JWT_RUN_RE = re.compile(r"(?<![\w-])[\w-]+\.[\w-]+\.[\w-]+")


def _jwt_start(match):
    """Index in the run where the JWT begins, or None when it is not one."""
    first, second, _third = match.group(0).split(".", 2)
    at = first.find("eyJ")
    if at == 0 or (at > 0 and second.startswith("eyJ")):
        return at
    return None


def _jwt_run(match):
    at = _jwt_start(match)
    if at is None:
        return match.group(0)
    return match.group(0)[:at] + "<redacted>"


# Secret-named fields whose value is opaque (no recognisable prefix). The
# VALUE is masked and the field name stays so the detail remains readable.
# Forms: JSON ("k": "v"), JSON embedded in a string at any depth of escaping
# (\\\"k\\\": \\\"v\\\"), Python repr ('k': 'v'), and unquoted k=v, k%3Dv or k: v.
# Quoted values run to the closing unescaped quote; an unquoted value runs to
# whitespace or a delimiter. No length is capped, and every branch consumes
# its text once, so the scan stays linear. The name is not anchored on the left:
# it can follow a letter ("\\nrefresh_token=", "%26access_token=").
_FIELD_NAMES = (
    r"(?:refresh[_-]?token|access[_-]?token|id[_-]?token|client[_-]?secret"
    r"|api[_-]?key)")
SECRET_FIELD_RE = re.compile(
    _FIELD_NAMES + r"(?:\\*[\"'])?\s*(?:[:=]|%3[dD])\s*(?:"
    r"(?P<pa>\\+\")(?P<va>(?:[^\\]|\\+(?!\"))+)"
    r"|(?P<pb>\")(?P<vb>(?:\\.|[^\"\\])+)"
    r"|(?P<pc>')(?P<vc>(?:\\.|[^'\\])+)"
    r"|(?P<pd>)(?P<vd>[^\s&\"'<>,;\\]+)"
    r")",
    re.IGNORECASE)
_FIELD_HINTS = ("token", "secret", "key")


def _secret_field(match):
    for tag in "abcd":
        if match.group("p" + tag) is not None:
            cut = match.start("v" + tag) - match.start()
            return match.group(0)[:cut] + "<redacted>"
    return match.group(0)


def redact_secrets(text):
    """Mask API keys, OAuth tokens, bearer tokens and JWTs.

    Run before any truncation. Prefix-shaped keys are replaced whole; for the
    opaque token fields only the value is replaced. Every pattern is linear in
    the input: bounded repetition, left anchors, and a cheap substring test
    before each pass.

    The values omni knows it handed to a child (providers.known_secrets) are
    masked first and verbatim, in every encoding, so the shape patterns below
    are a supplementary layer, not the only one.
    """
    text = known_secrets.mask_known(text)
    low = text.lower()
    if "bearer" in low:
        text = BEARER_RE.sub("Bearer <redacted>", text)
    if "eyJ" in text:
        text = _JWT_RUN_RE.sub(_jwt_run, text)
    if "basic" in low:
        text = BASIC_RE.sub(r"\1<redacted>", text)
    if any(hint in low for hint in _FIELD_HINTS):
        text = SECRET_FIELD_RE.sub(_secret_field, text)
    if any(hint in text for hint in _KEY_HINTS):
        text = SECRET_KEY_RE.sub("<redacted>", text)
    if "sk-" in text:
        text = _SK_RUN_RE.sub(_sk_run, text)
    return text


# An address: bounded local part, bounded labels, an alphabetic top-level
# label, so "python@3.12" and "pkg@2.0.14" are left alone. The left anchor
# and the bounds keep the scan linear.
EMAIL_RE = re.compile(
    r"(?<![\w.+-])[\w.+-]{1,64}@[A-Za-z0-9-]{1,63}"
    r"(?:\.[A-Za-z0-9-]{1,63}){0,15}"
    r"\.(?:xn--[A-Za-z0-9-]{1,59}|[A-Za-z]{2,24})(?![\w-]|\.[\w-])")


# What the apiKeyHelper check refused before this module masked more shapes.
# It is kept verbatim, with no cut at an inner "sk-", so acceptance is at least
# as strict as it always was.
_PRIOR_HELPER_KEY_RE = re.compile(
    r"sk-ant-[A-Za-z0-9_-]+|sk-[A-Za-z0-9_-]{16,}|xai-[A-Za-z0-9_-]{16,}")


def contains_key_literal(text):
    """True when *text* embeds a key-shaped literal with its full body.

    Narrower than what redact_secrets touches: a command that merely names a
    secret field ("fetch-key --api_key=prod", a header carrying a variable)
    carries no key. Refused: anything the earlier check refused, the
    distinctive-prefix keys, a long generic "sk-" body, a bearer token or a JWT.
    """
    if _PRIOR_HELPER_KEY_RE.search(text) or SECRET_KEY_RE.search(text):
        return True
    # A generic "sk-" body (the maximal run of letters, digits, hyphens and
    # underscores) is a key when it is 24+ characters or holds a 20+
    # alphanumeric segment, whatever precedes it: the rule of a glued match.
    for run in _SK_RUN_RE.finditer(text):
        body = run.group(0)[3:]
        if len(body) >= 24 or any(
                len(seg) >= 20 and all(c in _ASCII_ALNUM for c in seg)
                for seg in _SK_SEGMENT_SPLIT_RE.split(body)):
            return True
    if BEARER_RE.search(text):
        return True
    return any(_jwt_start(m) is not None for m in _JWT_RUN_RE.finditer(text))


def mask_text(text):
    """Secrets and e-mail addresses masked; no cut."""
    text = redact_secrets(text)
    if "@" in text:
        text = EMAIL_RE.sub("<email>", text)
    return text


def safe_detail(text, limit=300):
    """Mask secrets and e-mail addresses, THEN cut to ``limit``.

    A key or address straddling the cut is never left as a visible prefix,
    which a mask applied after the cut cannot catch (the half-written text no
    longer matches the pattern).
    """
    if text is None:
        return ""
    return mask_text(text if isinstance(text, str) else str(text))[:limit]


# The claude CLI's notice that a still-valid sign-in will end soon. It names
# "claude auth login", so the generic auth markers match it, yet it reports
# nothing about the call that carried it.
#
# The match is the EXACT notice template and nothing looser:
#
#   Warning: this sign-in ends between <DATE> and <DATE> (in <N> day[s] at the
#   earliest). Renewing it does not move those dates. Run claude auth login
#   before then, or use claude setup-token for unattended jobs.
#
# Only <DATE> (YYYY-MM-DD) and <N> (digits) vary. The words come in that order,
# joined only by whitespace (one line break allowed, for wrapping) and ANSI
# colour codes; no extra word or line may sit between them. The match starts a
# line. Anything else, a reworded notice or a real error line that quotes part
# of it, is left alone and still reads as the auth failure it may be.
_ANSI = r"(?:\x1b\[[0-9;]*m)*"
_NL = r"(?:\r\n|\n|\r)"
_SIGNIN_SEP = (_ANSI + r"(?:[ \t]+(?:" + _NL + r"[ \t]*)?|" + _NL + r"[ \t]*)"
               + _ANSI)
_SIGNIN_DATE = r"\d{4}-\d{2}-\d{2}"
_SIGNIN_NOTICE_TOKENS = (
    r"this", r"sign-in", r"ends", r"between", _SIGNIN_DATE, r"and", _SIGNIN_DATE,
    r"\(in", r"\d+", r"days?", r"at", r"the", r"earliest\)\.",
    r"Renewing", r"it", r"does", r"not", r"move", r"those", r"dates\.",
    r"Run", r"claude", r"auth", r"login", r"before", r"then,", r"or", r"use",
    r"claude", r"setup-token", r"for", r"unattended", r"jobs\.")
_SIGNIN_EXPIRY_NOTICE = re.compile(
    r"^" + _ANSI + _ANSI.join(tuple("Warning:"))
    + _SIGNIN_SEP + _SIGNIN_SEP.join(_SIGNIN_NOTICE_TOKENS)
    + _ANSI + r"(?:\r\n|\n|\r|\Z)",
    re.MULTILINE)


def strip_signin_expiry_warning(text):
    """Return *text* without the claude sign-in expiry notice.

    The notice is advice, not a failure: classification and failure detail
    must read what is left. Only the matched notice (with its own line end)
    is removed; every other line, blank lines and CRLF included, is kept as
    it was. Text without an exact notice comes back unchanged.
    """
    text = text if isinstance(text, str) else ("" if text is None else str(text))
    if "sign-in" not in text:
        return text
    stripped = _SIGNIN_EXPIRY_NOTICE.sub("", text)
    return stripped if stripped.strip() else ""


# Fields of an envelope that carry failure prose (never the model's answer).
FAILURE_TEXT_FIELDS = frozenset((
    "detail", "error", "message", "metrics_error", "last_error", "warning"))
# Outside a failure subtree these hold the successful model output or a
# stored artifact body: masking an answer would corrupt it.
_UNMASKED_FIELDS = frozenset(("text", "preview", "content"))
_MAX_DEPTH = 32


def _mask_failure(value, depth):
    """Every string at any depth under a failure-text field is masked,
    whatever key holds it (a ``text`` inside an error object included)."""
    if isinstance(value, str):
        return mask_text(value)
    if isinstance(value, (dict, list, tuple)):
        if depth > _MAX_DEPTH:
            return mask_text(str(value))
        if isinstance(value, dict):
            return {k: _mask_failure(v, depth + 1) for k, v in value.items()}
        items = [_mask_failure(v, depth + 1) for v in value]
        return tuple(items) if isinstance(value, tuple) else items
    return value


def mask_envelope(value, _depth=0):
    """A copy of an envelope with failure-text fields masked.

    The one function applied where an envelope leaves omni (MCP response, CLI
    output, job result file). It walks nested dicts, lists and tuples; every
    string under a failure-text field name is masked at any depth, while the
    successful model output (``text``) elsewhere is left alone. Past the depth
    limit a subtree outside any failure field is returned unchanged (successful
    content is never rewritten) and one inside a failure field is replaced by
    its masked ``str()``.
    The per-site masking at each construction site stays the inner layer;
    this is the layer that catches a site that forgot it.
    """
    if _depth == 0:
        # Known values are masked in EVERY string, whatever field holds it
        # (including a successful answer) and at any depth.
        value = known_secrets.mask_known_deep(value)
    if not isinstance(value, (dict, list, tuple)):
        return value
    if _depth > _MAX_DEPTH:
        return value
    if not isinstance(value, dict):
        items = [mask_envelope(v, _depth + 1) for v in value]
        return tuple(items) if isinstance(value, tuple) else items
    out = {}
    for key, item in value.items():
        if key in _UNMASKED_FIELDS:
            out[key] = item
        elif key in FAILURE_TEXT_FIELDS:
            out[key] = _mask_failure(item, _depth + 1)
        else:
            out[key] = mask_envelope(item, _depth + 1)
    return out


def exception_detail(exc, limit=300):
    """Type name plus message of a LOCAL exception, secrets masked then cut.

    A local exception never picks the failure reason (see
    classify_exception_message): its text can carry a caller path or prompt
    that merely contains "login" or "429". Redact before truncating so a key
    straddling the limit is not left half visible.
    """
    return redact_secrets("%s: %s" % (type(exc).__name__, exc))[:limit]


LOCAL_ORIGIN_KEY = "_local_origin"


class _LocalOrigin(str):
    """Type of the local-origin marker value.

    Only omni's own code can build one. Anything parsed from a child process
    (JSON, text) is a plain ``str``/``dict``, so :func:`is_local_failure`
    cannot be satisfied by vendor output, whatever it prints.
    """

    __slots__ = ()


_LOCAL_ORIGIN = _LocalOrigin("local")


def mark_local(envelope):
    """Stamp a failure envelope as raised by omni's own code, not a vendor run.

    Call it only where omni's own exception boundary built the envelope (a local
    catch-all, a local refusal, a spawn failure): never from anything derived
    from child output. The stamp is internal to dispatch; it is stripped before
    an envelope leaves it (see :func:`strip_local_marker`).
    """
    envelope[LOCAL_ORIGIN_KEY] = _LOCAL_ORIGIN
    return envelope


def is_local_failure(envelope):
    """True only for an envelope stamped by :func:`mark_local` in this process."""
    return (isinstance(envelope, dict)
            and type(envelope.get(LOCAL_ORIGIN_KEY)) is _LocalOrigin)


def strip_local_marker(envelope):
    """The envelope without any local-origin key (same object when it has none)."""
    if isinstance(envelope, dict) and LOCAL_ORIGIN_KEY in envelope:
        return {k: v for k, v in envelope.items() if k != LOCAL_ORIGIN_KEY}
    return envelope


class SpawnTracker:
    """Whether a vendor child process has been spawned for one provider call."""

    __slots__ = ("spawned",)

    def __init__(self):
        self.spawned = False


# One tracker per provider request (all account attempts of one call).
_REQUEST_TRACKER = contextvars.ContextVar("omni_spawn_request", default=None)


def note_spawned(tracker):
    """Record that a vendor child exists (``run_cli`` spawned it or returned).

    Marks the per-attempt tracker and the request-level tracker, if any.
    """
    if tracker is not None:
        tracker.spawned = True
    request = _REQUEST_TRACKER.get()
    if request is not None:
        request.spawned = True


def local_before_spawn(fn):
    """Stamp a failure envelope as local when no child was spawned.

    Wraps a provider coroutine that takes a ``_spawn`` keyword and calls
    :func:`note_spawned` as soon as ``run_cli`` returns. A failure envelope
    returned before that point was produced by omni's own code (argument and
    profile checks, a missing binary, a refusal by omni's own gate, a fence
    refusal, a spawn error) and gets the local stamp. Anything after the
    child ran, including an exception while parsing its output, is a
    vendor-side failure and stays unstamped. ``run_cli`` itself raising is
    omni's own failure and stays local.
    """
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        tracker = SpawnTracker()
        res = await fn(*args, _spawn=tracker, **kwargs)
        if (not tracker.spawned and isinstance(res, dict)
                and res.get("ok") is False):
            # In place: the envelope is the one omni just built, and callers
            # that hold a reference to it (a raised refusal) must see the same
            # object they always saw.
            mark_local(res)
        return res
    return wrapper


def local_request_before_spawn(unstamped_reasons=()):
    """Stamp a provider request's failure as local when NO attempt spawned.

    Wraps a public provider entry point (account selection plus its attempts).
    A refusal produced by the account layer itself (``selection.exhausted()``,
    a profile or ring config error, a selection deadline, a ``run_selected``
    refusal before any attempt) is omni's own, as long as no attempt in this
    request ever spawned a child. ``unstamped_reasons`` lists reasons that stay
    unstamped even then: codex passes ``timeout``, because its account
    selection runs the codex binary for the quota check, so a selection
    deadline can be vendor time.
    """
    def decorate(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            tracker = SpawnTracker()
            token = _REQUEST_TRACKER.set(tracker)
            try:
                res = await fn(*args, **kwargs)
            finally:
                _REQUEST_TRACKER.reset(token)
            if (not tracker.spawned and isinstance(res, dict)
                    and res.get("ok") is False
                    and res.get("reason") not in unstamped_reasons):
                mark_local(res)
            return res
        return wrapper
    return decorate


def codex_safe_output(stdout):
    """Codex's own error-event text, secrets masked; "" when there is none.

    The only Codex output a failure envelope may carry. stderr is never
    used: it can carry the caller's prompt echoed back, and a prompt can
    imitate any log line, so no filter on it can establish provenance.
    """
    return redact_secrets(codex_error_text(stdout))


def codex_has_events(stdout):
    """True when stdout carries at least one Codex ``--json`` event object."""
    for line in (stdout if isinstance(stdout, str) else "").splitlines():
        try:
            event = json.loads(line)
        except (TypeError, ValueError):
            continue
        if not isinstance(event, dict):
            continue
        # Current events carry a top-level type; legacy ones wrap it in msg.
        if isinstance(event.get("type"), str):
            return True
        inner = event.get("msg")
        if isinstance(inner, dict) and isinstance(inner.get("type"), str):
            return True
    return False


def codex_classification_stderr(stdout, stderr):
    """The stderr a Codex failure may be classified from: none once Codex
    has spoken in events.

    With ``--json`` Codex reports its failures as events, so they decide.
    stderr only counts for a run that failed before emitting any event
    (not signed in, bad config). Text on stderr has no provenance: a
    prompt or a tool's output there must not turn a failure into an
    account failure and cool the seat down.
    """
    return "" if codex_has_events(stdout) else (stderr or "")


def codex_failure_detail(stdout, returncode):
    """Bounded detail for a failed Codex run: its error event, else the exit.

    Masking happens before the 300-character cut, so a cut cannot strip
    the prefix the masks key on.
    """
    text = codex_safe_output(stdout)
    if not text:
        text = "exit %d" % (returncode if returncode is not None else -1)
    return text[:300]


def codex_error_text(stdout, with_codes=False):
    """Join text from error-typed events on a Codex ``--json`` stdout stream.

    An event is error-typed when its top-level ``type`` contains ``error``,
    when a nested ``item`` / ``msg`` dict has a ``type`` containing ``error``,
    or when its ``error`` key holds a truthy payload — a non-empty str or a
    non-empty dict. A missing key or ``error: null`` does not error-type
    the event.

    THE INVARIANT: harvest is a structural property of which node is
    error-typed, not a scan of the event. When the event is error-typed
    only by the ``error`` key, strings are taken from that subtree only.
    When a ``type`` field (top-level or nested ``item`` / ``msg``)
    contains ``error``, strings are taken from that error-typed node —
    its ``message``, ``text``, and ``error`` — and never from a sibling.
    ``agent_message`` items and any other non-error-typed node are never
    read, so no part of the result can be driven by text the caller or
    the model can author. That is a structural property of what this
    function reads, not a list of exclusions. Non-JSON lines are ignored.
    The raw stream is never returned.
    """
    text = stdout if isinstance(stdout, str) else ""
    stripped = text.strip()
    if not stripped:
        return ""

    def type_contains_error(typ):
        return isinstance(typ, str) and "error" in typ.lower()

    def truthy_error_payload(value):
        if isinstance(value, str) and value:
            return True
        if isinstance(value, dict) and value:
            return True
        return False

    def is_error_typed(event):
        if type_contains_error(event.get("type")):
            return True
        for nest in ("item", "msg"):
            inner = event.get(nest)
            if isinstance(inner, dict) and type_contains_error(inner.get("type")):
                return True
        return truthy_error_payload(event.get("error"))

    def own(node):
        """Strings of a node already known to be an error: its message,
        text and error subtree (never an agent_message)."""
        found = []
        if not isinstance(node, dict) or node.get("type") == "agent_message":
            return found
        for key in ("message", "text"):
            val = node.get(key)
            if isinstance(val, str) and val.strip():
                found.append(val.strip())
        code = node.get("code")
        if with_codes and isinstance(code, str) and code.strip():
            # usage_limit_reached -> "usage limit reached": the classifier's
            # markers, from the structured code Codex sends without a message.
            found.append(code.strip().replace("_", " "))
        found.extend(subtree(node.get("error")))
        for nest in ("item", "msg"):
            inner = node.get(nest)
            if isinstance(inner, dict) and type_contains_error(inner.get("type")):
                found.extend(own(inner))
        return found

    def subtree(err):
        if isinstance(err, str) and err.strip():
            return [err.strip()]
        if isinstance(err, dict):
            return own(err)
        return []

    def take_strings(event):
        """Harvest only error nodes: the event when its type says error,
        else its error-typed item/msg and its error payload — never the
        event's own message/text, which no error marker vouches for."""
        if type_contains_error(event.get("type")):
            return own(event)
        found = []
        for nest in ("item", "msg"):
            inner = event.get(nest)
            if isinstance(inner, dict) and type_contains_error(inner.get("type")):
                found.extend(own(inner))
        found.extend(subtree(event.get("error")))
        return found

    parts = []
    for line in stripped.splitlines():
        try:
            event = json.loads(line)
        except (TypeError, ValueError):
            continue
        if not isinstance(event, dict) or not is_error_typed(event):
            continue
        parts.extend(take_strings(event))
    return " ".join(parts)


def classify_cli_error(stdout, stderr, returncode=None, extra_auth_markers=()):
    """Map CLI stdout/stderr text to a typed failure reason.

    A negative returncode other than -1 (the timeout sentinel, never passed
    here) is ``signal_exit`` before any text matching. Prefer auth over
    rate_limit when both markers appear.
    """
    if (isinstance(returncode, int) and not isinstance(returncode, bool)
            and returncode < -1):
        return "signal_exit"
    low = ((stderr or "") + " " + (stdout or "")).lower()
    auth_markers = AUTH_MARKERS + tuple(extra_auth_markers or ())
    if any(k in low for k in auth_markers):
        return "auth_error"
    if any(k in low for k in RATE_LIMIT_MARKERS):
        return "rate_limit"
    # Narrow match for agy "not recognized as a known model" (unknown_model).
    # Keep this substring tight so ordinary runtime errors stay runtime_error.
    if "not recognized as a known model" in low:
        return "unknown_model"
    return "runtime_error"


def classify_exception_message(err):
    """Map an exception / free-text error to a typed reason."""
    low = str(err or "").lower()
    if any(k in low for k in RATE_LIMIT_MARKERS):
        return "rate_limit"
    if any(k in low for k in AUTH_MARKERS):
        return "auth_error"
    return "runtime_error"


def reason_from_http_status(code):
    """Map an HTTP status code from an OpenAI-compatible API to a typed reason."""
    if code is None:
        return "runtime_error"
    try:
        code = int(code)
    except (TypeError, ValueError):
        return "runtime_error"
    return HTTP_STATUS_REASONS.get(code, "runtime_error")
