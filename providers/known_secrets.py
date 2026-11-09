# no-opencc
"""Process-local registry of the credential values omni itself knows.

The key-shape patterns in :mod:`providers.errors` guess at what a secret looks
like and keep missing shapes. This module masks the values verbatim instead:
omni hands credentials to vendor children (environment variables, account
overrides, the OAuth files in an account home), so it knows what to hide.

Rules of the registry:

* It lives in this process's memory only. It is never written to disk, never
  put in an environment, argv, job file or log, and has no ``repr`` that shows
  a value. ``clear()`` exists for tests.
* It is additive for the life of the process. Failure text is masked after the
  child returned, so a value cannot be dropped when the spawn ends.
* Two classes of registration. *Authoritative* values are ones omni itself
  sets or passes (a known credential variable, a profile override): only the
  length bounds apply. *Heuristic* values are discovered from a name pattern
  or from a file the child can write: the bounds plus content filters apply,
  so a planted common phrase cannot make omni mask ordinary text.
* A value is registered at 8 to 8192 characters, in both classes. The upper
  bound keeps the cost of matching bounded; a longer string is not a
  credential omni can mask verbatim.
* Each value is masked in every derived encoding (raw, JSON-escaped, URL
  encoded, base64 at the three byte alignments), as text and as UTF-8 bytes.
  All matches of all patterns are merged, so overlapping or nested values are
  replaced as one span and the longest always wins. A value holding bytes
  that are not UTF-8 (a lone surrogate from the environment) is registered as
  ``errors="replace"`` spells it, and, because that spelling depends on the
  bytes next to it, also as the decoder actually spelled it in a child's
  output (``register_decoded_spans``). The decoded text a caller parses is
  never changed; the failure text built from it is masked by the text pass.
* A byte stream masks a cut-off back (8+ characters) of a value at the start
  of the stream and of the first buffer after the known values changed, as
  the text path does at the start of a text. Until the next chunk or
  ``flush()`` the buffer is held back whenever it is, so far, the start-aligned
  beginning of a registered form's tail of 8 or more bytes (whatever it says);
  the hold is bounded by the longest value.
* Login files are read from exact locations only (see ``_AUTH_LAYOUT``), never
  by key name at any depth, because the child can write them. A token that is
  not a JWT must be 32 or more characters of two or more character classes.
"""
import base64
import codecs
import json
import os
import re
import stat
import threading
from urllib.parse import quote, quote_plus

MARKER = "<redacted>"
_BMARKER = MARKER.encode("ascii")
MIN_LEN = 8
MAX_LEN = 8192
_DEEP = 200
_MIN_DISTINCT = 4
_FILE_TOKEN_CAP = 16
_DENY = frozenset((
    "true", "false", "null", "none", "undefined", "changeme", "password",
    "redacted", "<redacted>", "localhost", "dummy-key-for-local", "disabled",
    "not-needed", "placeholder", "example", "default",
))

_lock = threading.Lock()
_values = set()
_str_patterns = ()      # longest first
_byte_patterns = ()     # longest first
_byte_by_first = {}     # first byte -> tuple of byte patterns
_prefix8 = frozenset()  # first 8 chars of every text pattern
_suffix8 = frozenset()  # last 8 chars of every text pattern
_bprefix8 = frozenset()
_bsuffix8 = frozenset()
_max_len = 0
_max_blen = 0
_undecodable = ()       # byte patterns that are not valid UTF-8, longest first
_file_seen = {}         # path -> stat fingerprint (never values)


def clear():
    """Forget everything (test hook)."""
    global _values, _str_patterns, _byte_patterns, _byte_by_first
    global _prefix8, _suffix8, _bprefix8, _bsuffix8, _max_len, _max_blen
    global _undecodable
    with _lock:
        _values = set()
        _str_patterns = ()
        _byte_patterns = ()
        _byte_by_first = {}
        _prefix8 = _suffix8 = _bprefix8 = _bsuffix8 = frozenset()
        _max_len = _max_blen = 0
        _undecodable = ()
        _file_seen.clear()


def active():
    return bool(_str_patterns)


# --- derivation ---------------------------------------------------------

_PCT_RE = re.compile(r"%[0-9A-F]{2}")


def _try(fn, *args):
    try:
        return fn(*args)
    except Exception:  # never let a value travel inside an exception
        return None


def _escapes(value):
    out = set()
    for form in (_try(lambda: json.dumps(value)[1:-1]),
                 _try(lambda: json.dumps(value, ensure_ascii=False)[1:-1]),
                 _try(lambda: repr(value)[1:-1])):
        if not form:
            continue
        slashed = form.replace("/", "\\/")
        out.update((form, slashed, json.dumps(form)[1:-1], json.dumps(slashed)[1:-1]))
    return out


def _b64_interiors(value):
    """Base64 substrings fixed by the value alone, at the 3 byte alignments."""
    raw = _encode(value)
    out = set()
    if raw is None:
        return out
    for pad in range(3):
        enc = base64.b64encode(b"\0" * pad + raw).decode("ascii")
        start = -(-8 * pad // 6)
        stop = (8 * (pad + len(raw))) // 6
        interior = enc[start:stop]
        if len(interior) >= MIN_LEN:
            out.add(interior)
            out.add(interior.replace("+", "-").replace("/", "_"))
    return out


def _encode(text):
    """UTF-8 bytes of *text*; a lone surrogate degrades, it never raises."""
    for errors in ("strict", "surrogateescape", "surrogatepass"):
        try:
            return text.encode("utf-8", errors)
        except UnicodeEncodeError:
            continue
    return None


def _replaced_spelling(value):
    """What a decoder using ``errors="replace"`` makes of the value's bytes.

    A credential that holds bytes which are not UTF-8 reaches the failure
    text through ``bytes.decode(errors="replace")`` (run_cli does this for the
    child's output) with a U+FFFD for each undecodable run. That spelling is
    not the value, nor any escaping of it, so it is derived from the same
    bytes the child wrote. Empty when the value decodes to itself, or when
    fewer than ``MIN_LEN`` characters that are not U+FFFD would be left to
    match (a run of replacement characters is not a secret).

    The spelling depends on the bytes around the value (a lead byte that
    follows it can complete a truncated sequence at its end), so it is the
    second line: ``register_decoded_spans`` registers what the decoder made of
    the actual bytes.
    """
    raw = _encode(value)
    if raw is None:
        return ""
    decoded = raw.decode("utf-8", errors="replace")
    if decoded == value or sum(c != "\ufffd" for c in decoded) < MIN_LEN:
        return ""
    return decoded


def _variants(value):
    forms = {value}
    forms |= _try(_escapes, value) or set()
    replaced = _try(_replaced_spelling, value)
    if replaced:
        forms.add(replaced)
        forms |= _try(_escapes, replaced) or set()
    for text in (_try(quote, value, ""), _try(quote_plus, value)):
        if text:
            forms.add(text)
            forms.add(_PCT_RE.sub(lambda m: m.group(0).lower(), text))
    forms |= _try(_b64_interiors, value) or set()
    return {f for f in forms if len(f) >= MIN_LEN}


def _plausible(value, authoritative):
    if not isinstance(value, str):
        return False
    if len(value) < MIN_LEN or len(value) > MAX_LEN:
        return False
    if authoritative:
        return True
    if len(set(value)) < _MIN_DISTINCT or value.strip().lower() in _DENY:
        return False
    return True


def register_many(values, authoritative=False):
    """Add values (and their encodings) with ONE matcher rebuild.

    Every variant is derived before anything is added, so a value is never
    half-registered. True when something new was added.
    """
    try:
        return _register_many(values, authoritative)
    except Exception:  # registration must never break a spawn or carry a value
        return False


def _register_many(values, authoritative):
    staged = []
    for value in values:
        if not isinstance(value, str):
            continue
        forms = [value]
        stripped = value.strip()
        if stripped != value:
            forms.append(stripped)
        for cand in forms:
            if _plausible(cand, authoritative) and cand not in _values:
                variants = _try(_variants, cand)
                if variants:
                    staged.append((cand, variants))
    if not staged:
        return False
    with _lock:
        extra, fresh = set(), []
        for cand, variants in staged:
            if cand not in _values and cand not in fresh:
                fresh.append(cand)
                extra |= variants
        if fresh:
            _rebuild(extra)       # a failure here leaves nothing half-added
            _values.update(fresh)
    return bool(fresh)


def register(value, authoritative=False):
    """Add one value; True when something new was added."""
    return register_many((value,), authoritative)


def _is_utf8(raw):
    try:
        raw.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def _rebuild(extra):
    global _str_patterns, _byte_patterns, _byte_by_first
    global _prefix8, _suffix8, _bprefix8, _bsuffix8, _max_len, _max_blen
    global _undecodable
    ordered = tuple(sorted(set(_str_patterns) | extra, key=lambda p: (-len(p), p)))
    byte_set = set()
    for pat in ordered:
        raw = _encode(pat)
        if raw is not None:
            byte_set.add(raw)
    bytes_ordered = tuple(sorted(byte_set, key=lambda p: (-len(p), p)))
    by_first = {}
    for pat in bytes_ordered:
        by_first.setdefault(pat[:1], []).append(pat)
    prefix8 = frozenset(p[:8] for p in ordered)
    suffix8 = frozenset(p[-8:] for p in ordered)
    bprefix8 = frozenset(p[:8] for p in bytes_ordered)
    bsuffix8 = frozenset(p[-8:] for p in bytes_ordered)
    undecodable = tuple(p for p in bytes_ordered if not _is_utf8(p))
    # Published together; readers take one global at a time, each is valid alone.
    _prefix8, _suffix8, _bprefix8, _bsuffix8 = prefix8, suffix8, bprefix8, bsuffix8
    _max_len = max((len(p) for p in ordered), default=0)
    _max_blen = max((len(p) for p in bytes_ordered), default=0)
    _byte_by_first = {k: tuple(v) for k, v in by_first.items()}
    _undecodable = undecodable
    _byte_patterns = bytes_ordered
    _str_patterns = ordered


# --- masking ------------------------------------------------------------

def _spans(text, pats, starts=None):
    """Merged-per-pattern [start, end) spans of every occurrence, overlapping
    ones included. With *starts* (a list) every raw occurrence's
    (start, end) is appended too."""
    spans = []
    for pat in pats:
        if pat not in text:
            continue
        size = len(pat)
        i = text.find(pat)
        run_start = run_end = -1
        while i >= 0:
            if starts is not None:
                starts.append((i, i + size))
            if i <= run_end:
                run_end = i + size
            else:
                if run_end >= 0:
                    spans.append((run_start, run_end))
                run_start, run_end = i, i + size
            i = text.find(pat, i + 1)
        spans.append((run_start, run_end))
    return spans


def _merge(spans):
    spans.sort()
    merged = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def _replace(text, merged, marker):
    if not merged:
        return text
    out, pos = [], 0
    for start, end in merged:
        out.append(text[pos:start])
        out.append(marker)
        pos = end
    out.append(text[pos:])
    return text[:0].join(out)


def _tail_edge(text, pats, prefix8, max_len):
    """Start of a cut-off FRONT of a value that ends *text*, or -1."""
    n = len(text)
    for i in range(max(0, n - (max_len - 1)), n - MIN_LEN + 1):
        if text[i:i + MIN_LEN] in prefix8:
            tail = text[i:]
            if any(len(p) > len(tail) and p.startswith(tail) for p in pats):
                return i
    return -1


def _head_edge(text, pats, suffix8, max_len):
    """End of a cut-off BACK of a value that starts *text*, or -1."""
    n = len(text)
    for j in range(min(n, max_len - 1), MIN_LEN - 1, -1):
        if text[j - MIN_LEN:j] in suffix8:
            head = text[:j]
            if any(len(p) > len(head) and p.endswith(head) for p in pats):
                return j
    return -1


def _mask_spans(text, pats, prefix8, suffix8, max_len, head, tail, pre=()):
    spans = _spans(text, pats)
    spans.extend(pre)
    n = len(text)
    if tail:
        i = _tail_edge(text, pats, prefix8, max_len)
        if i >= 0:
            spans.append((i, n))
    if head:
        j = _head_edge(text, pats, suffix8, max_len)
        if j >= 0:
            spans.append((0, j))
    return _merge(spans)


def _mask(text, pats, marker, prefix8, suffix8, max_len, head, tail, pre=()):
    spans = _mask_spans(text, pats, prefix8, suffix8, max_len, head, tail, pre)
    return _replace(text, spans, marker) if spans else text


def mask_known(text, edges=True):
    """*text* with every known value replaced; unchanged when none is known.

    With *edges* a cut-off front or back (8+ characters) of a value at the
    ends of the text is masked as well.
    """
    pats = _str_patterns
    if not pats or not isinstance(text, str) or len(text) < MIN_LEN:
        return text
    return _mask(text, pats, MARKER, _prefix8, _suffix8, _max_len, edges, edges)


_frag_cache = ((), frozenset())    # (patterns the windows came from, windows)


def _windows():
    global _frag_cache
    pats = _str_patterns
    cached_pats, windows = _frag_cache
    if cached_pats is pats:
        return windows
    windows = frozenset(
        pat[i:i + MIN_LEN] for pat in pats for i in range(len(pat) - MIN_LEN + 1))
    _frag_cache = (pats, windows)
    return windows


def mask_fragments(text):
    """*text* with every run of 8 or more characters that lies inside a known
    value replaced.

    ``mask_known`` finds whole values (and a cut-off front or back at the ends
    of the text). A value split by something between its pieces, or cut off in
    the middle of a text, leaves pieces it cannot see. This looks at every
    8-character window of the text against an index of the 8-character windows
    of every registered form of every value, and replaces each run of
    overlapping or touching hits. It only adds masking; the persisted job
    streams use it on top of ``mask_known``.
    """
    if not isinstance(text, str) or len(text) < MIN_LEN or not _str_patterns:
        return text
    windows = _windows()
    spans = []
    start = end = -1
    for i in range(len(text) - MIN_LEN + 1):
        if text[i:i + MIN_LEN] in windows:
            if i <= end:
                end = i + MIN_LEN
            else:
                if end >= 0:
                    spans.append((start, end))
                start, end = i, i + MIN_LEN
    if end >= 0:
        spans.append((start, end))
    return _replace(text, spans, MARKER) if spans else text


_check_cache = ((), frozenset())   # (patterns the windows came from, windows)


def _check_windows():
    global _check_cache
    pats = _str_patterns
    cached_pats, windows = _check_cache
    if cached_pats is pats:
        return windows
    # Built differently from _windows(): the checker must not share its bugs.
    windows = frozenset(
        "".join(chars) for pat in pats
        for chars in zip(*[pat[k:] for k in range(MIN_LEN)]))
    _check_cache = (pats, windows)
    return windows


def find_fragment(text):
    """Index of the first 8-character window of *text* that lies inside a known
    value (any registered form), or -1.

    The output check for what the job streams persist. It shares no code with
    ``mask_fragments`` and says nothing about what the text looks like.
    """
    if not isinstance(text, str) or len(text) < MIN_LEN or not _str_patterns:
        return -1
    windows = _check_windows()
    for i in range(len(text) - MIN_LEN + 1):
        if text[i:i + MIN_LEN] in windows:
            return i
    return -1


def mask_known_deep(value, _depth=0):
    """A copy of *value* with every string VALUE at any depth masked.

    Dict keys are schema and are never touched. The structure is kept (a deep
    success payload is not reshaped); only past a depth no real envelope
    reaches is a container replaced by its masked ``str()``.
    """
    if not _str_patterns:
        return value
    if isinstance(value, str):
        return mask_known(value)
    if isinstance(value, (bytes, bytearray)):
        return mask_bytes(bytes(value))
    if isinstance(value, (dict, list, tuple)):
        if _depth > _DEEP:
            try:
                return mask_known(str(value))
            except RecursionError:
                return MARKER
        if isinstance(value, dict):
            return {k: mask_known_deep(v, _depth + 1) for k, v in value.items()}
        items = [mask_known_deep(v, _depth + 1) for v in value]
        return tuple(items) if isinstance(value, tuple) else items
    return value


def mask_bytes(data, edges=False):
    pats = _byte_patterns
    if not pats or len(data) < MIN_LEN:
        return data
    return _mask(data, pats, _BMARKER, _bprefix8, _bsuffix8, _max_blen, edges, edges)


def register_decoded_spans(data):
    """Register how the decoder spelled each known value in the bytes *data*.

    run_cli decodes the child's output with ``errors="replace"``. A known
    value that is not valid UTF-8 comes out as U+FFFD for its bad bytes, and
    which characters those are depends on the bytes beside it (a truncated
    sequence at its end is completed by the next byte). No fixed variant can
    cover that.

    So every OCCURRENCE of every registered byte form in *data* (plus a
    cut-off front or back at the ends) is looked at on its own, never merged
    with its neighbours: touching or overlapping occurrences would otherwise
    form one span that is valid UTF-8, or longer than any value can be, and
    would register nothing. An occurrence that is valid UTF-8 decodes to
    itself and is skipped. For the others, one incremental ``replace`` decoder
    pass over *data* gives the characters emitted while consuming each
    occurrence's bytes (a truncated sequence that the next byte completes is
    emitted after the occurrence and is not part of it), and the distinct
    pieces with 8 or more characters that are not U+FFFD are registered, after
    U+FFFD at either end (from a neighbouring or cut-off byte) is stripped; a
    piece still longer than MAX_LEN is registered as overlapping chunks. The
    work is linear in *data*.

    The failure and detail text built from the decoded output is masked by
    the usual text pass and so holds no fragment of such a value; the decoded
    text itself is not changed. Pieces apply process-wide, with the semantics
    of any registered value (``mask_fragments`` included). Does nothing unless
    a registered value is not valid UTF-8. Never raises.
    """
    try:
        pats = _byte_patterns
        if (not _undecodable or not isinstance(data, (bytes, bytearray))
                or len(data) < MIN_LEN):
            return
        data = bytes(data)
        occurrences = []
        _spans(data, pats, occurrences)
        tail = _tail_edge(data, pats, _bprefix8, _max_blen)
        if tail >= 0:
            occurrences.append((tail, len(data)))
        head = _head_edge(data, pats, _bsuffix8, _max_blen)
        if head >= 0:
            occurrences.append((0, head))
        spans = {(a, b) for a, b in occurrences if not _is_utf8(data[a:b])}
        if not spans:
            return
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        emitted = {0: 0}
        chunks, count, prev = [], 0, 0
        for cut in sorted({x for span in spans for x in span}):
            if cut > prev:
                chunk = decoder.decode(data[prev:cut])
                chunks.append(chunk)
                count += len(chunk)
                prev = cut
            emitted[cut] = count
        text = "".join(chunks)
        found = set()
        step = MAX_LEN - (MIN_LEN - 1)
        for a, b in spans:
            # U+FFFD at either end comes from the bytes beside the occurrence
            # or from its own cut-off end, never from a clean run of the value.
            piece = text[emitted[a]:emitted[b]].strip("\ufffd")
            # A piece longer than one value can be is registered as chunks that
            # overlap by MIN_LEN - 1 characters, so every MIN_LEN-character
            # window of it lies whole inside one chunk.
            start = 0
            while True:
                chunk = piece[start:start + MAX_LEN]
                if sum(c != "\ufffd" for c in chunk) >= MIN_LEN:
                    found.add(chunk)
                if start + MAX_LEN >= len(piece):
                    break
                start += step
        register_many(sorted(found), authoritative=True)
    except Exception:
        return


def _held_start(buf):
    """Index where the first tail starts that could still grow into a value."""
    n = len(buf)
    view = memoryview(buf)
    by_first = _byte_by_first
    for i in range(max(0, n - _max_blen + 1), n):
        cands = by_first.get(buf[i:i + 1])
        if cands:
            rest = n - i
            tail = view[i:]
            for pat in cands:
                if len(pat) > rest and pat.startswith(tail):
                    return i
    return n


def _back_candidate(buf, pats):
    """True when *buf* could still grow into a cut-off BACK of a value: it
    lies at the start of some pattern's tail that is longer than *buf* and at
    least ``MIN_LEN`` long. (*pats* are longest first.)"""
    n = len(buf)
    need = max(MIN_LEN, n + 1)
    for pat in pats:
        if len(pat) < need:
            return False
        if pat.find(buf, 0, len(pat) - need + n) >= 0:
            return True
    return False


class StreamMasker:
    """Mask a byte stream chunk by chunk without leaking a value split across
    two chunks: the tail that could still become (part of) a value is held
    back until the next chunk or ``flush()``, and nothing is replaced while a
    longer value may still be pending.

    Where the front of a value may never have been seen (the start of the
    stream, and the first buffer after the set of known values changed, since
    a value registered by another child can begin before this stream's
    window), a cut-off back of 8 or more bytes at the start of the buffer is
    masked as ``mask_known`` does. The buffer is held back while it is, so far,
    the start-aligned beginning of the 8+ byte tail of some registered form
    (so ordinary output that happens to match such a beginning, say "hello"
    against a value that ends in "hello00000000", waits for the next chunk or
    ``flush()``); the hold is bounded by the longest value."""

    __slots__ = ("_carry", "_covered", "_anchored", "_pats")

    def __init__(self):
        self._carry = b""
        self._covered = 0   # leading carry bytes already inside a longer match
        self._anchored = True   # the start of the carry may be a cut-off back
        self._pats = None   # the patterns the last call judged against

    def _check_anchor(self, pats):
        if pats is not self._pats:
            self._pats = pats
            self._anchored = True
        return self._anchored

    def feed(self, data):
        pats = _byte_patterns
        anchored = self._check_anchor(pats)
        if not pats:
            out, self._carry, self._covered = self._carry + data, b"", 0
            return out
        buf = self._carry + data
        if anchored and buf and _back_candidate(buf, pats):
            self._carry = buf       # nothing leaves while it may be a back
            return b""
        cut = _held_start(buf)
        raw = []
        spans = _spans(buf, pats, raw)
        pseudo_end = min(self._covered, len(buf))
        if anchored:
            back = _head_edge(buf, pats, _bsuffix8, _max_blen)
            pseudo_end = max(pseudo_end, back)
        if pseudo_end:
            spans.append((0, pseudo_end))
        spans = _merge(spans)
        # Coverage that was owed to the carry and is not emitted this time
        # (the cut is still at its start) stays owed.
        covered = max(0, pseudo_end - cut)
        partial = None
        for start, end in spans:
            if start < cut < end:
                # A run of back-to-back (or nested) matches crosses the cut.
                # Everything before the last occurrence that starts ahead of
                # the cut is inside complete matches, so it is replaced now
                # and only that occurrence is carried (the carry stays below
                # twice the longest pattern however long the run is). What a
                # longer match that began earlier still covers beyond the
                # cut is remembered, so it is masked on the next call.
                last = max((a for a, _ in raw if start <= a < cut), default=start)
                if last > start:
                    partial = (start, last)
                cut = last
                reach = max([e for a, e in raw if a < last] + [pseudo_end])
                covered = max(0, reach - last)
                break
        self._carry = buf[cut:]
        self._covered = covered
        self._anchored = anchored and cut == 0
        head = [s for s in spans if s[1] <= cut]
        if partial is not None:
            head.append(partial)
        return _replace(buf[:cut], head, _BMARKER)

    def flush(self):
        anchored = self._check_anchor(_byte_patterns)
        rest, self._carry = self._carry, b""
        covered, self._covered = self._covered, 0
        if not rest or not _byte_patterns:
            return rest
        pre = [(0, min(covered, len(rest)))] if covered else ()
        return _mask(rest, _byte_patterns, _BMARKER, _bprefix8, _bsuffix8,
                     _max_blen, anchored, True, pre)


# --- discovery ----------------------------------------------------------

_KNOWN_ENV_RE = re.compile(
    r"^(?:ANTHROPIC_API_KEY|ANTHROPIC_AUTH_TOKEN|CLAUDE_CODE_OAUTH_TOKEN"
    r"|CODEX_API_KEY|OPENAI_API_KEY|XAI_API_KEY|GROK_API_KEY|CURSOR_API_KEY)"
    r"(?:_[A-Z0-9_]+)?$")
_NAME_RE = re.compile(
    r"(?:^|_)(?:API_?KEY|ACCESS_?KEY|SECRET_?KEY|PRIVATE_?KEY|CLIENT_SECRET"
    r"|SECRET|TOKEN|PASSWORD|PASSWD|PASSPHRASE)(?:_([A-Z0-9]+))?$", re.IGNORECASE)
_NOT_A_VALUE_SUFFIX = frozenset((
    "URL", "URI", "BASE", "HOME", "ID", "FILE", "PATH", "DIR", "COMMAND", "CMD",
    "ENDPOINT", "HOST", "PORT", "SOCK", "SOCKET", "NAME", "TYPE", "MODE", "ENABLED",
    "ISSUER", "SCOPE", "SCOPES", "AUDIENCE", "PROVIDER", "REF",
))


def is_credential_name(name):
    """True for a variable name that holds a credential by convention."""
    if not isinstance(name, str):
        return False
    match = _NAME_RE.search(name)
    if not match:
        return False
    suffix = match.group(1)
    return suffix is None or suffix.upper() not in _NOT_A_VALUE_SUFFIX


def register_env(env, authoritative_only=False):
    """Register credential values found in an environment mapping.

    Names omni itself manages (the vendor key variables and their per-profile
    siblings) are authoritative; other credential-looking names are heuristic.
    """
    if not env:
        return
    try:
        items = list(env.items())
    except Exception:
        return
    sure, guessed = [], []
    for name, value in items:
        if not isinstance(name, str) or not isinstance(value, str):
            continue
        if _KNOWN_ENV_RE.match(name.upper()):
            sure.append(value)
        elif not authoritative_only and is_credential_name(name):
            guessed.append(value)
    register_many(sure, authoritative=True)
    register_many(guessed, authoritative=False)


def register_overrides(overrides):
    """Register the credential values of a profile override mapping."""
    if not overrides:
        return
    register_many([value for name, value in overrides.items()
                   if isinstance(name, str) and (
                       _KNOWN_ENV_RE.match(name.upper()) or is_credential_name(name))],
                  authoritative=True)


_JWT_SHAPE_RE = re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*\Z")
# A string that is not a JWT needs 32 characters. Below that the shape also
# matches command-line flags and model identifiers ("--some-long-flag",
# "vendor-model-4-5-20250929"), which would then be masked in later replies;
# the tokens of the vendor login files are longer (hex keys 32, prefixed keys
# and OAuth tokens 40 and up).
_TOKEN_SHAPE_RE = re.compile(r"[A-Za-z0-9._~+=-]{32,}\Z")
_FILE_MAX_BYTES = 65536


def _token_shaped(value):
    if not isinstance(value, str):
        return False
    if _JWT_SHAPE_RE.match(value):
        return True
    if not _TOKEN_SHAPE_RE.match(value):  # no "/" outside a JWT
        return False
    classes = (any(c.islower() for c in value), any(c.isupper() for c in value),
               any(c.isdigit() for c in value),
               any(not c.isalnum() for c in value))
    return sum(classes) >= 2


def _codex_tokens(data):
    tokens = data.get("tokens")
    yield data.get("OPENAI_API_KEY")
    if isinstance(tokens, dict):
        for name in ("access_token", "refresh_token", "id_token"):
            yield tokens.get(name)


def _claude_tokens(data):
    oauth = data.get("claudeAiOauth")
    if isinstance(oauth, dict):
        for name in ("accessToken", "refreshToken"):
            yield oauth.get(name)


def _entry_tokens(*names):
    """Top-level entries that are objects, reading the named fields of each."""
    def walk(data):
        for entry in data.values():
            if isinstance(entry, dict):
                for name in names:
                    yield entry.get(name)
    return walk


# (env var naming the home, default dir under HOME, file, exact token paths).
# grok and claude layouts were read from real files; the pi layout (one object
# per provider holding "access"/"refresh" for OAuth or "key" for an API key) is
# an unverified assumption: a wrong path only means those tokens go unmasked.
_AUTH_LAYOUT = (
    ("CODEX_HOME", ".codex", "auth.json", _codex_tokens),
    ("GROK_HOME", ".grok", "auth.json", _entry_tokens("key", "refresh_token")),
    ("CLAUDE_CONFIG_DIR", ".claude", ".credentials.json", _claude_tokens),
    (None, os.path.join(".pi", "agent"), "auth.json",
     _entry_tokens("access", "refresh", "key")),
)


def _read_json_file(directory, name):
    """Parsed JSON of ``directory/name`` or None. No final symlink, regular
    file, owned by this user, bounded size. Any failure (including a
    pathological nesting) means unreadable, never an exception."""
    try:
        real_dir = os.path.realpath(directory)
        dir_fd = os.open(real_dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                         dir_fd=dir_fd)
        finally:
            os.close(dir_fd)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                return None
            key = os.path.join(real_dir, name)
            fingerprint = (info.st_ino, info.st_ctime_ns, info.st_size,
                           info.st_mtime_ns)
            if _file_seen.get(key) == fingerprint:
                return None
            raw = os.read(fd, _FILE_MAX_BYTES + 1)
        finally:
            os.close(fd)
        if len(raw) > _FILE_MAX_BYTES:
            return None
        data = json.loads(raw.decode("utf-8"))
        _file_seen[key] = fingerprint
        return data
    except Exception:
        return None


def register_auth_files(env=None):
    """Register the tokens in the login files of the account homes.

    *env* names the homes (CODEX_HOME and friends) the child will see; an
    unset variable means the default home under the env's HOME.
    """
    try:
        env = os.environ if env is None else env
        home = env.get("HOME")
        found = []
        for var, default, name, tokens in _AUTH_LAYOUT:
            base = env.get(var) if var else None
            if base:
                directory = base
            elif home:
                directory = os.path.join(home, default)
            else:
                continue
            data = _read_json_file(directory, name)
            if not isinstance(data, dict):
                continue
            picked = []
            for token in tokens(data):
                if _token_shaped(token):
                    picked.append(token)
                    if len(picked) >= _FILE_TOKEN_CAP:
                        break
            found.extend(picked)
        register_many(found, authoritative=False)
    except Exception:
        return


def register_child_env(env):
    """Everything a spawn with *env* hands to its child (call before output)."""
    try:
        effective = os.environ if env is None else env
        register_env(effective)
        register_auth_files(effective)
    except Exception:
        return


def prime():
    """Register what this process's own environment and default homes hold."""
    register_child_env(None)
