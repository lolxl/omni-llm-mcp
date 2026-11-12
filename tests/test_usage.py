#!/usr/bin/env python3
# no-opencc
"""Executable seam for issue #22 P1.1: normalized usage envelopes.

WHAT THIS PINS
The issue asks for usage normalized into stable fields — `input_tokens`,
`cached_input_tokens`, `cache_write_tokens`, `output_tokens`, `source`
(`provider|estimated|unavailable`) — with the raw provider usage preserved
separately, and with this acceptance line binding:

  "Usage envelopes distinguish measured values from unavailable values;
   no zero/safe-looking default represents unknown."

That last line is the whole point and it is what most of these tests attack. A
backend that CANNOT report a field must yield `None` for it, never `0`. Zero is a
measurement ("the cache was not read"); unknown is the absence of one. Collapsing
them makes every later token-saving claim unfalsifiable.

EVIDENCE BASE (why the fixtures look like this)
Every field name below was captured LIVE on this host on 2026-08-11, not read off
documentation:

  agy 1.1.12      `agy -p ... --output-format json`   -> .usage
                  input_tokens / output_tokens / thinking_tokens /
                  cache_read_tokens / total_tokens
                  NOTE: agy reports NO cache-write field at all.
  grok 1.0.0      `grok --output-format json`         -> .usage
                  input_tokens / cache_read_input_tokens /
                  cache_creation_input_tokens / output_tokens /
                  reasoning_tokens / total_tokens
  claude 2.1.227  `claude -p --output-format json`    -> .usage
                  input_tokens / cache_creation_input_tokens /
                  cache_read_input_tokens / output_tokens
  codex 0.147.0   `codex exec --json`                 -> turn.completed.usage
                  input_tokens / cached_input_tokens / cache_write_input_tokens /
                  output_tokens / reasoning_output_tokens

Fixtures live in `tests/fixtures/usage_*.json{,l}` and are those captures with ids
and costs scrubbed; the token numbers and every field NAME are verbatim.

KNOWN EVIDENCE GAP — read before trusting the openai path
There is no live capture for the HTTP `openai_compat` backend (no API key on this
host). Its mapping is pinned here against the documented OpenAI response shape
(`usage.prompt_tokens` / `usage.completion_tokens` /
`usage.prompt_tokens_details.cached_tokens`), and is therefore UNVERIFIED-LIVE.
It must be re-verified against a real response before anyone quotes an
openai-path number in a benchmark. The tests below say so where it matters.

ALL pins are intentionally RED at this commit: `providers/usage.py` does not exist.
No production code lives in this file.
"""
import json
import os
import unittest

from tests import offline_home

_HOME = None


def setUpModule():
    # Run directly, the fenced spawns below admit ~/.claude.json and the login
    # keychain by their real names, and a CLI session rewriting them flips a
    # test. Point protected_home() at a synthetic home for this module.
    global _HOME
    _HOME = offline_home.use_synthetic_home()


def tearDownModule():
    try:
        _HOME.assert_module_hermetic()
    finally:
        _HOME.close()


HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(HERE, "tests", "fixtures")

# The stable field set the issue names, plus the two extras the live captures
# proved are universally available enough to be worth normalizing.
NORMALIZED_TOKEN_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_tokens",
    "output_tokens",
    "reasoning_tokens",
    "total_tokens",
)


def _summable(backend, uncached):
    """A usage payload complete enough for the summed backends.

    `input_tokens` is total logical input, so a payload carrying only the
    uncached part yields None by design (see TestInputTokensIsTotalLogicalInput).
    Tests about WHICH object was recovered should not trip over that rule.
    """
    if backend == "agy":
        return {"input_tokens": uncached, "cache_read_tokens": 0}
    return {"input_tokens": uncached, "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0}


def load_json(name):
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(name):
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


class TestEnvelopeShape(unittest.TestCase):
    """The envelope contract itself, independent of any one backend."""

    def test_every_backend_returns_the_same_key_set(self):
        from providers.usage import normalize_usage
        envelopes = [
            normalize_usage("agy", load_json("usage_agy.json").get("usage")),
            normalize_usage("grok", load_json("usage_grok.json").get("usage")),
            normalize_usage("claude", load_json("usage_claude.json").get("usage")),
            normalize_usage("codex", load_jsonl("usage_codex.jsonl")[-1].get("usage")),
            normalize_usage("openai", {"prompt_tokens": 1, "completion_tokens": 2}),
            normalize_usage("pi", load_json("usage_pi.json").get("usage")),
            normalize_usage("cursor", None),
            normalize_usage("agy", None),
        ]
        key_sets = {frozenset(e) for e in envelopes}
        self.assertEqual(len(key_sets), 1,
                         "every backend must produce the SAME key set; got %r"
                         % ([sorted(e) for e in envelopes],))
        keys = key_sets.pop()
        for f in NORMALIZED_TOKEN_FIELDS:
            self.assertIn(f, keys)
        self.assertIn("source", keys)
        self.assertIn("raw", keys)

    def test_source_is_one_of_three_literals(self):
        from providers.usage import normalize_usage
        for raw, expected in (
            ({"input_tokens": 1, "output_tokens": 2}, "provider"),
            (None, "unavailable"),
            ({}, "unavailable"),
        ):
            env = normalize_usage("grok", raw)
            self.assertEqual(env["source"], expected, "raw=%r" % (raw,))
        self.assertIn("estimated", _source_literals(),
                      "the 'estimated' source must exist in the contract even if "
                      "no backend uses it yet")

    def test_raw_is_preserved_verbatim(self):
        from providers.usage import normalize_usage
        raw = load_json("usage_grok.json")["usage"]
        env = normalize_usage("grok", raw)
        self.assertEqual(env["raw"], raw)
        # Preserved, not aliased: mutating the envelope must not corrupt the source.
        env["raw"]["input_tokens"] = -1
        self.assertEqual(load_json("usage_grok.json")["usage"]["input_tokens"], 8305)

    def test_token_values_are_int_or_none_only(self):
        from providers.usage import normalize_usage
        for backend, raw in (
            ("agy", load_json("usage_agy.json")["usage"]),
            ("grok", load_json("usage_grok.json")["usage"]),
            ("claude", load_json("usage_claude.json")["usage"]),
            ("codex", load_jsonl("usage_codex.jsonl")[-1]["usage"]),
        ):
            env = normalize_usage(backend, raw)
            for f in NORMALIZED_TOKEN_FIELDS:
                v = env[f]
                self.assertTrue(v is None or (isinstance(v, int)
                                              and not isinstance(v, bool)),
                                "%s.%s must be int or None, got %r"
                                % (backend, f, v))


class TestInputTokensIsTotalLogicalInput(unittest.TestCase):
    """`input_tokens` means the SAME thing on every backend: total logical input.

    Pinned in the #22 P1 scoping ruling — "input_tokens = total logical input
    (cached+uncached); providers reporting uncached-only are summed at mapping
    time" — and raised against this branch by the parallel P1.2a session.

    Which backends report which was settled by live measurement on this host on
    2026-08-11, not by reading docs:

      grok    EXCLUDES cache. Its own arithmetic proves it:
              8305 + 4864 + 29 == 13198, the total_tokens it reported.
      agy     EXCLUDES cache. Two runs of the same ~65.5k-token prompt:
              cold gave input=65497 cache_read=0; warm gave input=45048
              cache_read=20451 (45048+20451 == 65499, the same prompt), and
              agy's own total_tokens is input+output in BOTH runs.
      claude  EXCLUDES cache. 9 uncached input against 13525 read + 12954
              written is only coherent as uncached-only.
      codex   INCLUDES cache. `cached_input_tokens` names a SUBSET of
              `input_tokens`, per OpenAI's usage accounting.
      openai  INCLUDES cache. `prompt_tokens_details.cached_tokens` likewise.

    Without this, a token comparison across seats — which is the entire point of
    P1.3's benchmark — silently compares different quantities.
    """

    def test_grok_input_is_summed_and_matches_its_own_total(self):
        from providers.usage import normalize_usage
        raw = load_json("usage_grok.json")["usage"]
        env = normalize_usage("grok", raw)
        self.assertEqual(env["input_tokens"], 8305 + 4864 + 0)
        # The provider's own total is the independent check on that sum.
        self.assertEqual(env["input_tokens"] + env["output_tokens"],
                         raw["total_tokens"])

    def test_claude_input_is_summed(self):
        from providers.usage import normalize_usage
        env = normalize_usage("claude", load_json("usage_claude.json")["usage"])
        self.assertEqual(env["input_tokens"], 9 + 13525 + 12954)

    def test_agy_input_is_summed(self):
        from providers.usage import normalize_usage
        # Live warm-cache shape: agy has no cache-write field, so the sum is
        # uncached + cache_read only.
        env = normalize_usage("agy", {"input_tokens": 45048,
                                      "cache_read_tokens": 20451,
                                      "output_tokens": 5, "total_tokens": 45053})
        self.assertEqual(env["input_tokens"], 45048 + 20451)
        self.assertEqual(env["cached_input_tokens"], 20451)

    def test_cache_inclusive_backends_are_not_double_counted(self):
        from providers.usage import normalize_usage
        codex = normalize_usage(
            "codex", load_jsonl("usage_codex.jsonl")[-1]["usage"])
        self.assertEqual(codex["input_tokens"], 15354,
                         "codex already includes cached input; summing would "
                         "double-count it")
        openai = normalize_usage("openai", {
            "prompt_tokens": 100, "completion_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 40}})
        self.assertEqual(openai["input_tokens"], 100)

    def test_cached_and_write_fields_keep_their_own_meaning(self):
        # The sum must not swallow the components: cache-read and cache-write
        # stay separately reported.
        from providers.usage import normalize_usage
        env = normalize_usage("grok", load_json("usage_grok.json")["usage"])
        self.assertEqual(env["cached_input_tokens"], 4864)
        self.assertEqual(env["cache_write_tokens"], 0)

    def test_missing_component_makes_the_sum_unknown_not_understated(self):
        # A summed backend whose cache fields are absent cannot report a total
        # logical input. Returning the uncached part alone would UNDERSTATE it
        # while looking like a measurement -- the same class of lie as
        # substituting 0 for unknown.
        from providers.usage import normalize_usage
        env = normalize_usage("grok", {"input_tokens": 100})
        self.assertIsNone(env["input_tokens"])
        self.assertIsNone(env["cached_input_tokens"])
        # ...but anything the backend DID report still comes through.
        env = normalize_usage("grok", {"input_tokens": 100, "output_tokens": 7})
        self.assertEqual(env["output_tokens"], 7)
        self.assertEqual(env["source"], "provider")

    def test_summed_zero_is_still_zero(self):
        from providers.usage import normalize_usage
        env = normalize_usage("grok", {"input_tokens": 0,
                                       "cache_read_input_tokens": 0,
                                       "cache_creation_input_tokens": 0})
        self.assertEqual(env["input_tokens"], 0)


class TestUnknownIsNeverZero(unittest.TestCase):
    """The acceptance line that makes the whole feature falsifiable."""

    def test_agy_cache_write_is_none_not_zero(self):
        # agy's live JSON has NO cache-write field. The honest answer is None.
        from providers.usage import normalize_usage
        raw = load_json("usage_agy.json")["usage"]
        self.assertNotIn("cache_write_tokens", raw)
        self.assertNotIn("cache_creation_input_tokens", raw)
        env = normalize_usage("agy", raw)
        self.assertIsNone(env["cache_write_tokens"],
                          "agy cannot report cache writes: unknown must be None, "
                          "not %r" % (env["cache_write_tokens"],))

    def test_reported_zero_stays_zero(self):
        # The mirror image, and the reason this cannot be done with `or 0`/`or None`:
        # agy really did report cache_read_tokens == 0. That is a MEASUREMENT.
        from providers.usage import normalize_usage
        raw = load_json("usage_agy.json")["usage"]
        self.assertEqual(raw["cache_read_tokens"], 0)
        env = normalize_usage("agy", raw)
        self.assertEqual(env["cached_input_tokens"], 0,
                         "a reported 0 must survive as 0, not become None")
        self.assertEqual(env["source"], "provider")

    def test_no_usage_at_all_yields_all_none(self):
        from providers.usage import normalize_usage
        env = normalize_usage("claude", None)
        for f in NORMALIZED_TOKEN_FIELDS:
            self.assertIsNone(env[f], "%s must be None when nothing was measured" % f)
        self.assertEqual(env["source"], "unavailable")
        self.assertIsNone(env["raw"])

    def test_partial_usage_keeps_measured_and_nulls_the_rest(self):
        # NOTE: grok is a summed backend, so a payload carrying only the uncached
        # part cannot yield input_tokens -- see
        # TestInputTokensIsTotalLogicalInput. output_tokens is unsummed and
        # passes straight through, which is what this test is about.
        from providers.usage import normalize_usage
        env = normalize_usage("grok", {"output_tokens": 100})
        self.assertEqual(env["output_tokens"], 100)
        self.assertIsNone(env["input_tokens"])
        self.assertIsNone(env["cached_input_tokens"])
        self.assertEqual(env["source"], "provider")

    def test_garbage_types_do_not_become_zero(self):
        from providers.usage import normalize_usage
        env = normalize_usage("grok", {"input_tokens": "lots", "output_tokens": None})
        self.assertIsNone(env["input_tokens"],
                          "a non-integer count is unknown, not 0")
        self.assertIsNone(env["output_tokens"])
        self.assertEqual(env["source"], "unavailable",
                         "nothing was actually measured, so the envelope must not "
                         "claim a provider measurement")

    def test_non_dict_usage_is_unavailable_not_a_crash(self):
        from providers.usage import normalize_usage
        for raw in ("notanobject", [1, 2], 7, True):
            env = normalize_usage("grok", raw)
            self.assertEqual(env["source"], "unavailable", "raw=%r" % (raw,))
            self.assertIsNone(env["input_tokens"])


class TestPerBackendMapping(unittest.TestCase):
    """Each mapping pinned against the LIVE capture, field by field."""

    def test_agy(self):
        from providers.usage import normalize_usage
        env = normalize_usage("agy", load_json("usage_agy.json")["usage"])
        self.assertEqual(env["input_tokens"], 17238 + 0)          # summed
        self.assertEqual(env["output_tokens"], 10)
        self.assertEqual(env["cached_input_tokens"], 0)      # cache_read_tokens
        self.assertIsNone(env["cache_write_tokens"])         # agy has no such field
        self.assertEqual(env["reasoning_tokens"], 0)         # thinking_tokens
        self.assertEqual(env["total_tokens"], 17248)
        self.assertEqual(env["source"], "provider")

    def test_grok(self):
        from providers.usage import normalize_usage
        env = normalize_usage("grok", load_json("usage_grok.json")["usage"])
        self.assertEqual(env["input_tokens"], 8305 + 4864 + 0)   # summed
        self.assertEqual(env["cached_input_tokens"], 4864)
        self.assertEqual(env["cache_write_tokens"], 0)
        self.assertEqual(env["output_tokens"], 29)
        self.assertEqual(env["reasoning_tokens"], 24)
        self.assertEqual(env["total_tokens"], 13198)
        self.assertEqual(env["source"], "provider")

    def test_claude(self):
        from providers.usage import normalize_usage
        env = normalize_usage("claude", load_json("usage_claude.json")["usage"])
        self.assertEqual(env["input_tokens"], 9 + 13525 + 12954)  # summed
        self.assertEqual(env["cached_input_tokens"], 13525)
        self.assertEqual(env["cache_write_tokens"], 12954)
        self.assertEqual(env["output_tokens"], 188)
        self.assertIsNone(env["reasoning_tokens"],
                          "the claude CLI reports no reasoning-token field")
        self.assertIsNone(env["total_tokens"],
                          "the claude CLI reports no total; do NOT synthesize one "
                          "by addition — that would silently invent a measurement")
        self.assertEqual(env["source"], "provider")

    def test_codex(self):
        from providers.usage import normalize_usage
        raw = load_jsonl("usage_codex.jsonl")[-1]["usage"]
        env = normalize_usage("codex", raw)
        self.assertEqual(env["input_tokens"], 15354)
        self.assertEqual(env["cached_input_tokens"], 9984)   # cached_input_tokens
        self.assertEqual(env["cache_write_tokens"], 0)       # cache_write_input_tokens
        self.assertEqual(env["output_tokens"], 5)
        self.assertEqual(env["reasoning_tokens"], 0)         # reasoning_output_tokens
        self.assertIsNone(env["total_tokens"])
        self.assertEqual(env["source"], "provider")

    def test_pi(self):
        from providers.usage import normalize_usage
        raw = load_json("usage_pi.json")["usage"]
        env = normalize_usage("pi", raw)
        self.assertEqual(env["input_tokens"], 490 + 128)  # uncached + cacheRead
        self.assertEqual(env["cached_input_tokens"], 128)
        self.assertEqual(env["cache_write_tokens"], 0)
        self.assertEqual(env["output_tokens"], 2)
        self.assertEqual(env["reasoning_tokens"], 27)
        self.assertEqual(env["total_tokens"], 620)
        self.assertEqual(env["source"], "provider")
        self.assertEqual(env["input_tokens"] + env["output_tokens"],
                         raw["totalTokens"])

    def test_openai_shape_unverified_live(self):
        # UNVERIFIED-LIVE (see module docstring). Documented OpenAI shape only.
        from providers.usage import normalize_usage
        env = normalize_usage("openai", {
            "prompt_tokens": 50, "completion_tokens": 12, "total_tokens": 62,
            "prompt_tokens_details": {"cached_tokens": 40},
            "completion_tokens_details": {"reasoning_tokens": 4},
        })
        self.assertEqual(env["input_tokens"], 50)
        self.assertEqual(env["output_tokens"], 12)
        self.assertEqual(env["cached_input_tokens"], 40)
        self.assertEqual(env["total_tokens"], 62)
        self.assertEqual(env["reasoning_tokens"], 4)
        self.assertIsNone(env["cache_write_tokens"],
                          "the OpenAI usage object has no cache-write field")

    def test_openai_cache_write_is_read_when_present(self):
        # From the sol review: the openai usage object CAN carry a cache-write
        # count under prompt_tokens_details, and hardcoding the field as
        # unreportable turned a measured 0 into null — the exact inversion of the
        # rule this feature exists to enforce. Still UNVERIFIED-LIVE: reading the
        # field is safe either way, because an absent field yields None.
        from providers.usage import normalize_usage
        env = normalize_usage("openai", {
            "prompt_tokens": 100, "completion_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 1920,
                                      "cache_write_tokens": 0},
        })
        self.assertEqual(env["cached_input_tokens"], 1920)
        self.assertEqual(env["cache_write_tokens"], 0,
                         "a reported 0 must stay 0, not become None")

    def test_openai_without_details_blocks_are_unknown_not_zero(self):
        from providers.usage import normalize_usage
        env = normalize_usage("openai", {"prompt_tokens": 50, "completion_tokens": 12})
        self.assertIsNone(env["cached_input_tokens"])
        self.assertIsNone(env["reasoning_tokens"])

    def test_unknown_backend_name_is_loud_not_silently_empty(self):
        from providers.usage import normalize_usage
        with self.assertRaises(ValueError):
            normalize_usage("nosuchbackend", {"input_tokens": 1})


class TestCliJsonExtraction(unittest.TestCase):
    """Reading usage back out of what each CLI actually prints.

    P1.1 has to change agy/grok/claude from their current plain output format to
    `--output-format json` to get usage at all (codex is free: `--json` events on
    stdout while the reply text still arrives via `-o`). The reply TEXT must come
    through unchanged, and a CLI that prints something unparseable must degrade to
    `source: unavailable` — never lose the reply.
    """

    def test_agy_text_and_usage(self):
        from providers.usage import extract_cli_result
        text, env = extract_cli_result("agy", json.dumps(load_json("usage_agy.json")))
        self.assertEqual(text, "ok")           # .response, stripped
        self.assertEqual(env["input_tokens"], 17238)

    def test_agy_stream_uses_last_cumulative_terminal_result_once(self):
        from providers.usage import extract_cli_result
        cid = "fixture-conversation"
        frames = [{"event": "init", "conversation_id": cid,
                   "init": {"model": "fixture-model"}}]
        for turns, response, raw_input, cached, output in (
                (3, "ACK_01", 10, 5, 1), (7, "FINAL", 20, 10, 2)):
            frames.append({"event": "step_update", "step_update": {
                "conversation_id": cid, "step_index": turns,
                "step_type": "progress", "state": "ACTIVE",
                "text_delta": ""}})
            frames.append({"event": "result", "result": {
                "conversation_id": cid, "num_turns": turns,
                "status": "SUCCESS", "response": response,
                "usage": {"input_tokens": raw_input,
                          "cache_read_tokens": cached,
                          "output_tokens": output,
                          "total_tokens": raw_input + cached + output}}})
        text, env = extract_cli_result(
            "agy", "\n".join(json.dumps(frame) for frame in frames))
        self.assertEqual(text, "FINAL")
        self.assertEqual(env["input_tokens"], 30)
        self.assertEqual(env["cached_input_tokens"], 10)
        self.assertEqual(env["output_tokens"], 2)

    def test_agy_single_terminal_stream_success(self):
        from providers.usage import extract_cli_result, inband_error
        cid = "fixture-conversation"
        frames = [{"event": "init", "conversation_id": cid,
                   "init": {"model": "fixture-model"}},
                  {"event": "result", "result": {
                      "conversation_id": cid, "num_turns": 7,
                      "status": "SUCCESS", "response": "single answer",
                      "usage": {"input_tokens": 4,
                                "cache_read_tokens": 1,
                                "output_tokens": 2,
                                "total_tokens": 7}}}]
        stdout = "\n".join(json.dumps(frame) for frame in frames)
        text, env = extract_cli_result("agy", stdout)
        self.assertEqual(text, "single answer")
        self.assertEqual(env["input_tokens"], 5)
        self.assertIsNone(inband_error("agy", stdout))

    def test_agy_success_result_followed_by_step_update_fails_closed(self):
        from providers.usage import extract_cli_result, inband_error
        cid = "fixture-conversation"
        frames = [{"event": "init", "conversation_id": cid,
                   "init": {"model": "fixture-model"}},
                  {"event": "result", "result": {
                      "conversation_id": cid, "num_turns": 1,
                      "status": "SUCCESS", "response": "stale success",
                      "usage": {"input_tokens": 4,
                                "cache_read_tokens": 1,
                                "output_tokens": 2}}},
                  {"event": "step_update", "step_update": {
                      "conversation_id": cid, "step_index": 2,
                      "step_type": "agent_response", "state": "ACTIVE",
                      "text_delta": "unfinished next turn"}}]
        stdout = "\n".join(json.dumps(frame) for frame in frames)
        self.assertEqual(extract_cli_result("agy", stdout)[0], "")
        self.assertEqual(inband_error("agy", stdout)[0], "runtime_error")

    def test_agy_stream_failure_hides_partial_and_reports_vendor_cause(self):
        from pathlib import Path
        from providers.usage import extract_cli_result, inband_error
        stream = Path(FIXTURES, "agy_stream_interrupted.jsonl").read_text(
            encoding="utf-8")
        text, usage = extract_cli_result("agy", stream)
        failure = inband_error("agy", stream)
        self.assertEqual(text, "")
        self.assertEqual(usage["source"], "unavailable")
        self.assertEqual(failure[0], "runtime_error")
        self.assertIn("The stream was interrupted", failure[1])
        self.assertNotIn("PARTIAL_OUTPUT_MUST_NOT_BE_PUBLISHED", text)
        self.assertNotIn("ACK_01", text)

    def test_agy_stream_progress_does_not_hide_final_failure(self):
        from providers.usage import extract_cli_result, inband_error
        cid = "fixture-conversation"
        frames = [{"event": "init", "conversation_id": cid,
                   "init": {"model": "fixture-model"}},
                  {"event": "step_update", "step_update": {
                      "conversation_id": cid, "step_index": 1,
                      "step_type": "progress", "state": "ERROR",
                      "text_delta": "nonterminal"}},
                  {"event": "result", "result": {
                      "conversation_id": cid, "num_turns": 1,
                      "status": "ERROR", "response": "PARTIAL",
                      "error": "The stream was interrupted.",
                      "usage": {"input_tokens": 10,
                                "cache_read_tokens": 2,
                                "output_tokens": 3}}}]
        stdout = "\n".join(json.dumps(frame) for frame in frames)
        self.assertEqual(inband_error("agy", stdout)[0], "runtime_error")
        self.assertEqual(extract_cli_result("agy", stdout)[0], "")

    def test_agy_malformed_or_incomplete_stream_fails_closed(self):
        from providers.usage import extract_cli_result, inband_error
        cid = "fixture-conversation"
        prefix = json.dumps({"event": "init", "conversation_id": cid,
                             "init": {"model": "fixture-model"}})
        malformed = prefix + "\n{"
        missing_terminal = prefix + "\n" + json.dumps({
            "event": "step_update", "step_update": {
                "conversation_id": cid, "step_index": 1,
                "step_type": "agent_response", "text_delta": "partial"}})
        for stdout in (malformed, missing_terminal):
            with self.subTest(stdout=stdout):
                self.assertEqual(extract_cli_result("agy", stdout)[0], "")
                self.assertEqual(inband_error("agy", stdout)[0], "runtime_error")

    def test_agy_recognized_init_prefix_truncations_fail_closed(self):
        from pathlib import Path
        from providers.usage import extract_cli_result, inband_error
        init_line = Path(FIXTURES, "agy_stream_interrupted.jsonl").read_text(
            encoding="utf-8").splitlines()[0]
        event_key = init_line.index('"event"')
        event_end = init_line.index('"init"', event_key) + len('"init"')
        # Once the complete known event value is present, every truncation is
        # recognized as protocol; the complete init alone still lacks a result.
        for end in range(event_end, len(init_line) + 1):
            with self.subTest(prefix_length=end):
                prefix = init_line[:end]
                self.assertEqual(extract_cli_result("agy", prefix)[0], "")
                failure = inband_error("agy", prefix)
                self.assertEqual(failure[0], "runtime_error")
        uncertain = init_line[:event_end - 1]
        self.assertEqual(extract_cli_result("agy", uncertain)[0], uncertain)
        self.assertIsNone(inband_error("agy", uncertain))

    def test_agy_stream_init_payload_must_be_an_object(self):
        from providers.usage import extract_cli_result, inband_error
        cid = "fixture-conversation"
        frames = [{"event": "init", "conversation_id": cid, "init": None},
                  {"event": "result", "result": {
                      "conversation_id": cid, "num_turns": 1,
                      "status": "SUCCESS", "response": "must not publish",
                      "usage": {"input_tokens": 4,
                                "cache_read_tokens": 1,
                                "output_tokens": 2}}}]
        stdout = "\n".join(json.dumps(frame) for frame in frames)
        self.assertEqual(extract_cli_result("agy", stdout)[0], "")
        failure = inband_error("agy", stdout)
        self.assertEqual(failure[0], "runtime_error")
        self.assertIn("init payload", failure[1])

    def test_agy_stream_init_must_be_the_first_frame(self):
        from providers.usage import extract_cli_result, inband_error
        cid = "fixture-conversation"
        frames = [{"event": "step_update", "step_update": {
                      "conversation_id": cid, "step_index": 1,
                      "step_type": "progress", "state": "ACTIVE",
                      "text_delta": ""}},
                  {"event": "init", "conversation_id": cid,
                   "init": {"model": "fixture-model"}},
                  {"event": "result", "result": {
                      "conversation_id": cid, "num_turns": 1,
                      "status": "SUCCESS", "response": "must not publish",
                      "usage": {"input_tokens": 4,
                                "cache_read_tokens": 1,
                                "output_tokens": 2}}}]
        stdout = "\n".join(json.dumps(frame) for frame in frames)
        self.assertEqual(extract_cli_result("agy", stdout)[0], "")
        failure = inband_error("agy", stdout)
        self.assertEqual(failure[0], "runtime_error")
        self.assertIn("init must be the first", failure[1])

    def test_agy_plain_text_fallback_is_unchanged(self):
        from providers.usage import extract_cli_result, inband_error
        for plain_text in ("plain answer", "{draft answer, not JSON"):
            text, env = extract_cli_result("agy", plain_text)
            self.assertEqual(text, plain_text)
            self.assertEqual(env["source"], "unavailable")
            self.assertIsNone(inband_error("agy", plain_text))
        banner_json = "update banner\n" + json.dumps(load_json("usage_agy.json"))
        text, _env = extract_cli_result("agy", banner_json)
        self.assertEqual(text, "ok")
        self.assertIsNone(inband_error("agy", banner_json))

    def test_grok_text_and_usage(self):
        from providers.usage import extract_cli_result
        text, env = extract_cli_result("grok", json.dumps(load_json("usage_grok.json")))
        self.assertEqual(text, "ok")           # .text
        self.assertEqual(env["output_tokens"], 29)

    def test_claude_text_and_usage(self):
        from providers.usage import extract_cli_result
        text, env = extract_cli_result("claude",
                                       json.dumps(load_json("usage_claude.json")))
        self.assertEqual(text, "ok")           # .result
        self.assertEqual(env["cache_write_tokens"], 12954)

    def test_codex_usage_from_event_stream(self):
        from providers.usage import extract_cli_result
        stdout = "".join(json.dumps(e) + "\n" for e in load_jsonl("usage_codex.jsonl"))
        text, env = extract_cli_result("codex", stdout)
        # codex's reply text comes from the `-o` file, not the event stream; the
        # extractor may surface the agent_message text but MUST return the usage.
        self.assertEqual(env["input_tokens"], 15354)
        self.assertEqual(env["cached_input_tokens"], 9984)

    def test_pi_text_and_usage_from_event_stream(self):
        from providers.usage import extract_cli_result
        from pathlib import Path
        raw = Path(FIXTURES, "pi_stream.jsonl").read_text(encoding="utf-8")
        text, env = extract_cli_result("pi", raw)
        self.assertEqual(text, "PONG")
        self.assertEqual(env["input_tokens"], 490 + 128)
        self.assertEqual(env["cached_input_tokens"], 128)
        self.assertEqual(env["output_tokens"], 2)
        self.assertEqual(env["reasoning_tokens"], 27)
        self.assertEqual(env["total_tokens"], 620)
        self.assertEqual(env["source"], "provider")

    def test_pi_all_zero_placeholder_is_unavailable(self):
        from providers.usage import extract_cli_result
        zero = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0,
                "totalTokens": 0}
        stdout = "\n".join((
            json.dumps({"type": "session"}),
            json.dumps({"type": "message_update", "usage": zero}),
            json.dumps({
                "type": "message_end",
                "message": {"role": "assistant",
                             "content": [{"type": "text", "text": "PONG"}]},
            }),
        ))
        text, env = extract_cli_result("pi", stdout)
        self.assertEqual(text, "PONG")
        self.assertEqual(env["source"], "unavailable")
        self.assertIsNone(env["input_tokens"])

    def test_unparseable_output_keeps_the_reply_and_reports_unavailable(self):
        from providers.usage import extract_cli_result
        text, env = extract_cli_result("grok", "plain old answer, not json")
        self.assertEqual(text, "plain old answer, not json",
                         "a CLI that did not emit JSON must not cost us the reply")
        self.assertEqual(env["source"], "unavailable")

    def test_empty_output_is_unavailable(self):
        from providers.usage import extract_cli_result
        text, env = extract_cli_result("claude", "   ")
        self.assertEqual(text, "")
        self.assertEqual(env["source"], "unavailable")

    def test_noise_line_before_json_does_not_become_the_reply(self):
        # Both review seats found this independently and it was reproduced before
        # being fixed. A CLI that prints an update banner before its JSON used to
        # fail the whole-document parse, so the ENTIRE protocol document — session
        # ids, cost fields and all — came back as the model's answer, and the
        # perfectly readable usage was thrown away with it.
        from providers.usage import extract_cli_result
        for backend, doc in (
            ("grok", {"text": "real answer", "usage": _summable("grok", 9),
                      "sessionId": "s"}),
            ("agy", {"response": "real answer", "usage": _summable("agy", 9),
                     "conversation_id": "c"}),
            ("claude", {"result": "real answer", "usage": _summable("claude", 9),
                        "session_id": "s"}),
        ):
            stdout = "Update available: v1.0.1\n" + json.dumps(doc)
            text, env = extract_cli_result(backend, stdout)
            self.assertEqual(text, "real answer",
                             "%s: the reply must be recovered from the JSON "
                             "document, not buried in it (%r)" % (backend, text))
            self.assertEqual(env["input_tokens"], 9, backend)
            self.assertEqual(env["source"], "provider", backend)

    def test_noise_before_pretty_printed_json_is_still_recovered(self):
        # Found by the foreman attacking the first fix round. The line scan only
        # recognised single-line objects -- but the grok CLI's real output is
        # PRETTY-PRINTED across many lines, which is exactly the shape a banner
        # prefix would wreck. Fixture-shaped, not invented.
        from providers.usage import extract_cli_result
        for backend, doc in (
            ("grok", {"text": "real answer", "usage": _summable("grok", 9)}),
            ("agy", {"response": "real answer", "usage": _summable("agy", 9)}),
            ("claude", {"result": "real answer", "usage": _summable("claude", 9)}),
        ):
            stdout = "Update available: v1.0.1\n" + json.dumps(doc, indent=2)
            text, env = extract_cli_result(backend, stdout)
            self.assertEqual(text, "real answer", backend)
            self.assertEqual(env["input_tokens"], 9, backend)

    def test_noise_after_json_is_recovered(self):
        from providers.usage import extract_cli_result
        stdout = json.dumps({"text": "ok", "usage": _summable("grok", 9)}) + "\ndone."
        text, env = extract_cli_result("grok", stdout)
        self.assertEqual(text, "ok")
        self.assertEqual(env["input_tokens"], 9)

    def test_result_envelope_wins_over_an_earlier_object(self):
        # The scan must not hand back a progress/banner object that happens to
        # carry the reply field before the real envelope arrives.
        from providers.usage import extract_cli_result
        stdout = (json.dumps({"text": "partial", "type": "progress"}) + "\n"
                  + json.dumps({"text": "final", "usage": _summable("grok", 9)}))
        text, env = extract_cli_result("grok", stdout)
        self.assertEqual(text, "final")
        self.assertEqual(env["input_tokens"], 9)

    def test_codex_noise_line_does_not_lose_usage_or_leak_the_stream(self):
        # The codex JSONL parse was all-or-nothing: one non-JSON line discarded
        # every valid event AND returned the raw stream as the reply, which
        # providers/codex.py then hands back as a model answer when -o is empty.
        from providers.usage import extract_cli_result
        stdout = ('WARN: update available\n'
                  '{"type":"turn.completed","usage":{"input_tokens":5}}')
        text, env = extract_cli_result("codex", stdout)
        self.assertEqual(env["input_tokens"], 5,
                         "a readable turn.completed must survive a bad neighbour")
        self.assertNotIn("turn.completed", text)
        self.assertNotIn("WARN", text)

    def test_truncated_envelope_is_not_a_reply(self):
        # From the sol review. A half-written envelope must not leak onto the wire
        # as an answer. It happened to come out empty before this pin, but only
        # because the scan picked up the nested `{"input_tokens": 9}` fragment --
        # an accident, not a contract. Anything that starts as JSON and yields no
        # envelope is protocol debris, and the caller gets a LOUD empty_response.
        from providers.usage import extract_cli_result
        text, env = extract_cli_result(
            "grok", '{"text":"real answer","usage":{"input_tokens":9}')
        self.assertEqual(text, "")
        self.assertEqual(env["source"], "unavailable")

    def test_unparseable_text_starting_with_a_brace_is_still_a_reply(self):
        # From the sol review. The debris rule keyed on the first character, so a
        # reply that merely BEGINS with "{" was thrown away entirely. Debris is
        # output that contains readable JSON yet no envelope — not output that
        # happens to start with a brace and parses nowhere.
        from providers.usage import extract_cli_result
        text, env = extract_cli_result("grok", "{draft answer, not JSON")
        self.assertEqual(text, "{draft answer, not JSON")
        self.assertEqual(env["source"], "unavailable")

    def test_progress_object_never_outranks_the_result_envelope(self):
        # From the sol review. Preferring "the last object carrying usage" was
        # unconditional, so a progress line that happened to carry a token count
        # beat the real answer. The discriminator now comes from the fixtures:
        # claude's real envelope is `type: "result"`, agy's and grok's carry no
        # `type` at all, and progress-shaped objects declare something else.
        from providers.usage import extract_cli_result
        stdout = (json.dumps({"text": "FINAL"}) + "\n"
                  + json.dumps({"type": "progress", "text": "STILL WORKING",
                                "usage": {"input_tokens": 1}}))
        text, _ = extract_cli_result("grok", stdout)
        self.assertEqual(text, "FINAL")
        # ...and in the other order, so this is not an artefact of position.
        text, _ = extract_cli_result("grok", "\n".join(reversed(stdout.split("\n"))))
        self.assertEqual(text, "FINAL")

    def test_progress_status_does_not_veto_a_healthy_agy_envelope(self):
        # From the sol review: scanning every object let a non-result object's
        # status field fail a run that actually succeeded.
        from providers.usage import inband_error
        stdout = (json.dumps({"type": "progress", "status": "RUNNING"}) + "\n"
                  + json.dumps({"status": "SUCCESS", "response": "ok",
                                "usage": {"input_tokens": 1}}))
        self.assertIsNone(inband_error("agy", stdout))

    def test_lone_progress_document_is_not_an_answer(self):
        # Raised by the round-4 review as an explicitly hypothetical gap and closed
        # anyway: the whole-document parse returned before the result-like check,
        # so a stdout consisting of ONE progress object was served as the reply.
        # Not reachable from the four committed fixtures -- grok's `json` mode
        # emits a single terminal object and increments live in `streaming-json` --
        # but the failure mode is protocol text sold as a model answer, so it is
        # cheaper to close than to reason about again later.
        from providers.usage import extract_cli_result
        text, env = extract_cli_result(
            "grok", json.dumps({"type": "progress", "text": "STILL WORKING"}))
        self.assertEqual(text, "")
        self.assertEqual(env["source"], "unavailable")

    def test_cursor_json_result_usage_unavailable(self):
        from providers.usage import extract_cli_result
        text, env = extract_cli_result(
            "cursor", json.dumps(load_json("usage_cursor.json")))
        self.assertEqual(text, "HI")
        self.assertEqual(env["source"], "unavailable")
        self.assertIsNone(env["input_tokens"])
        self.assertIsNone(env["output_tokens"])

    def test_claude_typed_result_document_still_parses(self):
        # Falsifier guard: claude's REAL envelope declares type "result".
        from providers.usage import extract_cli_result
        doc = dict(load_json("usage_claude.json"))
        self.assertEqual(doc["type"], "result")
        text, env = extract_cli_result("claude", json.dumps(doc))
        self.assertEqual(text, "ok")
        self.assertEqual(env["output_tokens"], 188)

    def test_array_wrapped_envelope_is_recovered(self):
        from providers.usage import extract_cli_result
        text, env = extract_cli_result(
            "grok",
            json.dumps([{"text": "real answer", "usage": _summable("grok", 9)}]))
        self.assertEqual(text, "real answer")
        self.assertEqual(env["input_tokens"], 9)

    def test_plain_text_is_still_returned_verbatim(self):
        # Falsifier guard for the debris rule: output that never looked like JSON
        # is a real reply and must survive.
        from providers.usage import extract_cli_result
        text, env = extract_cli_result("grok", "plain old answer, not json")
        self.assertEqual(text, "plain old answer, not json")
        self.assertEqual(env["source"], "unavailable")

    def test_inband_error_survives_a_trailing_progress_object(self):
        # From the sol review: picking one object and inspecting only that one let
        # a trailing progress line hide a committed error envelope.
        from providers.usage import extract_cli_result, inband_error
        stdout = (json.dumps({"type": "result", "subtype": "error_during_execution",
                              "is_error": True, "api_error_status": 429,
                              "result": "", "usage": {"input_tokens": 5}}) + "\n"
                  + json.dumps({"type": "progress", "result": "still running"}))
        self.assertEqual(inband_error("claude", stdout),
                         ("rate_limit", "error_during_execution"))
        text, _ = extract_cli_result("claude", stdout)
        self.assertNotEqual(text, "still running")

    def test_inband_error_found_even_without_usage_on_the_error_object(self):
        from providers.usage import inband_error
        stdout = (json.dumps({"type": "result", "is_error": True,
                              "api_error_status": 401, "result": ""}) + "\n"
                  + json.dumps({"type": "progress", "result": "still running",
                                "usage": {"input_tokens": 1}}))
        self.assertEqual(inband_error("claude", stdout)[0], "auth_error")

    def test_codex_truncated_json_is_not_a_reply(self):
        from providers.usage import extract_cli_result
        text, env = extract_cli_result("codex", '{"type":"turn.completed"')
        self.assertEqual(text, "",
                         "half a JSON event is protocol debris, never an answer")
        self.assertEqual(env["source"], "unavailable")

    def test_json_without_usage_still_yields_text(self):
        from providers.usage import extract_cli_result
        text, env = extract_cli_result("grok", json.dumps({"text": "hello"}))
        self.assertEqual(text, "hello")
        self.assertEqual(env["source"], "unavailable")


class TestEmptyReplyStaysLoud(unittest.TestCase):
    """Switching to JSON output must not turn an empty reply into a silent success.

    Before P1.1 the providers checked `if not out` against the raw stdout, which WAS
    the reply. Under `--output-format json` stdout is always a non-empty document
    even when the model returned nothing, so that check stops firing and an empty
    answer would come back as `{"ok": true, "text": ""}` — exactly the silent
    success the server's LOUD-failure contract forbids.
    """

    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)

    def _fake_cli(self, stdout):
        async def fake_run_cli(argv, **kwargs):
            return {"returncode": 0, "stdout": stdout, "stderr": "",
                    "timed_out": False, "watched": False}
        return fake_run_cli

    def test_agy_empty_reply_is_empty_response(self):
        import asyncio
        from unittest import mock
        import providers.agy as agy_mod
        payload = json.dumps({"response": "   ",
                              "usage": {"input_tokens": 5, "output_tokens": 0}})
        with mock.patch.object(agy_mod.vendor_bin, "which", return_value="/usr/bin/agy"), \
             mock.patch.object(agy_mod, "fence_agy_spawn",
                               side_effect=lambda argv, env: (argv, env, "/synthetic")), \
             mock.patch.object(agy_mod, "_rmtree"), \
             mock.patch.object(agy_mod, "run_cli", self._fake_cli(payload)):
            res = asyncio.run(agy_mod.run_agy_subprocess("hi", "m", 60, "default"))
        self.assertFalse(res["ok"], res)
        self.assertEqual(res["reason"], "empty_response")

    def test_grok_empty_reply_is_empty_response(self):
        import asyncio
        from unittest import mock
        import providers.grok as grok_mod
        payload = json.dumps({"text": "", "usage": {"input_tokens": 5}})
        with mock.patch.object(grok_mod, "shutil") as sh, \
             mock.patch.object(grok_mod, "run_cli", self._fake_cli(payload)), \
             mock.patch.object(grok_mod, "_consult_result",  # json fake, not a stream
                               side_effect=lambda out, sid: (*grok_mod.extract_cli_result("grok", out), None)):
            sh.which.return_value = "/usr/bin/grok"
            res = asyncio.run(grok_mod.run_grok_subprocess("hi", "grok-4.5", 60))
        self.assertFalse(res["ok"], res)
        self.assertEqual(res["reason"], "empty_response")

    def test_claude_empty_reply_is_empty_response(self):
        import asyncio
        from unittest import mock
        import providers.claude as claude_mod
        payload = json.dumps({"result": "", "usage": {"input_tokens": 5}})
        with mock.patch.object(claude_mod, "shutil") as sh, \
             mock.patch.object(claude_mod, "run_cli", self._fake_cli(payload)), \
             mock.patch.object(claude_mod, "env_depth", return_value=0):
            sh.which.return_value = "/usr/bin/claude"
            res = asyncio.run(claude_mod.run_claude_subprocess("hi", "opus", 60))
        self.assertFalse(res["ok"], res)
        self.assertEqual(res["reason"], "empty_response")

    def test_claude_inband_error_is_not_a_successful_reply(self):
        # Both seats: the JSON envelopes carry their own error state, and reading
        # only the reply field means a CLI that exits 0 while reporting a failure
        # is either dressed up as a successful answer or misfiled as
        # `empty_response`, hiding the real cause.
        import asyncio
        from unittest import mock
        import providers.claude as claude_mod
        payload = json.dumps({"type": "result",
                              "subtype": "error_during_execution",
                              "is_error": True, "api_error_status": 429,
                              "result": "", "usage": {"input_tokens": 5}})

        async def fake_run_cli(argv, **kwargs):
            return {"returncode": 0, "stdout": payload, "stderr": "",
                    "timed_out": False, "watched": False}

        with mock.patch.object(claude_mod, "shutil") as sh, \
             mock.patch.object(claude_mod, "run_cli", fake_run_cli), \
             mock.patch.object(claude_mod, "env_depth", return_value=0):
            sh.which.return_value = "/usr/bin/claude"
            res = asyncio.run(claude_mod.run_claude_subprocess("hi", "opus", 60))
        self.assertFalse(res["ok"], res)
        self.assertEqual(res["reason"], "rate_limit",
                         "api_error_status 429 must be typed, not swallowed as "
                         "empty_response: %r" % (res,))

    def test_agy_inband_failure_status_is_loud(self):
        import asyncio
        from unittest import mock
        import providers.agy as agy_mod
        payload = json.dumps({"status": "ERROR", "response": "",
                              "usage": {"input_tokens": 5}})

        async def fake_run_cli(argv, **kwargs):
            return {"returncode": 0, "stdout": payload, "stderr": "",
                    "timed_out": False, "watched": False}

        with mock.patch.object(agy_mod.vendor_bin, "which", return_value="/usr/bin/agy"), \
             mock.patch.object(agy_mod, "fence_agy_spawn",
                               side_effect=lambda argv, env: (argv, env, "/synthetic")), \
             mock.patch.object(agy_mod, "_rmtree"), \
             mock.patch.object(agy_mod, "run_cli", fake_run_cli):
            res = asyncio.run(agy_mod.run_agy_subprocess("hi", "m", 60, "default"))
        self.assertFalse(res["ok"], res)
        self.assertNotEqual(res["reason"], "empty_response",
                            "an explicit non-SUCCESS status must not be filed as "
                            "an empty reply: %r" % (res,))

    def test_agy_rc0_stream_interruption_never_returns_partial_text(self):
        import asyncio
        from pathlib import Path
        from unittest import mock
        import providers.agy as agy_mod
        stream = Path(FIXTURES, "agy_stream_interrupted.jsonl").read_text(
            encoding="utf-8")

        async def fake_run_cli(argv, **kwargs):
            return {"returncode": 0, "stdout": stream, "stderr": "",
                    "timed_out": False, "watched": False}

        with mock.patch.object(agy_mod.vendor_bin, "which",
                               return_value="/usr/bin/agy"), \
             mock.patch.object(agy_mod, "fence_agy_spawn",
                               side_effect=lambda argv, env: (argv, env, "/synthetic")), \
             mock.patch.object(agy_mod, "_rmtree"), \
             mock.patch.object(agy_mod, "run_cli", fake_run_cli):
            result = asyncio.run(
                agy_mod.run_agy_subprocess("hi", "m", 60, "default"))
        self.assertFalse(result["ok"], result)
        self.assertEqual(result["reason"], "runtime_error")
        self.assertIn("The stream was interrupted", result["detail"])
        self.assertNotIn("PARTIAL_OUTPUT_MUST_NOT_BE_PUBLISHED",
                         result.get("text", ""))

    def test_successful_inband_status_still_succeeds(self):
        # Falsifier guard: the in-band check must not reject healthy envelopes.
        import asyncio
        from unittest import mock
        import providers.agy as agy_mod
        payload = json.dumps(load_json("usage_agy.json"))

        async def fake_run_cli(argv, **kwargs):
            return {"returncode": 0, "stdout": payload, "stderr": "",
                    "timed_out": False, "watched": False}

        with mock.patch.object(agy_mod.vendor_bin, "which", return_value="/usr/bin/agy"), \
             mock.patch.object(agy_mod, "fence_agy_spawn",
                               side_effect=lambda argv, env: (argv, env, "/synthetic")), \
             mock.patch.object(agy_mod, "_rmtree"), \
             mock.patch.object(agy_mod, "run_cli", fake_run_cli):
            res = asyncio.run(agy_mod.run_agy_subprocess("hi", "m", 60, "default"))
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["text"], "ok")

    def test_codex_event_stream_never_becomes_the_reply(self):
        # codex's reply comes from the -o file. With --json its stdout is a JSONL
        # event stream, and falling back to raw stdout would hand the caller
        # protocol noise dressed as a model answer.
        from providers.usage import extract_cli_result
        stdout = "".join(json.dumps(e) + "\n" for e in load_jsonl("usage_codex.jsonl"))
        text, _ = extract_cli_result("codex", stdout)
        self.assertEqual(text, "ok",
                         "only the agent_message text may surface as the reply, "
                         "never the raw event stream")
        self.assertNotIn("turn.completed", text)


def _source_literals():
    from providers.usage import USAGE_SOURCES
    return tuple(USAGE_SOURCES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
