"""P1.3: pin tool-description economy and schema structure (issue #22 P1).

Hard limits from wo-p1.3.md:
- compact TOOLS serialization strictly under the P1.2a reference (9625) and
  within the declared budget (≤ 9500)
- tool names/order unchanged
- inputSchema/outputSchema structure byte-equivalent once description fields
  are stripped (hash computed from base tree 0a2150b before edits)
- required call-time safety substrings present; bulk static prose absent
"""
import hashlib
import json
import unittest

from providers.dispatch import TOOLS


# Base tree (0a2150b) compact bytes before P1.3 edits: 10643
P12A_REFERENCE_BYTES = 9625
DECLARED_BUDGET_BYTES = 9500

# SHA-256 of json.dumps([t["name"] for t in TOOLS], separators=(',',':'))
# at base 0a2150b (and after P1.3 name/order-preserving edit).
TOOL_NAMES_SHA256 = (
    "fdffa92e8cba2919c05a1d794426f213329ebbe73655118ceabcedf35b451580"
)

# SHA-256 of compact json.dumps of TOOLS with every "description" field
# stripped, computed on base tree 0a2150b BEFORE any P1.3 description edits.
# Amended for issue #22 P1.4: result_mode / artifact_ref are intentional schema
# additions (scoping comment on #22). Amended again for the optional `effort`
# input on llm_generate / llm_big_context. Amended for issue #72: `seat` on
# the text output schema and `require_read_wall` on llm_generate /
# llm_big_context. Any future structural change must update this hash in its
# own reviewed commit, never as a drive-by.
BASE_STRIPPED_PROJECTION_SHA256 = (
    "4763c4a35753c5811ca915265f640f464787e92c18514f8d0c41910129910241"
)

# Call-time safety markers that must remain somewhere in the four tools.
REQUIRED_SUBSTRINGS = (
    "PUBLIC",
    "fallback",
    "unsupported_files",
    "inlined as text",  # grok consult-lane text attachments must stay advertised
    "source_path",
    "OMNI_LLM_SOURCE_ROOTS",  # grok allowlist env name
    "codex can READ any local file",  # codex read-scope warning must never leave the schema (ASCII-safe: json.dumps escapes non-ASCII)
    "LOUD",
    "unsupported_effort",
    "pi/",  # pi lane must stay discoverable from tools/list
    "cursor/",  # cursor lane (Fable/Sol transport) must stay discoverable
)

# Distinctive static sentences moved to README — must be absent from TOOLS.
FORBIDDEN_BULK_MARKERS = (
    "Pipes your prompt into Google Antigravity (agy), OpenAI, Grok Build CLI, "
    "or local OpenAI-compatible LLMs.",
    "rides your SuperGrok/X subscription",
    "POST {XAI_API_BASE|https://api.x.ai/v1}/images/generations",
)


def _strip_descriptions(obj):
    if isinstance(obj, dict):
        return {
            k: _strip_descriptions(v)
            for k, v in obj.items()
            if k != "description"
        }
    if isinstance(obj, list):
        return [_strip_descriptions(x) for x in obj]
    return obj


def _compact_tools():
    return json.dumps(TOOLS, separators=(",", ":"))


class TestDescriptionEconomy(unittest.TestCase):
    def test_compact_bytes_under_gate_and_budget(self):
        n = len(_compact_tools())
        self.assertLess(
            n, P12A_REFERENCE_BYTES,
            "tools_compact_bytes must be strictly below P1.2a reference %d; got %d"
            % (P12A_REFERENCE_BYTES, n),
        )
        self.assertLessEqual(
            n, DECLARED_BUDGET_BYTES,
            "tools_compact_bytes must be within declared budget %d; got %d"
            % (DECLARED_BUDGET_BYTES, n),
        )

    def test_tool_name_list_sha256_unchanged(self):
        names = [t["name"] for t in TOOLS]
        digest = hashlib.sha256(
            json.dumps(names, separators=(",", ":")).encode()
        ).hexdigest()
        self.assertEqual(digest, TOOL_NAMES_SHA256)
        self.assertEqual(
            names,
            ["llm_generate", "llm_big_context", "list_models", "llm_image"],
        )

    def test_schema_structure_byte_equivalent_to_base(self):
        stripped = _strip_descriptions(TOOLS)
        ser = json.dumps(stripped, separators=(",", ":"))
        digest = hashlib.sha256(ser.encode()).hexdigest()
        self.assertEqual(
            digest, BASE_STRIPPED_PROJECTION_SHA256,
            "description-stripped TOOLS projection diverged from base pin",
        )

    def test_required_safety_substrings_present(self):
        blob = _compact_tools()
        for token in REQUIRED_SUBSTRINGS:
            with self.subTest(token=token):
                self.assertIn(token, blob)

    def test_forbidden_bulk_markers_absent(self):
        blob = _compact_tools()
        for marker in FORBIDDEN_BULK_MARKERS:
            with self.subTest(marker=marker[:48]):
                self.assertNotIn(marker, blob)


if __name__ == "__main__":
    unittest.main()
