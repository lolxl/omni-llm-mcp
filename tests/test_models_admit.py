# no-opencc
"""``omni models admit`` reads one v2 record and calls admission.decide.

No vendor CLI, no network, no alias read, no writes.
"""
import contextlib
import datetime as _datetime
import io
import json
import os
import tempfile
import unittest
from unittest import mock

import omni_cli
from providers import admission, catalog_v2, cli_run, model_catalog
from providers.catalog_record import validate_strict
from providers.host_fence import _VENDOR_AUTH_ENV

# Seeds v2 records; tests/offline_home.py gives this module the real gate.
OMNI_TEST_REAL_ADMISSION = True


def _fresh_timestamp():
    """UTC stamp taken when the record is built, so a 48h stale window cannot lapse."""
    now = _datetime.datetime.now(_datetime.timezone.utc).replace(microsecond=0)
    return now.isoformat().replace("+00:00", "Z")


_KEYS = {
    "ok", "admitted", "provider", "profile", "model", "effort",
    "effort_source", "refusal", "warnings",
}


def _record(provider, rows):
    models = []
    for row in rows:
        item = {
            "id": row["id"],
            "name": row.get("name", row["id"]),
            "routable_id": "%s/%s" % (provider, row["id"]),
            "supported_efforts": list(row.get("efforts") or []),
        }
        models.append(item)
    value = {
        "provider": provider,
        "source": "live",
        "observed_at": _fresh_timestamp(),
        "availability": "provider_advertised",
        "models": models,
    }
    record, _warnings = validate_strict(provider, value)
    return record


class ModelsAdmitTests(unittest.TestCase):
    def setUp(self):
        self.state = tempfile.TemporaryDirectory(prefix="omni-models-admit-")
        self.addCleanup(self.state.cleanup)
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ["OMNI_LLM_STATE_DIR"] = self.state.name
        os.environ["OMNI_LLM_NOTIFY_QUIET"] = "1"
        for names in _VENDOR_AUTH_ENV.values():
            for name in names:
                os.environ.pop(name, None)
        for key in list(os.environ):
            if key.startswith("OMNI_LLM_CATALOG_ENDPOINT_"):
                os.environ.pop(key, None)

    def _save(self, provider="claude", profile="default"):
        catalog_v2.save(provider, profile, _record(provider, [{
            "id": "claude-opus",
            "efforts": ["low", "high"],
        }]))

    def _run(self, argv):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = omni_cli.main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def _json(self, argv):
        code, out, err = self._run(argv)
        self.assertEqual(err, "")
        payload = json.loads(out)
        self.assertEqual(set(payload), _KEYS)
        return code, payload

    def _decide(self, model, effort, provider="claude", profile="default"):
        record, reason = catalog_v2.load(provider, profile)
        self.assertIsNone(reason)
        self.assertIsNotNone(record)
        return admission.decide(
            {"provider": provider, "model": model, "effort": effort},
            record, now=model_catalog._datetime.datetime.now(
                model_catalog._datetime.timezone.utc))

    def test_admitted_model_and_effort(self):
        self._save()
        code, payload = self._json([
            "models", "admit", "--provider", "claude", "--model", "claude-opus",
            "--effort", "high", "--json",
        ])
        self.assertEqual(code, 0)
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["admitted"])
        self.assertEqual(payload["provider"], "claude")
        self.assertEqual(payload["profile"], "default")
        self.assertEqual(payload["model"], "claude-opus")
        self.assertEqual(payload["effort"], "high")
        self.assertEqual(payload["effort_source"], "request")
        self.assertIsNone(payload["refusal"])
        self.assertEqual(payload["warnings"], [])

        line_code, line, err = self._run([
            "models", "admit", "--provider", "claude", "--model", "claude-opus",
            "--effort", "high",
        ])
        self.assertEqual(err, "")
        self.assertEqual(line_code, 0)
        self.assertEqual(line.count("\n"), 1)
        self.assertIn("admitted=true", line)
        self.assertIn("model=claude-opus", line)

    def test_unlisted_model_and_unlisted_effort(self):
        self._save()
        code, payload = self._json([
            "models", "admit", "--provider", "claude", "--model", "not-listed",
            "--effort", "high", "--json",
        ])
        decision = self._decide("not-listed", "high")
        self.assertEqual(code, 1)
        self.assertFalse(payload["admitted"])
        self.assertEqual(payload["refusal"], decision["refusal"])
        self.assertEqual(payload["refusal"], "unsupported_model")

        code, payload = self._json([
            "models", "admit", "--provider", "claude", "--model", "claude-opus",
            "--effort", "max", "--profile", "default", "--json",
        ])
        decision = self._decide("claude-opus", "max")
        self.assertEqual(code, 1)
        self.assertFalse(payload["admitted"])
        self.assertEqual(payload["refusal"], decision["refusal"])
        self.assertEqual(payload["refusal"], "unsupported_effort")
        self.assertEqual(payload["effort"], "max")

    def test_missing_record_is_no_catalog(self):
        code, payload = self._json([
            "models", "admit", "--provider", "claude", "--model", "claude-opus",
            "--effort", "high", "--json",
        ])
        self.assertEqual(code, 1)
        self.assertFalse(payload["ok"])
        self.assertFalse(payload["admitted"])
        self.assertEqual(payload["refusal"], "no_catalog")
        self.assertEqual(payload["provider"], "claude")
        self.assertEqual(payload["profile"], "default")
        self.assertEqual(payload["warnings"], [])

    def test_alias_file_does_not_change_the_answer(self):
        self._save()
        argv = [
            "models", "admit", "--provider", "claude", "--model", "claude-opus",
            "--effort", "high", "--json",
        ]
        before_code, before = self._json(argv)
        alias_path = os.path.join(self.state.name, "aliases.json")
        with open(alias_path, "w", encoding="utf-8") as handle:
            handle.write("{not json\n")
        with mock.patch("providers.aliases._load_overrides",
                        side_effect=AssertionError("alias file was read")):
            after_code, after = self._json(argv)
        self.assertEqual(after_code, before_code)
        self.assertEqual(after, before)
        self.assertTrue(after["admitted"])

    def test_spawns_nothing(self):
        self._save()
        path, _lock = catalog_v2._paths("claude", "default")
        before = path.read_bytes()
        with mock.patch.object(model_catalog.subprocess, "Popen") as popen, \
                mock.patch.object(cli_run, "_spawn") as spawn, \
                mock.patch.object(cli_run, "_run_sync") as run_sync, \
                mock.patch.object(admission, "gate",
                                  side_effect=AssertionError("gate")):
            code, payload = self._json([
                "models", "admit", "--provider", "claude", "--model", "claude-opus",
                "--effort", "high", "--json",
            ])
        self.assertEqual(code, 0)
        self.assertTrue(payload["admitted"])
        popen.assert_not_called()
        spawn.assert_not_called()
        run_sync.assert_not_called()
        self.assertEqual(path.read_bytes(), before)

    def test_usage_error_exits_2(self):
        with self.assertRaises(SystemExit) as caught:
            omni_cli.main(["models", "admit", "--model", "claude-opus"])
        self.assertEqual(caught.exception.code, 2)
        with self.assertRaises(SystemExit) as caught:
            omni_cli.main([
                "models", "admit", "--provider", "pi", "--model", "claude-opus",
            ])
        self.assertEqual(caught.exception.code, 2)

    def test_invalid_profile_is_usage_error(self):
        code, out, err = self._run([
            "models", "admit", "--provider", "claude", "--model", "claude-opus",
            "--profile", "bad/name", "--json",
        ])
        self.assertEqual(err, "")
        self.assertEqual(code, 2)
        payload = json.loads(out)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["reason"], "usage_error")
        self.assertEqual(payload["exit_code"], 2)
        self.assertNotIn("runtime_error", out)

    def test_cursor_folded_id(self):
        catalog_v2.save("cursor", "default", _record("cursor", [{
            "id": "composer-high",
        }]))
        code, payload = self._json([
            "models", "admit", "--provider", "cursor", "--model", "composer",
            "--effort", "high", "--json",
        ])
        self.assertEqual(code, 0)
        self.assertTrue(payload["admitted"])
        self.assertEqual(payload["model"], "composer-high")
        self.assertEqual(payload["effort"], "high")
        self.assertEqual(payload["effort_source"], "request")
        self.assertIsNone(payload["refusal"])
        self.assertEqual(payload["warnings"], [])

    def test_omitted_effort_uses_row_default(self):
        catalog_v2.save("claude", "default", _record("claude", [{
            "id": "claude-opus",
            "efforts": ["low", "medium", "high"],
        }]))
        code, payload = self._json([
            "models", "admit", "--provider", "claude", "--model", "claude-opus",
            "--json",
        ])
        self.assertEqual(code, 0)
        self.assertTrue(payload["admitted"])
        self.assertEqual(payload["model"], "claude-opus")
        self.assertEqual(payload["effort"], "medium")
        self.assertEqual(payload["effort_source"], "omni_default")
        self.assertIsNone(payload["refusal"])
        self.assertEqual(payload["warnings"], [])

    def test_omitted_effort_refused_when_row_needs_one(self):
        catalog_v2.save("claude", "default", _record("claude", [{
            "id": "claude-opus",
            "efforts": ["low", "high"],
        }]))
        code, payload = self._json([
            "models", "admit", "--provider", "claude", "--model", "claude-opus",
            "--json",
        ])
        self.assertEqual(code, 1)
        self.assertFalse(payload["admitted"])
        self.assertEqual(payload["refusal"], "effort_required")
        self.assertIsNone(payload["effort"])
        self.assertEqual(payload["warnings"], [])

    def test_second_profile_uses_its_own_record(self):
        work_home = os.path.join(self.state.name, "grok-work")
        os.mkdir(work_home)
        os.environ["GROK_HOME_WORK"] = work_home
        catalog_v2.save("grok", "default", _record("grok", [{
            "id": "grok-4",
            "efforts": ["low", "high"],
        }]))
        catalog_v2.save("grok", "work", _record("grok", [{
            "id": "grok-3",
            "efforts": ["low", "high"],
        }]))
        code, payload = self._json([
            "models", "admit", "--provider", "grok", "--model", "grok-3",
            "--effort", "low", "--profile", "work", "--json",
        ])
        self.assertEqual(code, 0)
        self.assertTrue(payload["admitted"])
        self.assertEqual(payload["profile"], "work")
        self.assertEqual(payload["model"], "grok-3")
        self.assertEqual(payload["effort"], "low")

        code, payload = self._json([
            "models", "admit", "--provider", "grok", "--model", "grok-4",
            "--effort", "high", "--profile", "work", "--json",
        ])
        self.assertEqual(code, 1)
        self.assertEqual(payload["profile"], "work")
        self.assertEqual(payload["refusal"], "unsupported_model")

        code, payload = self._json([
            "models", "admit", "--provider", "grok", "--model", "grok-3",
            "--effort", "low", "--json",
        ])
        self.assertEqual(code, 1)
        self.assertEqual(payload["profile"], "default")
        self.assertEqual(payload["refusal"], "unsupported_model")

    def test_corrupt_record_is_catalog_unreadable(self):
        self._save()
        path, _lock = catalog_v2._paths("claude", "default")
        path.write_text("{not json\n", encoding="utf-8")
        code, payload = self._json([
            "models", "admit", "--provider", "claude", "--model", "claude-opus",
            "--effort", "high", "--json",
        ])
        self.assertEqual(code, 1)
        self.assertFalse(payload["ok"])
        self.assertFalse(payload["admitted"])
        self.assertEqual(payload["refusal"], "catalog_unreadable")
        self.assertEqual(payload["warnings"], [])


if __name__ == "__main__":
    unittest.main()
