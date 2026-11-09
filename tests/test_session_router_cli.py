"""Real subprocess contracts, isolated HOME/state and local Git fixtures."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent.parent


class RouterCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(os.path.realpath(self.tmp.name))
        self.repo = self.base / 'repo'
        self.repo.mkdir()
        self.home = self.base / 'home'
        self.home.mkdir()
        self.env = {'PATH': os.environ['PATH'], 'HOME': str(self.home),
                    'OMNI_LLM_STATE_DIR': str(self.base / 'state'),
                    'OMNI_SESSION_KEY': 'synthetic-a', 'OMNI_SESSION_INCARNATION': 'one',
                    'OMNI_LLM_NOTIFY_QUIET': '1'}
        subprocess.run(['git', 'init', '-q', str(self.repo)], check=True, env=self.env)

    def cli(self, *args, stdin=None, code=0):
        p = subprocess.run([sys.executable, str(HERE / 'omni_cli.py'), 'session', *args],
                           cwd=self.repo, env=self.env, input=stdin, text=True, capture_output=True)
        self.assertEqual(p.returncode, code, p.stdout + p.stderr)
        self.assertEqual(p.stderr, '')
        return json.loads(p.stdout)

    def test_real_attach_update_list_observe_leave(self):
        a = self.cli('attach', '--route', 'synthetic-a')
        self.assertIn('instructions', a)
        self.assertIn('update', a['instructions'])
        self.assertIn('omni session receive', a['instructions'])
        self.assertIn('handoff', a['instructions'])
        self.assertIn('read', a['instructions'])
        self.assertEqual(a['session']['presence_id'],
                         self.cli('attach', '--route', 'synthetic-a')['session']['presence_id'])
        b = self.cli('update', '--expected-revision', '0', '--task', 'synthetic task',
                     '--work-state', 'working')['session']
        self.assertEqual(b['task_revision'], 1)
        self.cli('observe', '--sequence', '1', '--runtime-state', 'busy')
        peer = self.cli('peers')['sessions'][0]
        self.assertEqual((peer['work_state'], peer['runtime_state']), ('working', 'busy'))
        self.cli('leave')
        self.assertEqual(self.cli('peers')['sessions'], [])
        self.assertEqual(self.cli('get', a['session']['presence_id'])['session']['work_state'], 'working')
        self.cli('observe', '--sequence', '2', code=1)

    def test_hook_ensures_registration_and_context_warning(self):
        event = {'event': 'checkpoint', 'sequence': 1, 'runtime_state': 'idle',
                 'route': 'synthetic-a'}
        a = self.cli('hook', stdin=json.dumps(event))
        self.assertIn('task_empty', a['warnings'])
        self.assertTrue(a['needs_task_update'])
        self.assertEqual(a['session']['runtime_state'], 'idle')
        self.assertEqual(len(self.cli('peers')['sessions']), 1)
        self.cli('hook', stdin=json.dumps(event), code=1)

    def test_hook_exit_rejects_missing_or_old_sequence_without_closing(self):
        result = self.cli('hook', stdin=json.dumps(
            {'event': 'checkpoint', 'sequence': 20, 'route': 'a', 'runtime_state': 'busy'}))
        presence = result['session']['presence_id']
        for event, reason in [({'event': 'exit', 'sequence': 19}, 'stale_observation'),
                              ({'event': 'exit', 'sequence': 20}, 'stale_observation'),
                              ({'event': 'exit'}, 'invalid_sequence')]:
            failed = self.cli('hook', stdin=json.dumps(event), code=1)
            self.assertEqual(failed['reason'], reason)
            row = self.cli('get', presence)['session']
            self.assertEqual((row['state'], row['observation_sequence']), ('attached', 20))
            self.assertEqual(len(self.cli('peers')['sessions']), 1)
        result = self.cli('hook', stdin=json.dumps({'event': 'exit', 'sequence': 21}))
        self.assertEqual(result['session']['state'], 'closed')
        self.assertEqual(result['session']['observation_sequence'], 21)
        self.assertEqual(result['session']['work_state'], 'unknown')
        self.assertEqual(self.cli('peers')['sessions'], [])

    def test_invalid_first_hook_leaves_no_registration(self):
        events = [
            ({'sequence': -1}, 'invalid_sequence'),
            ({'sequence': True}, 'invalid_sequence'),
            ({'sequence': 0, 'runtime_state': 'finished'}, 'invalid_runtime_state'),
            ({'sequence': 0, 'cwd': str(self.home)}, 'not_git_repository'),
            ({'sequence': 0, 'harness': ''}, 'invalid_metadata'),
            ({'sequence': 0, 'harness': None}, 'invalid_metadata'),
            ({'sequence': 0, 'route': None}, 'invalid_route'),
            ({'sequence': 0, 'adapter': 'unix', 'route': {'path': 'relative', 'nonce': 'a'}}, 'invalid_route'),
        ]
        for overrides, reason in events:
            with self.subTest(overrides=overrides):
                event = dict(event='start', route='synthetic-a')
                event.update(overrides)
                result = self.cli('hook', stdin=json.dumps(event), code=1)
                self.assertEqual(result['reason'], reason)
                self.assertEqual(self.cli('peers')['sessions'], [])
        self.env['OMNI_SESSION_INCARNATION'] = 'actual-incarnation'
        attached = self.cli('attach', '--route', 'synthetic-a')
        self.assertEqual(attached['session']['incarnation'], 'actual-incarnation')

    def test_invalid_existing_hook_preserves_entire_record(self):
        original = self.cli('attach', '--route', 'synthetic-a')['session']
        for event in [{'event': 'checkpoint', 'sequence': 1, 'role': ''},
                      {'event': 'checkpoint', 'sequence': 1, 'cwd': str(self.home)},
                      {'event': 'exit', 'sequence': 1, 'runtime_state': 'finished'},
                      {'event': 'checkpoint', 'sequence': 1, 'role': None},
                      {'event': 'checkpoint', 'sequence': 1, 'route': None}]:
            self.cli('hook', stdin=json.dumps(event), code=1)
            self.assertEqual(self.cli('get', original['presence_id'])['session'], original)

    def test_root_help_advertises_session_command(self):
        process = subprocess.run([sys.executable, str(HERE / 'omni_cli.py'), '--help'],
                                 cwd=self.repo, env=self.env, text=True, capture_output=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn('session', process.stdout)
        self.assertIn('session table', process.stdout)

    def test_hook_preserves_existing_route_and_flags_changed_context(self):
        self.cli('attach', '--route', 'synthetic-a')
        self.cli('update', '--expected-revision', '0', '--task', 'original context',
                 '--work-state', 'working')
        subprocess.run(['git', '-C', str(self.repo), 'symbolic-ref', 'HEAD',
                        'refs/heads/new-context'], check=True, env=self.env,
                       capture_output=True)
        event = {'event': 'checkpoint', 'sequence': 2, 'runtime_state': 'busy'}
        result = self.cli('hook', stdin=json.dumps(event))
        self.assertEqual(result['session']['route'], 'synthetic-a')
        self.assertTrue(result['needs_task_update'])
        self.assertIn('task_context_changed', result['warnings'])
        self.assertEqual(result['session']['task'], 'original context')

    def test_send_stdin_failure_never_reflects_body(self):
        a = self.cli('attach', '--route', 'synthetic-a')['session']
        marker = 'SYNTHETIC_PRIVATE_MESSAGE_17834'
        result = self.cli('send', a['presence_id'], stdin=marker, code=1)
        self.assertEqual(result['reason'], 'adapter_unavailable')
        self.assertNotIn(marker, json.dumps(result))
        for file in (self.base / 'state').rglob('*'):
            if file.is_file():
                self.assertNotIn(marker.encode(), file.read_bytes())
        self.cli('send', a['presence_id'], stdin='x' * 65537, code=1)

    def test_missing_binding_and_unknown_hook_fields_fail_closed(self):
        del self.env['OMNI_SESSION_KEY']
        self.assertEqual(self.cli('attach', '--route', 'a', code=1)['reason'], 'invalid_binding')
        self.env['OMNI_SESSION_KEY'] = 'synthetic-a'
        self.cli('hook', stdin=json.dumps({'event': 'checkpoint', 'sequence': 1,
                                         'route': 'a', 'task': 'forged'}), code=1)

    def test_concurrent_subprocess_attach_does_not_lose_rows(self):
        children = []
        for i in range(8):
            env = dict(self.env, OMNI_SESSION_KEY='process-%d' % i)
            children.append(subprocess.Popen(
                [sys.executable, str(HERE / 'omni_cli.py'), 'session', 'attach', '--route', 'route-%d' % i],
                cwd=self.repo, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE))
        self.addCleanup(lambda: [p.kill() for p in children if p.poll() is None])
        ids = []
        for process in children:
            out, err = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 0, out + err)
            self.assertEqual(err, '')
            ids.append(json.loads(out)['session']['presence_id'])
        self.assertEqual(len(set(ids)), 8)
        self.assertEqual(len(self.cli('peers')['sessions']), 8)

    def test_real_unix_receive_twice_after_manual_attach(self):
        # Requires host local socket permission; deliberately uses no fake transport.
        recipient = self.cli('attach', '--route', 'synthetic-a')['session']
        recipient_id = recipient['presence_id']
        self.env['OMNI_SESSION_KEY'] = 'synthetic-b'
        sender = self.cli('attach', '--route', 'synthetic-b')['session']
        self.env['OMNI_SESSION_KEY'] = 'synthetic-a'
        previous = None
        for cycle in range(2):
            receiver_env = dict(self.env)
            process = subprocess.Popen(
                [sys.executable, str(HERE / 'omni_cli.py'), 'session', 'receive', '--timeout', '5'],
                cwd=self.repo, env=receiver_env, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                deadline = time.monotonic() + 4
                route = None
                while time.monotonic() < deadline:
                    record = self.cli('get', recipient_id)['session']
                    if (record['adapter'] == 'unix' and record['route'] != previous
                            and isinstance(record['route'], dict)):
                        route = record['route']
                        break
                    if process.poll() is not None:
                        out, err = process.communicate()
                        self.fail('receiver exited before readiness: ' + out + err)
                    time.sleep(.02)
                self.assertIsNotNone(route, 'receiver never registered endpoint')
                if previous is not None:
                    self.assertNotEqual(route['path'], previous['path'])
                    self.assertNotEqual(route['nonce'], previous['nonce'])
                marker = 'SYNTHETIC_PRIVATE_UNIX_%d' % cycle
                body = marker + ': 雪\nline two\n"quoted"'
                self.env['OMNI_SESSION_KEY'] = 'synthetic-b'
                sent = self.cli('send', recipient_id, stdin=body)
                self.env['OMNI_SESSION_KEY'] = 'synthetic-a'
                self.assertEqual(sent, {'ok': True, 'status': 'handoff'})
                out, err = process.communicate(timeout=6)
                self.assertEqual(process.returncode, 0, out + err)
                self.assertEqual(err, '')
                received = json.loads(out)
                self.assertEqual(received['body'], body)
                self.assertEqual(received['sender_id'], sender['presence_id'])
                self.assertEqual(received['presence_id'], recipient_id)
                self.assertNotIn(marker, json.dumps(sent, ensure_ascii=False))
                for file in (self.base / 'state').rglob('*'):
                    if file.is_file():
                        self.assertNotIn(marker.encode(), file.read_bytes())
                self.assertFalse(Path(route['path']).exists())
                # Project JSON and environment can retain the original attach route.
                if cycle == 0:
                    checkpoint = {'event': 'checkpoint', 'sequence': 1, 'route': 'synthetic-a'}
                else:
                    self.env['OMNI_SESSION_ROUTE'] = 'synthetic-a'
                    checkpoint = {'event': 'checkpoint', 'sequence': 2}
                observed = self.cli('hook', stdin=json.dumps(checkpoint))['session']
                self.assertEqual((observed['adapter'], observed['route']), ('unix', route))
                self.assertEqual(observed['presence_id'], recipient_id)
                previous = route
            finally:
                self.env['OMNI_SESSION_KEY'] = 'synthetic-a'
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.communicate(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.communicate(timeout=2)

        closed = self.cli('hook', stdin=json.dumps({'event': 'exit', 'sequence': 3}))['session']
        self.assertEqual((closed['state'], closed['observation_sequence']), ('closed', 3))
        self.assertEqual(closed['route'], previous)

    def test_documentation_examples_execute(self):
        # Only fenced sh lines prefixed with this marker are runnable examples.
        doc = (HERE / 'docs' / 'session-router.md').read_text()
        import shlex
        lines = [line[len('$ omni session '):] for line in doc.splitlines()
                 if line.startswith('$ omni session ')]
        self.assertTrue(lines)
        for line in lines:
            self.cli(*shlex.split(line))


if __name__ == '__main__':
    unittest.main()
