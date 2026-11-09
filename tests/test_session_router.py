"""Behavioral Session Router contracts; temporary local Git only."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from providers.session_router import Registry, RouterError


class GuardedAdapter:
    def __init__(self):
        self.live = {}
        self.forwarded = []
        self.failure = None

    def guarded_forward(self, target, body, *, expected_incarnation):
        # Live binding and forwarding are one adapter operation, not a prior probe.
        if self.live.get(target['route']) != expected_incarnation:
            return {'ok': False, 'reason': 'target_replaced'}
        if self.failure:
            raise RuntimeError(body)
        self.forwarded.append((target['route'], expected_incarnation, body))
        return {'ok': True, 'status': 'handoff', 'text': body}


class RouterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(os.path.realpath(self.tmp.name))
        self.repo = self.base / 'repo'
        self.repo.mkdir()
        self.git(self.repo, 'init', '-q')
        self.git(self.repo, '-c', 'user.name=Synthetic', '-c', 'user.email=synthetic@example.invalid',
                 'commit', '-q', '--allow-empty', '-m', 'fixture')
        self.adapter = GuardedAdapter()
        self.state = self.base / 'state'
        self.router = Registry(self.state, adapters={'fake': self.adapter})

    def git(self, cwd, *args):
        return subprocess.run(['git', '-C', str(cwd), *args], check=True,
                              capture_output=True, text=True).stdout.strip()

    def attach(self, key='a', inc='one', cwd=None, **kwargs):
        return self.router.attach(session_key=key, incarnation=inc,
                                  cwd=cwd or self.repo, adapter='fake', route=key, **kwargs)

    def error(self, reason, fn, **kwargs):
        with self.assertRaises(RouterError) as caught:
            fn(**kwargs)
        self.assertEqual(caught.exception.reason, reason)

    def test_attach_reuses_id_and_closed_incarnation_never_resurrects(self):
        a = self.attach()
        self.assertEqual(a['presence_id'], self.attach()['presence_id'])
        self.router.leave(session_key='a', incarnation='one')
        self.error('session_closed', self.attach)
        b = self.attach(inc='two')
        self.assertNotEqual(a['presence_id'], b['presence_id'])
        self.error('session_closed', self.attach)
        self.assertEqual(len(self.router.peers(cwd=self.repo)), 1)

    def test_active_replacement_requires_explicit_leave(self):
        self.attach()
        self.error('incarnation_conflict', self.attach, inc='two')

    def test_route_cannot_be_changed_by_repeat_attach(self):
        self.attach()
        self.error('binding_conflict', self.router.attach, session_key='a', incarnation='one',
                   cwd=self.repo, adapter='fake', route='different')

    def test_worktrees_share_repo_but_nested_repo_is_separate(self):
        worktree = self.base / 'worktree'
        self.git(self.repo, 'worktree', 'add', '-q', '-b', 'fixture', str(worktree))
        nested = self.repo / 'nested'
        nested.mkdir()
        self.git(nested, 'init', '-q')
        self.attach()
        self.attach('b', cwd=worktree)
        self.attach('c', cwd=nested)
        self.assertEqual(len(self.router.peers(cwd=self.repo)), 2)
        self.assertEqual(len(self.router.peers(cwd=nested)), 1)

    def test_task_cas_and_observation_field_ownership(self):
        a = self.attach()
        a = self.router.update(session_key='a', incarnation='one', expected_revision=0,
                               task='implement router', work_state='working')
        self.assertEqual(a['task_revision'], 1)
        self.error('task_revision_conflict', self.router.update, session_key='a', incarnation='one',
                   expected_revision=0, task='old', work_state='done')
        b = self.router.observe(session_key='a', incarnation='one', sequence=2,
                                runtime_state='busy', cwd=self.repo)
        self.assertEqual((b['task'], b['work_state'], b['task_revision']),
                         ('implement router', 'working', 1))
        self.assertEqual(b['last_task_update'], a['last_task_update'])
        self.assertGreaterEqual(b['last_seen'], a['last_seen'])
        self.error('stale_observation', self.router.observe, session_key='a', incarnation='one',
                   sequence=1, runtime_state='idle', cwd=self.repo)
        for revision, state in enumerate(['blocked', 'working', 'done'], start=1):
            b = self.router.update(session_key='a', incarnation='one', expected_revision=revision,
                                   task='implement router', work_state=state)
        self.assertEqual(b['runtime_state'], 'busy')
        self.router.leave(session_key='a', incarnation='one')
        self.error('session_closed', self.router.observe, session_key='a', incarnation='one',
                   sequence=3, runtime_state='idle', cwd=self.repo)
        self.assertEqual(self.router.get(b['presence_id'])['work_state'], 'done')

    def test_leave_is_not_task_success_and_unauthorized_sender_fails(self):
        a = self.attach()
        self.error('caller_not_attached', self.router.update, session_key=a['presence_id'],
                   incarnation='one', expected_revision=0, task='forged', work_state='done')
        self.router.leave(session_key='a', incarnation='one')
        self.assertEqual(self.router.get(a['presence_id'])['work_state'], 'unknown')

    def test_concurrent_attach_preserves_rows_and_deduplicates(self):
        def attach(i):
            return self.attach(str(i % 8))['presence_id']
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            ids = list(pool.map(attach, range(40)))
        self.assertEqual(len(set(ids)), 8)
        self.assertEqual(len(self.router.peers(cwd=self.repo)), 8)

    def test_send_atomic_binding_and_no_body_in_state_or_results(self):
        a, b = self.attach(), self.attach('b')
        marker = 'SYNTHETIC_MESSAGE_DO_NOT_PERSIST_876'
        self.adapter.live['b'] = 'one'
        result = self.router.send(session_key='a', incarnation='one',
                                  target_id=b['presence_id'], body=marker)
        self.assertEqual(result, {'ok': True, 'status': 'handoff'})
        self.assertEqual(self.adapter.forwarded, [('b', 'one', marker)])
        self.assertNotIn(marker, json.dumps(result))
        for path in self.state.rglob('*'):
            if path.is_file():
                self.assertNotIn(marker.encode(), path.read_bytes())
        self.adapter.live['b'] = 'replacement'
        result = self.router.send(session_key='a', incarnation='one',
                                  target_id=b['presence_id'], body=marker)
        self.assertEqual(result['reason'], 'target_replaced')
        self.assertEqual(len(self.adapter.forwarded), 1)
        self.error('caller_not_attached', self.router.send, session_key=a['presence_id'],
                   incarnation='one', target_id=b['presence_id'], body=marker)
        self.assertEqual(len(self.adapter.forwarded), 1)

    def test_failure_sanitizes_adapter_exception_and_payload(self):
        self.attach()
        b = self.attach('b')
        self.adapter.live['b'] = 'one'
        self.adapter.failure = True
        result = self.router.send(session_key='a', incarnation='one',
                                  target_id=b['presence_id'], body='SYNTHETIC_PRIVATE_BODY')
        self.assertEqual(result, {'ok': False, 'reason': 'adapter_failure'})
        self.assertEqual(self.adapter.forwarded, [])

    def test_unavailable_adapter_is_explicit(self):
        self.attach()
        b = self.router.attach(session_key='b', incarnation='one', cwd=self.repo,
                               adapter='herdr', route='b')
        result = self.router.send(session_key='a', incarnation='one',
                                  target_id=b['presence_id'], body='hello')
        self.assertEqual(result['reason'], 'identity_binding_unsupported')

    def test_non_git_and_unsafe_paths_refused(self):
        self.error('not_git_repository', self.attach, cwd=self.base)
        link = self.base / 'link'
        link.symlink_to(self.state, target_is_directory=True)
        unsafe = Registry(link)
        self.error('unsafe_state_path', unsafe.peers, cwd=self.repo)

    def test_symlinked_table_and_malformed_table_refused(self):
        self.attach()
        table = self.state / 'sessions' / 'table.json'
        table.unlink()
        target = self.base / 'external'
        target.write_text('{}')
        table.symlink_to(target)
        self.error('unsafe_state_path', self.router.peers, cwd=self.repo)
        table.unlink()
        table.write_text('{broken')
        table.chmod(0o600)
        self.error('invalid_registry', self.router.peers, cwd=self.repo)

    def test_route_refresh_requires_current_binding_and_validated_metadata(self):
        route = {'path': str(self.base / 'socket'), 'nonce': 'initial'}
        a = self.router.attach(session_key='unix', incarnation='one', cwd=self.repo,
                               adapter='unix', route=route)
        fresh = {'path': str(self.base / 'socket2'), 'nonce': 'fresh'}
        b = self.router.set_route(session_key='unix', incarnation='one', adapter='unix', route=fresh)
        self.assertEqual(b['presence_id'], a['presence_id'])
        self.assertEqual(self.router.lookup(session_key='unix', incarnation='one')['route'], fresh)
        switched = self.router.set_route(session_key='unix', incarnation='one',
                                         adapter='manual', route='explicit-manual')
        self.assertEqual(switched['adapter'], 'manual')
        switched = self.router.set_route(session_key='unix', incarnation='one',
                                         adapter='unix', route=fresh)
        self.assertEqual(switched['presence_id'], a['presence_id'])
        self.error('caller_not_attached', self.router.set_route, session_key='other', incarnation='one',
                   adapter='unix', route=fresh)
        self.router.leave(session_key='unix', incarnation='one')
        self.error('session_closed', self.router.set_route, session_key='unix', incarnation='one',
                   adapter='unix', route=fresh)

    def test_observation_context_change_preserves_task_and_warns(self):
        self.attach()
        a = self.router.update(session_key='a', incarnation='one', expected_revision=0,
                               task='old branch task', work_state='working')
        self.git(self.repo, 'checkout', '-q', '-b', 'new-context')
        b = self.router.observe(session_key='a', incarnation='one', sequence=0,
                                runtime_state='idle', cwd=self.repo)
        self.assertIn('task_context_changed', self.router.warnings(b))
        self.assertEqual(a['last_task_update'], b['last_task_update'])
        self.assertEqual(b['task'], a['task'])
        c = self.router.update(session_key='a', incarnation='one', expected_revision=1,
                               task='new branch task', work_state='working')
        self.assertNotIn('task_context_changed', self.router.warnings(c))

    def test_unix_upgrade_hooks_accept_valid_bootstrap_without_changing_live_route(self):
        original = self.attach()
        active = {'path': str(self.base / 'active-socket'), 'nonce': 'a' * 64}
        self.router.set_route(session_key='a', incarnation='one', adapter='unix', route=active)
        bootstrap_metadata = [
            {'route': 'bootstrap'},
            {'adapter': 'manual', 'route': 'bootstrap'},
            {'route': {'path': str(self.base / 'old-socket'), 'nonce': 'b' * 64}},
        ]
        for sequence, metadata in enumerate(bootstrap_metadata, start=1):
            row = self.router.lifecycle(event='checkpoint', session_key='a', incarnation='one',
                                        cwd=self.repo, sequence=sequence, **metadata)
            self.assertEqual(row['presence_id'], original['presence_id'])
            self.assertEqual((row['adapter'], row['route']), ('unix', active))
            self.assertEqual(row['observation_sequence'], sequence)
        row = self.router.lifecycle(event='exit', session_key='a', incarnation='one',
                                    cwd=self.repo, sequence=4, route='bootstrap')
        self.assertEqual((row['state'], row['observation_sequence']), ('closed', 4))
        self.assertEqual((row['adapter'], row['route']), ('unix', active))

    def test_unix_upgrade_hooks_reject_malformed_bootstrap_without_mutation(self):
        self.attach()
        active = {'path': str(self.base / 'active-socket'), 'nonce': 'a' * 64}
        original = self.router.set_route(session_key='a', incarnation='one', adapter='unix', route=active)
        invalid = [
            {'route': {'path': 'relative', 'nonce': 'a' * 64}},
            {'route': {'unknown': 'value'}},
            {'adapter': 'unix', 'route': 'bootstrap'},
            {'adapter': 'manual', 'route': active},
            {'adapter': '', 'route': 'bootstrap'},
        ]
        for metadata in invalid:
            with self.subTest(metadata=metadata), self.assertRaises(RouterError):
                self.router.lifecycle(event='checkpoint', session_key='a', incarnation='one',
                                      cwd=self.repo, sequence=1, **metadata)
            self.assertEqual(self.router.lookup(session_key='a', incarnation='one'), original)

    def test_body_limit(self):
        self.attach()
        b = self.attach('b')
        self.error('invalid_body', self.router.send, session_key='a', incarnation='one',
                   target_id=b['presence_id'], body='x' * 65537)
        self.assertEqual(self.adapter.forwarded, [])


if __name__ == '__main__':
    unittest.main()
