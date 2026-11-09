"""Contract tests for the shared launcher-owned host boundary."""
import contextlib
import errno
import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from providers import host_fence as h

_STATE_PATCH = None
_STATE_DIR = None


def setUpModule():
    # #168: a write-mode run records a repository's protected branches under the
    # state dir. Keep these fixtures' repositories out of the real one.
    global _STATE_PATCH, _STATE_DIR
    _STATE_DIR = tempfile.mkdtemp(prefix="omni_state_")
    _STATE_PATCH = mock.patch.dict(os.environ, {"OMNI_LLM_STATE_DIR": _STATE_DIR})
    _STATE_PATCH.start()


def tearDownModule():
    import shutil
    _STATE_PATCH.stop()
    shutil.rmtree(_STATE_DIR, ignore_errors=True)



class HostFenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.stack = __import__('contextlib').ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(h.platform, 'system', return_value='Darwin'))
        self.stack.enter_context(mock.patch.object(h.kernel_sandbox, 'available', return_value=True))
        self.policy = self.stack.enter_context(mock.patch.object(h.kernel_sandbox, 'agent_fence_policy', return_value={'vendor_state_writable': True}))
        self.wrap_kw = self.stack.enter_context(mock.patch.object(h.kernel_sandbox, 'wrap_kwargs_from_policy', return_value={
            'read_wall': True, 'extra_deny_read': tuple(h._grok_instruction_rows(None))}))
        self.verify = self.stack.enter_context(mock.patch.object(h.kernel_sandbox, 'verify_fence', return_value=None))
        self.wrap = self.stack.enter_context(mock.patch.object(h.kernel_sandbox, 'wrap_argv', side_effect=lambda argv, *a, **kw: ['/usr/bin/sandbox-exec', '-p', 'fixture', *argv]))
        self.stack.enter_context(mock.patch.object(h.kernel_sandbox, 'agent_child_env', side_effect=lambda env, **kw: env))

    def prepare(self, **kw):
        return h.prepare_host_fence(['/fake/grok', '--sandbox', 'strict', '-p', 'fixture'], provider='grok', workdir=self.tmp.name, env={}, **kw)

    def test_read_only_reaches_policy_verification_and_application(self):
        argv, env, meta = self.prepare(read_only=True)
        self.assertTrue(self.policy.call_args.kwargs['read_only'])
        self.assertTrue(self.verify.call_args.kwargs['read_only'])
        self.assertTrue(self.wrap.call_args.args[2])
        self.assertFalse(meta['fence_detail']['workdir_writable'])
        self.assertEqual(argv[3:6], ['/fake/grok', '--sandbox', 'off'])

    def test_red_host_check_refuses_before_wrap(self):
        self.verify.return_value = {'ok': False, 'reason': 'fixture-denied'}
        with self.assertRaises(h.HostFenceError) as caught:
            self.prepare()
        self.assertEqual(caught.exception.envelope['reason'], 'fixture-denied')
        self.wrap.assert_not_called()

    def test_unwrapped_postcondition_is_not_accepted(self):
        self.wrap.side_effect = lambda argv, *a, **kw: argv
        with self.assertRaises(h.HostFenceError) as caught:
            self.prepare()
        self.assertEqual(caught.exception.envelope['reason'], 'fence_unavailable')

    def test_missing_host_mechanism_cannot_spawn_vendor(self):
        with mock.patch.object(h.kernel_sandbox, 'available', return_value=False), mock.patch.object(h.os, 'execvpe') as execute:
            rc = h.main(['--provider', 'grok', '--workdir', self.tmp.name, '--', '/fake/grok'])
        self.assertEqual(rc, 126)
        execute.assert_not_called()

    def test_environment_claim_cannot_bypass_verification(self):
        h.prepare_host_fence(['/fake/grok'], provider='grok', workdir=self.tmp.name,
            env={'OMNI_LLM_AGENT_FENCE': '0', 'HOST_FENCED': '1'})
        self.verify.assert_called_once()
        self.wrap.assert_called_once()

    def test_non_macos_contract_is_preserved(self):
        original = ['/fake/grok', '--sandbox', 'strict']
        env = {'FIXTURE': 'value'}
        with mock.patch.object(h.platform, 'system', return_value='Linux'):
            argv, actual_env, metadata = h.prepare_host_fence(original, provider='grok', workdir=self.tmp.name, env=env)
        self.assertEqual(argv, original)
        self.assertEqual(actual_env, env)
        self.assertEqual(metadata['fence'], 'none')
        self.verify.assert_not_called()

    def test_codex_read_only_app_server_keeps_inner_sandbox_and_outer_fence(self):
        binary = '/opt/codex/bin/codex'
        original = [binary, 'app-server']
        with mock.patch.object(h.vendor_bin, 'which', return_value=binary), \
                mock.patch.object(
                    h.kernel_sandbox, 'disable_inner_codex_sandbox',
                    wraps=h.kernel_sandbox.disable_inner_codex_sandbox) as rewrite:
            argv, _env, metadata = h.prepare_host_fence(
                original, provider='codex', workdir=self.tmp.name,
                read_only=True, env={})
        self.assertTrue(h.kernel_sandbox.is_wrapped(argv))
        self.assertEqual(argv[-2:], original)
        self.assertNotIn(h.kernel_sandbox.INNER_SANDBOX_BYPASS, argv)
        self.assertTrue(self.verify.call_args.kwargs['read_only'])
        self.assertFalse(metadata['fence_detail']['workdir_writable'])
        rewrite.assert_not_called()

    def test_codex_app_server_exception_rejects_near_matches(self):
        binary = '/opt/codex/bin/codex'
        link = os.path.join(self.tmp.name, 'codex-link')
        os.symlink(binary, link)
        cases = (
            ([binary, 'app-server', '-c', 'x'], True),
            ([binary, 'exec', 'app-server'], True),
            ([binary, 'exec', '--', 'app-server'], True),
            (['/usr/bin/env', 'codex', 'app-server'], True),
            (['/opt/other-codex/bin/codex', 'app-server'], True),
            ([link, 'app-server'], True),
            (['codex', 'app-server'], True),
            ([binary, 'app-server'], False),
        )
        with mock.patch.object(h.vendor_bin, 'which', return_value=binary):
            for original, read_only in cases:
                with self.subTest(argv=original, read_only=read_only):
                    if read_only:
                        argv, _env, _metadata = h.prepare_host_fence(
                            original, provider='codex', workdir=self.tmp.name,
                            read_only=True, env={})
                    else:
                        with mock.patch.object(
                                h, '_repo_write_roots',
                                return_value=((), (), (), (), ())):
                            argv, _env, _metadata = h.prepare_host_fence(
                                original, provider='codex', workdir=self.tmp.name,
                                read_only=False, env={})
                    self.assertTrue(h.kernel_sandbox.is_wrapped(argv))
                    self.assertIn(h.kernel_sandbox.INNER_SANDBOX_BYPASS, argv)

    def test_codex_exec_keeps_existing_inner_sandbox_rewrite(self):
        binary = '/opt/codex/bin/codex'
        with mock.patch.object(h.vendor_bin, 'which', return_value=binary):
            argv, _env, _metadata = h.prepare_host_fence(
                [binary, 'exec', '--sandbox', 'read-only'],
                provider='codex', workdir=self.tmp.name,
                read_only=True, env={})
        self.assertTrue(h.kernel_sandbox.is_wrapped(argv))
        self.assertIn(h.kernel_sandbox.INNER_SANDBOX_BYPASS, argv)
        self.assertFalse(h.kernel_sandbox.inner_vendor_sandbox_present(argv))

    def test_grok_refuses_when_global_instruction_denies_are_missing(self):
        rows = h._grok_instruction_rows(None)
        for label, wrap_kw in (
                ('no rows', {'read_wall': True}),
                ('one row short', {'read_wall': True, 'extra_deny_read': tuple(rows[1:])}),
                ('no read wall', {'read_wall': False, 'extra_deny_read': tuple(rows)})):
            with self.subTest(case=label):
                self.wrap_kw.return_value = wrap_kw
                with self.assertRaises(h.HostFenceError) as caught:
                    self.prepare(read_only=True)
                self.assertEqual(caught.exception.envelope['reason'], 'fence_unavailable')
                self.assertIn('global-instruction', caught.exception.envelope['detail'])

    def test_grok_instruction_rows_cover_grok_home_and_compat_roots(self):
        home = h.kernel_sandbox.protected_home()
        custom = os.path.join(self.tmp.name, 'grok-home')
        rows = h._grok_instruction_rows(custom)
        for root in (custom, os.path.join(home, '.claude'), os.path.join(home, '.cursor')):
            for name in ('AGENTS.md', 'Agents.md', 'CLAUDE.md', 'Claude.md', 'CLAUDE.local.md'):
                self.assertIn(('literal', os.path.join(root, name)), rows)
            self.assertIn(('subpath', os.path.join(root, 'rules')), rows)
        self.assertNotIn(('literal', os.path.join(home, '.grok', 'AGENTS.md')), rows)
        # The postcondition follows the child's GROK_HOME, not the default root.
        with self.assertRaises(h.HostFenceError):
            h.prepare_host_fence(['/fake/grok'], provider='grok', workdir=self.tmp.name,
                                 env={'GROK_HOME': custom})
        self.wrap_kw.return_value = {'read_wall': True, 'extra_deny_read': tuple(rows)}
        argv, _env, _meta = h.prepare_host_fence(
            ['/fake/grok'], provider='grok', workdir=self.tmp.name, env={'GROK_HOME': custom})
        self.assertEqual(argv[0], '/usr/bin/sandbox-exec')

    def test_command_starting_with_dash_is_refused(self):
        # sandbox-exec would read it as its own -D/-p option.
        for argv in (['-D', 'G0=1', '/fake/grok'], ['-p', '(version 1)(allow default)', '/bin/true'],
                     [b'-D', 'H0=/nonexistent', '/usr/bin/true'], [pathlib.PurePosixPath('-D'), 'G0=1', '/bin/true'],
                     [42, '/bin/true']):
            with self.assertRaises(h.HostFenceError) as caught:
                h.prepare_host_fence(argv, provider='grok', workdir=self.tmp.name, env={})
            self.assertEqual(caught.exception.envelope['reason'], 'fence_path_rejected')
        self.wrap.assert_not_called()

    def test_grok_pid_placeholder_gets_the_pid_shim(self):
        placeholder = ['-D', '%s=%s' % (h.kernel_sandbox.GROK_PID_PARAM, h.kernel_sandbox.GROK_PID_PLACEHOLDER)]
        self.wrap.side_effect = lambda argv, *a, **kw: ['/usr/bin/sandbox-exec', '-p', 'fixture', *placeholder, *argv]
        argv, _env, _meta = self.prepare(read_only=True)
        self.assertEqual(argv[:3], ['/bin/sh', '-c', h.kernel_sandbox.GROK_PID_SHIM])
        self.assertEqual(argv[3:8], ['/usr/bin/sandbox-exec', '-p', 'fixture', *placeholder])
        self.assertEqual(argv[8:11], ['/fake/grok', '--sandbox', 'off'])
        # Without the placeholder right after the profile there is nothing to bind.
        self.wrap.side_effect = lambda argv, *a, **kw: ['/usr/bin/sandbox-exec', *placeholder, '-p', 'fixture', *argv]
        argv, _env, _meta = self.prepare(read_only=True)
        self.assertEqual(argv[0], '/usr/bin/sandbox-exec')

    def test_f10_later_read_allow_over_instruction_rows_is_refused(self):
        home = h.kernel_sandbox.protected_home()
        grok_root = os.path.join(home, '.grok')
        for entry in (('subpath', grok_root), ('literal', os.path.join(grok_root, 'AGENTS.md')),
                      ('subpath', home), ('regex', '.*')):
            with self.subTest(entry=entry):
                self.wrap_kw.return_value = {
                    'read_wall': True,
                    'extra_deny_read': tuple(h._grok_instruction_rows(None)),
                    'extra_post_read': (entry,)}
                with self.assertRaises(h.HostFenceError) as caught:
                    self.prepare(read_only=True)
                self.assertEqual(caught.exception.envelope['reason'], 'fence_unavailable')

    def test_r2_4_later_read_allow_in_another_case_is_refused(self):
        home = h.kernel_sandbox.protected_home()
        grok_root = os.path.join(home, '.grok')
        for entry in (('literal', os.path.join(grok_root, 'aGeNtS.md')),
                      ('subpath', os.path.join(grok_root, 'RULES')),
                      ('literal', os.path.join(grok_root, 'Rules', 'topic.md')),
                      ('subpath', os.path.join(home, '.GROK')),
                      ('literal', os.path.join(home, '.Claude', 'claude.MD'))):
            with self.subTest(entry=entry):
                self.wrap_kw.return_value = {
                    'read_wall': True,
                    'extra_deny_read': tuple(h._grok_instruction_rows(None)),
                    'extra_post_read': (entry,)}
                with self.assertRaises(h.HostFenceError) as caught:
                    self.prepare(read_only=True)
                self.assertEqual(caught.exception.envelope['reason'], 'fence_unavailable')

    def test_r2_3_relative_grok_home_is_refused(self):
        os.mkdir(os.path.join(self.tmp.name, 'sub'))
        old_cwd = os.getcwd()
        os.chdir(self.tmp.name)
        self.addCleanup(os.chdir, old_cwd)
        # Rows as the parent would compute them for the relative value.
        self.wrap_kw.return_value = {
            'read_wall': True,
            'extra_deny_read': tuple(h._grok_instruction_rows(os.path.abspath('sub')))}
        for kw in ({'env': {'GROK_HOME': 'sub'}},
                   {'env': {}, 'config_dir': 'sub'},
                   {'env': {'GROK_HOME': 'sub'}, 'config_dir': 'sub'}):
            with self.subTest(kw=kw):
                with self.assertRaises(h.HostFenceError) as caught:
                    h.prepare_host_fence(['/fake/grok'], provider='grok',
                                         workdir=self.tmp.name, **kw)
                self.assertEqual(caught.exception.envelope['reason'], 'fence_path_rejected')

    def test_u2_later_allow_through_a_path_alias_is_refused(self):
        grok_home = os.path.realpath(tempfile.mkdtemp(prefix='omni_grok_home_', dir='/private/tmp'))
        self.addCleanup(__import__('shutil').rmtree, grok_home, True)
        rows = tuple(h._grok_instruction_rows(grok_home))
        alias = '/tmp' + grok_home[len('/private/tmp'):]
        self.assertTrue(grok_home.startswith('/private/tmp/'))
        for entry in (('subpath', '/tmp'), ('subpath', alias),
                      ('literal', os.path.join(alias, 'AGENTS.md')),
                      ('subpath', os.path.join(alias, 'rules'))):
            with self.subTest(entry=entry):
                self.wrap_kw.return_value = {'read_wall': True, 'extra_deny_read': rows,
                                             'extra_post_read': (entry,)}
                with self.assertRaises(h.HostFenceError) as caught:
                    h.prepare_host_fence(['/fake/grok'], provider='grok',
                                         workdir=self.tmp.name, env={'GROK_HOME': grok_home})
                self.assertEqual(caught.exception.envelope['reason'], 'fence_unavailable')
        # An unrelated later allow still passes.
        self.wrap_kw.return_value = {'read_wall': True, 'extra_deny_read': rows,
                                     'extra_post_read': (('subpath', self.tmp.name),)}
        h.prepare_host_fence(['/fake/grok'], provider='grok',
                             workdir=self.tmp.name, env={'GROK_HOME': grok_home})

    def test_non_grok_provider_does_not_need_grok_instruction_rows(self):
        self.wrap_kw.return_value = {'read_wall': True}
        argv, _env, _meta = h.prepare_host_fence(
            ['/fake/claude'], provider='claude', workdir=self.tmp.name, env={})
        self.assertEqual(argv[-1], '/fake/claude')

    def test_grok_known_value_flags_keep_sandbox_shaped_operands(self):
        for flag in ('--system-prompt-override', '--prompt-json'):
            for value in ('--sandbox', '--sandbox=strict'):
                with self.subTest(flag=flag, value=value):
                    argv, _env, _meta = h.prepare_host_fence(
                        ['/fake/grok', flag, value, '--sandbox', 'strict'],
                        provider='grok', workdir=self.tmp.name, env={})
                    self.assertEqual(argv[3:],
                                     ['/fake/grok', '--sandbox', 'off', flag, value])

    def test_cursor_inner_sandbox_is_disabled_after_wrap(self):
        argv, _env, _meta = h.prepare_host_fence(
            ['/fake/cursor-agent', '-p', '--mode', 'ask', '--sandbox', 'enabled',
             '--workspace', self.tmp.name, '--model', 'composer-2.5'],
            provider='cursor', workdir=self.tmp.name, env={})
        self.assertEqual(argv[0], '/usr/bin/sandbox-exec')
        self.assertEqual(h._sandbox_flag_values(argv), ('disabled',))
        self.assertEqual(argv[3:6], ['/fake/cursor-agent', '--sandbox', 'disabled'])

    def test_cursor_known_value_flags_keep_sandbox_shaped_operands(self):
        for flag in ('--model', '--workspace', '--mode'):
            for value in ('--sandbox', '--sandbox=enabled'):
                with self.subTest(flag=flag, value=value):
                    argv, _env, _meta = h.prepare_host_fence(
                        ['/fake/cursor-agent', flag, value, '--sandbox', 'enabled'],
                        provider='cursor', workdir=self.tmp.name, env={})
                    self.assertEqual(argv[3:],
                                     ['/fake/cursor-agent', '--sandbox', 'disabled',
                                      flag, value])

    def test_cursor_non_macos_keeps_vendor_sandbox(self):
        original = ['/fake/cursor-agent', '--sandbox', 'enabled']
        with mock.patch.object(h.platform, 'system', return_value='Linux'):
            argv, _env, metadata = h.prepare_host_fence(
                original, provider='cursor', workdir=self.tmp.name, env={})
        self.assertEqual(argv, original)
        self.assertEqual(metadata['fence'], 'none')

    def test_cursor_disable_postcondition_refuses_when_enabled_remains(self):
        original = ['/fake/cursor-agent', '--sandbox', 'enabled']
        with mock.patch.object(h, '_rewrite_inner_sandbox',
                               side_effect=lambda wrapped, orig, *a, **k: (wrapped, orig)):
            with self.assertRaises(h.HostFenceError) as caught:
                h.prepare_host_fence(
                    original, provider='cursor', workdir=self.tmp.name, env={})
        self.assertEqual(caught.exception.envelope['reason'], 'fence_unavailable')
        self.assertIn('inner vendor sandbox disable',
                      caught.exception.envelope['detail'])

    def test_cursor_trailing_bare_sandbox_is_stripped(self):
        argv, _env, _meta = h.prepare_host_fence(
            ['/fake/cursor-agent', '--sandbox', 'enabled', '--sandbox'],
            provider='cursor', workdir=self.tmp.name, env={})
        self.assertEqual(argv[3:],
                         ['/fake/cursor-agent', '--sandbox', 'disabled'])
        self.assertEqual(
            h._sandbox_flag_values(argv[3:], h._CURSOR_VALUE_OPTIONS),
            ('disabled',))

    def test_cursor_disable_postcondition_refuses_leftover_bare_sandbox(self):
        leftover = ['/fake/cursor-agent', '--sandbox', 'disabled', '--sandbox']
        original = ['/fake/cursor-agent', '--sandbox', 'enabled']
        with mock.patch.object(
                h, '_rewrite_inner_sandbox',
                side_effect=lambda wrapped, orig, *a, **k: (
                    wrapped[:-len(orig)] + leftover, leftover)):
            with self.assertRaises(h.HostFenceError) as caught:
                h.prepare_host_fence(
                    original, provider='cursor', workdir=self.tmp.name, env={})
        self.assertEqual(caught.exception.envelope['reason'], 'fence_unavailable')

    def _git(self, cwd, *args):
        return subprocess.run(
            ['git', '-C', str(cwd), *args], check=True,
            capture_output=True, text=True,
            env={**os.environ, 'GIT_AUTHOR_NAME': 'Fence test',
                 'GIT_AUTHOR_EMAIL': 'fence@example.invalid',
                 'GIT_COMMITTER_NAME': 'Fence test',
                 'GIT_COMMITTER_EMAIL': 'fence@example.invalid'},
        )

    def test_repo_write_opt_in_adds_repo_and_git_metadata_roots(self):
        repo = os.path.join(self.tmp.name, 'repo')
        os.mkdir(repo)
        repo = os.path.realpath(repo)
        self._git(repo, 'init', '-q')
        self._git(repo, 'checkout', '-qb', 'work')
        with mock.patch.dict(os.environ, {
                'GIT_DIR': os.path.join(self.tmp.name, 'foreign', '.git'),
                'GIT_WORK_TREE': os.path.join(self.tmp.name, 'foreign'),
        }):
            argv, _env, _meta = h.prepare_host_fence(
                ['/fake/claude'], provider='claude', workdir=repo, env={},
                allow_repo_write=True,
            )
        extras = self.verify.call_args.kwargs['extra_write_paths']
        paths = {entry.path for entry in extras}
        repo = os.path.realpath(repo)
        self.assertEqual(paths, {repo, os.path.join(repo, '.git')})
        self.assertEqual(self.wrap.call_args.kwargs['extra_write_paths'], extras)
        self.assertEqual(argv[-1], '/fake/claude')

    def test_repo_write_opt_in_rejects_symlinked_git_entry(self):
        repo = os.path.realpath(os.path.join(self.tmp.name, 'repo'))
        os.mkdir(repo)
        self._git(repo, 'init', '-q')
        self._git(repo, 'checkout', '-qb', 'work')
        metadata = os.path.join(self.tmp.name, 'real-git-metadata')
        os.rename(os.path.join(repo, '.git'), metadata)
        os.symlink(metadata, os.path.join(repo, '.git'))
        with self.assertRaises(h.HostFenceError) as caught:
            h.prepare_host_fence(['/fake/claude'], provider='claude',
                                 workdir=repo, env={}, allow_repo_write=True)
        self.assertEqual(caught.exception.envelope['reason'], 'fence_path_rejected')

    def test_repo_write_opt_in_refuses_preexisting_hardlink_before_wrap(self):
        repo = os.path.realpath(os.path.join(self.tmp.name, 'repo'))
        os.mkdir(repo)
        self._git(repo, 'init', '-q')
        self._git(repo, 'checkout', '-qb', 'work')
        outside = os.path.join(self.tmp.name, 'outside')
        with open(outside, 'w', encoding='utf-8') as stream:
            stream.write('outside sentinel')
        os.link(outside, os.path.join(repo, 'inside-alias'))
        with self.assertRaises(h.HostFenceError) as caught:
            h.prepare_host_fence(['/fake/claude'], provider='claude', workdir=repo,
                                 env={}, allow_repo_write=True)
        self.assertEqual(caught.exception.envelope['reason'], 'fence_path_rejected')
        self.wrap.assert_not_called()
        with open(outside, encoding='utf-8') as stream:
            self.assertEqual(stream.read(), 'outside sentinel')

    def test_repo_write_opt_in_resolves_linked_worktree_common_metadata(self):
        repo = os.path.join(self.tmp.name, 'main')
        linked = os.path.join(self.tmp.name, 'linked')
        os.mkdir(repo)
        repo = os.path.realpath(repo)
        self._git(repo, 'init', '-q')
        self._git(repo, 'commit', '--allow-empty', '-qm', 'seed')
        linked = os.path.realpath(linked)
        self._git(repo, 'worktree', 'add', '-qb', 'fixture', linked)
        h.prepare_host_fence(
            ['/fake/codex'], provider='codex', workdir=linked, env={},
            allow_repo_write=True,
        )
        paths = {entry.path for entry in self.verify.call_args.kwargs['extra_write_paths']}
        common = os.path.realpath(os.path.join(repo, '.git'))
        gitdir = os.path.realpath(
            self._git(linked, 'rev-parse', '--absolute-git-dir').stdout.strip())
        linked = os.path.realpath(linked)
        self.assertEqual(paths, {linked, common, gitdir})
        denies = self.verify.call_args.kwargs['extra_deny_write']
        self.assertIn(('subpath', os.path.join(common, 'worktrees')), denies)
        self.assertIn(('subpath', gitdir), self.verify.call_args.kwargs['extra_post_write'])
        finals = self.verify.call_args.kwargs['extra_deny_write_final']
        self.assertIn(('literal', os.path.join(gitdir, 'HEAD')), finals)
        self.assertIn(('literal', gitdir), finals)

    def test_repo_write_opt_in_refuses_read_only_non_git_and_non_macos(self):
        repo = os.path.join(self.tmp.name, 'repo')
        os.mkdir(repo)
        repo = os.path.realpath(repo)
        self._git(repo, 'init', '-q')
        with self.assertRaises(h.HostFenceError) as caught:
            h.prepare_host_fence(['/fake/codex'], provider='codex', workdir=repo,
                                 env={}, allow_repo_write=True, read_only=True)
        self.assertEqual(caught.exception.envelope['reason'], 'fence_path_rejected')
        with self.assertRaises(h.HostFenceError):
            h.prepare_host_fence(['/fake/claude'], provider='claude',
                                 workdir=self.tmp.name, env={}, allow_repo_write=True)
        with mock.patch.object(h.platform, 'system', return_value='Linux'):
            with self.assertRaises(h.HostFenceError) as caught:
                h.prepare_host_fence(['/fake/codex'], provider='codex', workdir=repo,
                                     env={}, allow_repo_write=True)
        self.assertEqual(caught.exception.envelope['reason'], 'fence_unavailable')

    def test_host_exec_cli_passes_repo_write_opt_in(self):
        with mock.patch.object(h, 'prepare_host_fence', return_value=(['/fake'], {}, {})) as prepare, \
             mock.patch.object(h.os, 'chdir'), mock.patch.object(h.os, 'execvpe'):
            h.main(['--provider', 'claude', '--workdir', self.tmp.name,
                    '--allow-repo-write', '--', '/fake'])
        self.assertTrue(prepare.call_args.kwargs['allow_repo_write'])

    def test_write_mode_git_worktree_admits_repo_without_opt_in(self):
        repo = os.path.realpath(os.path.join(self.tmp.name, 'repo'))
        os.mkdir(repo)
        self._git(repo, 'init', '-q')
        self._git(repo, 'checkout', '-qb', 'work')
        _argv, env, _meta = h.prepare_host_fence(
            ['/fake/claude'], provider='claude', workdir=repo, env={})
        extras = self.verify.call_args.kwargs['extra_write_paths']
        paths = {entry.path for entry in extras}
        self.assertEqual(paths, {repo, os.path.join(repo, '.git')})
        denies = self.verify.call_args.kwargs['extra_deny_write']
        gitdir = os.path.join(repo, '.git')
        self.assertIn(('subpath', os.path.join(gitdir, 'hooks')), denies)
        self.assertIn(('literal', os.path.join(gitdir, 'refs', 'heads', 'main')), denies)
        self.assertIn(('literal', os.path.join(gitdir, 'refs', 'heads', 'Main')), denies)
        self.assertIn(('literal', gitdir), denies)
        self.assertIn(('subpath', os.path.join(gitdir, 'worktrees')), denies)
        for leaf in ('config', 'packed-refs', 'HEAD'):
            self.assertIn(('literal', os.path.join(gitdir, leaf)), denies)
        self.assertEqual(env.get('GIT_ALLOW_PROTOCOL'), '')
        exec_denies = {path for _kind, path in self.verify.call_args.kwargs.get('extra_deny_exec') or ()}
        self.assertTrue(any(path.endswith('git-remote-https') for path in exec_denies), exec_denies)

    def test_write_mode_skips_repo_metadata_on_protected_branch(self):
        repo = os.path.realpath(os.path.join(self.tmp.name, 'repo'))
        os.mkdir(repo)
        self._git(repo, 'init', '-q')
        self._git(repo, 'checkout', '-B', 'main')
        h.prepare_host_fence(
            ['/fake/claude'], provider='claude', workdir=repo, env={})
        extras = self.verify.call_args.kwargs.get('extra_write_paths') or ()
        self.assertEqual(tuple(extras), ())
        gitdir = os.path.join(repo, '.git')
        denies = self.verify.call_args.kwargs.get('extra_deny_write') or ()
        self.assertIn(('literal', os.path.join(gitdir, 'config')), denies)
        self.assertIn(('literal', os.path.join(gitdir, 'packed-refs')), denies)
        self.assertIn(('literal', os.path.join(gitdir, 'HEAD')), denies)

    def test_required_repo_write_refuses_undetermined_branch(self):
        repo = os.path.realpath(os.path.join(self.tmp.name, 'repo'))
        os.mkdir(repo)
        self._git(repo, 'init', '-q')
        self._git(repo, 'checkout', '-qb', 'work')
        with open(os.path.join(repo, '.git', 'HEAD'), 'w', encoding='utf-8') as fh:
            fh.write('not-a-ref')
        with self.assertRaises(h.HostFenceError) as caught:
            h.prepare_host_fence(['/fake/claude'], provider='claude',
                                 workdir=repo, env={}, allow_repo_write=True)
        self.assertEqual(caught.exception.envelope['reason'], 'fence_path_rejected')

    def test_vendor_post_write_overlapping_git_metadata_is_refused(self):
        repo = os.path.realpath(os.path.join(self.tmp.name, 'repo'))
        os.mkdir(repo)
        self._git(repo, 'init', '-q')
        self._git(repo, 'checkout', '-qb', 'work')
        gitdir = os.path.join(repo, '.git')
        with mock.patch.object(
                h.kernel_sandbox, 'wrap_kwargs_from_policy',
                return_value={
                    'read_wall': True,
                    'extra_post_write': (('subpath', gitdir),),
                }):
            with self.assertRaises(h.HostFenceError) as caught:
                h.prepare_host_fence(['/fake/claude'], provider='claude',
                                     workdir=repo, env={})
        self.assertEqual(caught.exception.envelope['reason'], 'fence_path_rejected')


class DirectConsumerBoundaryTests(unittest.TestCase):
    def test_macos_missing_fence_refuses_each_consumer_before_run_cli(self):
        import asyncio
        from providers import claude, codex, cursor, pi
        cases = [(claude, claude.run_claude_subprocess, 'sonnet'),
                 (codex, codex.run_codex_subprocess, 'gpt-5.6-sol'),
                 (cursor, cursor.run_cursor_subprocess, 'fable'),
                 (pi, pi.run_pi_subprocess, 'fixture')]
        for module, fn, model in cases:
            with self.subTest(provider=module.__name__), \
                 mock.patch.object(h.platform, 'system', return_value='Darwin'), \
                 mock.patch.object(h.kernel_sandbox, 'available', return_value=False), \
                 mock.patch('shutil.which', return_value='/fake/bin/provider'), \
                 mock.patch.object(os.path, 'isfile', return_value=True), \
                 mock.patch.object(os, 'access', return_value=True), \
                 mock.patch.object(module, 'env_depth', return_value=0), \
                 mock.patch.object(module, 'run_cli', new_callable=mock.AsyncMock) as execute:
                result = asyncio.run(fn('synthetic fixture', model, 30))
            self.assertEqual(result.get('reason'), 'fence_unavailable', result)
            execute.assert_not_called()


@unittest.skipUnless(
    h.platform.system() == 'Darwin' and h.kernel_sandbox.available(),
    'requires the live macOS Seatbelt host fence',
)
class RepoWriteFenceLiveTests(unittest.TestCase):
    def test_opt_in_can_write_linked_repo_and_git_metadata_only(self):
        with tempfile.TemporaryDirectory(prefix='omni_repo_write_', dir='/private/tmp') as raw:
            root = Path(raw)
            main = root / 'main'
            linked = root / 'linked'
            main.mkdir()
            subprocess.run(['git', '-C', str(main), 'init', '-q'], check=True)
            identity = {
                **os.environ,
                'GIT_AUTHOR_NAME': 'Fence test',
                'GIT_AUTHOR_EMAIL': 'fence@example.invalid',
                'GIT_COMMITTER_NAME': 'Fence test',
                'GIT_COMMITTER_EMAIL': 'fence@example.invalid',
            }
            subprocess.run(['git', '-C', str(main), 'commit', '--allow-empty', '-qm', 'seed'],
                           check=True, env=identity)
            subprocess.run(['git', '-C', str(main), 'worktree', 'add', '-qb', 'feature',
                            str(linked)], check=True)
            common = Path(self._git_path(linked, '--git-common-dir'))
            gitdir = Path(self._git_path(linked, '--absolute-git-dir'))
            if not common.is_absolute():
                common = (linked / common).resolve()
            if not gitdir.is_absolute():
                gitdir = (linked / gitdir).resolve()
            source_probe = linked / 'ordinary-write-probe'
            common_probe = common / 'common-write-probe'
            gitdir_probe = gitdir / 'worktree-write-probe'
            outside_probe = root / 'outside-write-probe'
            command, child_env, _meta = h.prepare_host_fence(
                ['/usr/bin/touch', str(source_probe), str(common_probe), str(gitdir_probe)],
                provider='claude', workdir=str(linked.resolve()), env=identity,
                allow_repo_write=True,
            )
            written = subprocess.run(command, cwd=str(linked), env=child_env,
                                     capture_output=True, text=True, check=False)
            self.assertEqual(written.returncode, 0, written.stderr)
            self.assertTrue(source_probe.is_file())
            self.assertTrue(common_probe.is_file())
            self.assertTrue(gitdir_probe.is_file())
            self.assertEqual(child_env['GIT_CONFIG_GLOBAL'], os.devnull)
            self.assertEqual(child_env['GIT_CONFIG_SYSTEM'], os.devnull)
            self.assertEqual(child_env['GIT_CONFIG_NOSYSTEM'], '1')

            outside_sentinel = root / 'outside-sentinel'
            outside_sentinel.write_text('unchanged', encoding='utf-8')
            alias = linked / 'outside-alias'
            link_argv, link_env, _meta = h.prepare_host_fence(
                ['/bin/ln', str(outside_sentinel), str(alias)],
                provider='claude', workdir=str(linked.resolve()), env=identity,
                allow_repo_write=True,
            )
            linked_result = subprocess.run(link_argv, cwd=str(linked), env=link_env,
                                           capture_output=True, text=True, check=False)
            self.assertNotEqual(linked_result.returncode, 0)
            self.assertFalse(alias.exists())
            self.assertEqual(outside_sentinel.read_text(encoding='utf-8'), 'unchanged')

            denied_argv, denied_env, _meta = h.prepare_host_fence(
                ['/usr/bin/touch', str(outside_probe)],
                provider='claude', workdir=str(linked.resolve()), env=identity,
                allow_repo_write=True,
            )
            denied = subprocess.run(denied_argv, cwd=str(linked), env=denied_env,
                                    capture_output=True, text=True, check=False)
            self.assertNotEqual(denied.returncode, 0)
            self.assertFalse(outside_probe.exists())

            hooks_dir = common / 'hooks'
            hooks_dir.mkdir(exist_ok=True)
            hook = hooks_dir / 'pre-commit'
            hook_argv, hook_env, _meta = h.prepare_host_fence(
                ['/usr/bin/touch', str(hook)],
                provider='claude', workdir=str(linked.resolve()), env=identity,
                allow_repo_write=True,
            )
            hooked = subprocess.run(hook_argv, cwd=str(linked), env=hook_env,
                                    capture_output=True, text=True, check=False)
            self.assertNotEqual(hooked.returncode, 0, hooked.stderr)
            self.assertFalse(hook.exists())

            packed = common / 'packed-refs'
            packed_argv, packed_env, _meta = h.prepare_host_fence(
                ['/bin/sh', '-c', 'printf "x\\n" > "%s"' % packed],
                provider='claude', workdir=str(linked.resolve()), env=identity,
                allow_repo_write=True,
            )
            packed_run = subprocess.run(
                packed_argv, cwd=str(linked), env=packed_env,
                capture_output=True, text=True, check=False)
            self.assertNotEqual(packed_run.returncode, 0, packed_run.stderr)

            checkout_argv, checkout_env, _meta = h.prepare_host_fence(
                ['/bin/sh', '-c', 'git checkout -B main'],
                provider='claude', workdir=str(linked.resolve()), env=identity,
                allow_repo_write=True,
            )
            checked = subprocess.run(
                checkout_argv, cwd=str(linked), env=checkout_env,
                capture_output=True, text=True, check=False)
            self.assertNotEqual(checked.returncode, 0, checked.stderr)
            head = subprocess.check_output(
                ['git', '-C', str(linked), 'symbolic-ref', '--short', 'HEAD'],
                text=True).strip()
            self.assertEqual(head, 'feature')

            hooks_path_argv, hooks_path_env, _meta = h.prepare_host_fence(
                ['/bin/sh', '-c', 'git config core.hooksPath hooks'],
                provider='claude', workdir=str(linked.resolve()), env=identity,
                allow_repo_write=True,
            )
            hooked_cfg = subprocess.run(
                hooks_path_argv, cwd=str(linked), env=hooks_path_env,
                capture_output=True, text=True, check=False)
            self.assertNotEqual(hooked_cfg.returncode, 0, hooked_cfg.stderr)

            mv_argv, mv_env, _meta = h.prepare_host_fence(
                ['/bin/mv', str(linked / '.git'), str(linked / '.git-away')],
                provider='claude', workdir=str(linked.resolve()), env=identity,
                allow_repo_write=True,
            )
            moved = subprocess.run(mv_argv, cwd=str(linked), env=mv_env,
                                   capture_output=True, text=True, check=False)
            self.assertNotEqual(moved.returncode, 0, moved.stderr)
            self.assertTrue((linked / '.git').exists())
            self.assertFalse((linked / '.git-away').exists())

            sibling = common / 'worktrees' / 'evil'
            sib_argv, sib_env, _meta = h.prepare_host_fence(
                ['/bin/mkdir', str(sibling)],
                provider='claude', workdir=str(linked.resolve()), env=identity,
                allow_repo_write=True,
            )
            sib = subprocess.run(sib_argv, cwd=str(linked), env=sib_env,
                                 capture_output=True, text=True, check=False)
            self.assertNotEqual(sib.returncode, 0, sib.stderr)
            self.assertFalse(sibling.exists())

            remote_argv, remote_env, _meta = h.prepare_host_fence(
                ['/bin/sh', '-c',
                 'env -u GIT_ALLOW_PROTOCOL git ls-remote https://example.com/'],
                provider='claude', workdir=str(linked.resolve()), env=identity,
                allow_repo_write=True,
            )
            remote = subprocess.run(
                remote_argv, cwd=str(linked), env=remote_env,
                capture_output=True, text=True, check=False)
            self.assertNotEqual(remote.returncode, 0, remote.stderr)
            combined = (remote.stdout or '') + (remote.stderr or '')
            self.assertNotIn("Could not resolve host", combined)
            self.assertNotIn("not found", combined.lower())

    @staticmethod
    def _git_path(repo, option):
        result = subprocess.run(['git', '-C', str(repo), 'rev-parse', option],
                                check=True, capture_output=True, text=True)
        value = result.stdout.strip()
        return value if os.path.isabs(value) else os.path.join(str(repo), value)

    def test_real_linked_worktree_add_commit_updates_common_ref(self):
        with tempfile.TemporaryDirectory(prefix='omni_git_commit_', dir='/private/tmp') as raw:
            root = Path(raw)
            main = root / 'main'
            linked = root / 'linked'
            main.mkdir()
            env = {
                **os.environ,
                'GIT_AUTHOR_NAME': 'Fence test',
                'GIT_AUTHOR_EMAIL': 'fence@example.invalid',
                'GIT_COMMITTER_NAME': 'Fence test',
                'GIT_COMMITTER_EMAIL': 'fence@example.invalid',
            }
            subprocess.run(['git', '-C', str(main), 'init', '-q'], check=True)
            (main / 'base.txt').write_text('base\n', encoding='utf-8')
            subprocess.run(['git', '-C', str(main), 'add', '--', 'base.txt'],
                           check=True, env=env)
            subprocess.run(['git', '-C', str(main), 'commit', '-qm', 'seed'],
                           check=True, env=env)
            subprocess.run(['git', '-C', str(main), 'worktree', 'add', '-qb', 'feature',
                            str(linked)], check=True)
            (linked / 'change.txt').write_text('change\n', encoding='utf-8')
            common_ref = 'refs/heads/feature'
            command, child_env, _meta = h.prepare_host_fence(
                ['/bin/sh', '-c', 'git add -- change.txt && git commit -m "fenced commit"'],
                provider='claude', workdir=str(linked.resolve()), env=env,
                allow_repo_write=True,
            )
            committed = subprocess.run(command, cwd=str(linked), env=child_env,
                                       capture_output=True, text=True, check=False)
            self.assertEqual(committed.returncode, 0, committed.stderr)
            main_oid = subprocess.check_output(
                ['git', '-C', str(main), 'rev-parse', '--verify', common_ref],
                text=True).strip()
            linked_oid = subprocess.check_output(
                ['git', '-C', str(linked), 'rev-parse', 'HEAD'], text=True).strip()
            self.assertEqual(main_oid, linked_oid)
            self.assertNotEqual(main_oid, '')
            subprocess.run(['git', '-C', str(main), 'cat-file', '-e',
                            main_oid + '^{commit}'], check=True)
            contents = subprocess.check_output(
                ['git', '-C', str(main), 'show', main_oid + ':change.txt'])
            self.assertEqual(contents, b'change\n')


_READ_PROBE = r'''#!/usr/bin/python3
import json, sys
out = {}
# The fence inserts grok's `--sandbox off` after the executable.
for path in [arg for arg in sys.argv[1:] if arg.startswith('/')]:
    try:
        with open(path) as handle:
            handle.read()
        out[path] = 0
    except OSError as exc:
        out[path] = exc.errno
print(json.dumps(out))
'''


@unittest.skipUnless(
    h.platform.system() == 'Darwin' and h.kernel_sandbox.available(),
    'requires the live macOS Seatbelt host fence',
)
class GrokInstructionFenceLiveTests(unittest.TestCase):
    def test_global_instructions_are_unreadable_but_grok_config_is_not(self):
        with tempfile.TemporaryDirectory(prefix='omni_grok_home_', dir='/private/tmp') as grok_home, \
                tempfile.TemporaryDirectory(prefix='omni_grok_outside_', dir='/private/tmp') as outside, \
                tempfile.TemporaryDirectory(prefix='omni_grok_cwd_', dir='/private/tmp') as workdir:
            def put(path, text='synthetic fixture'):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, 'w', encoding='utf-8') as handle:
                    handle.write(text)
                return path
            odd_case = put(os.path.join(grok_home, 'aGeNtS.md'))
            rule = put(os.path.join(grok_home, 'rules', 'fixture.md'))
            config = put(os.path.join(grok_home, 'config.toml'), '# fixture\n')
            target = put(os.path.join(outside, 'target.md'))
            control = put(os.path.join(outside, 'control.md'))
            linked = os.path.join(grok_home, 'CLAUDE.md')
            os.symlink(target, linked)
            denied = [odd_case, os.path.join(grok_home, 'AGENTS.md'), rule, linked, target]
            allowed = [config, control]
            # Controls: every fixture is readable outside the fence.
            for path in denied + allowed:
                with open(path, encoding='utf-8') as handle:
                    self.assertTrue(handle.read())
            probe = put(os.path.join(workdir, 'read-probe'), _READ_PROBE)
            os.chmod(probe, 0o700)
            argv, child_env, meta = h.prepare_host_fence(
                [probe, *denied, *allowed],
                provider='grok', workdir=workdir, read_only=True,
                env={**os.environ, 'GROK_HOME': grok_home})
            self.assertEqual(meta['fence'], 'seatbelt-fs')
            result = subprocess.run(argv, cwd=workdir, env=child_env,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            observed = json.loads(result.stdout)
            for path in denied:
                self.assertEqual(observed[path], errno.EPERM, (path, observed))
            for path in allowed:
                self.assertEqual(observed[path], 0, (path, observed))


@unittest.skipUnless(
    h.platform.system() == 'Darwin' and h.kernel_sandbox.available(),
    'requires the live macOS Seatbelt host fence',
)
class GrokInstructionFenceReviewTests(unittest.TestCase):
    """Review findings F6-F9 against the real Seatbelt wrap."""

    def setUp(self):
        self.stack = __import__('contextlib').ExitStack()
        self.addCleanup(self.stack.close)
        mk = lambda prefix: os.path.realpath(self.stack.enter_context(
            tempfile.TemporaryDirectory(prefix=prefix, dir='/private/tmp')))
        self.grok_home = mk('omni_grok_home_')
        self.outside = mk('omni_grok_outside_')
        self.workdir = mk('omni_grok_cwd_')
        self.probe = self.put(os.path.join(self.workdir, 'read-probe'), _READ_PROBE)
        os.chmod(self.probe, 0o700)

    def put(self, path, text='synthetic fixture'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write(text)
        return path

    def prepare(self, grok_home=None, **kw):
        env = {**os.environ, 'GROK_HOME': grok_home or self.grok_home}
        return h.prepare_host_fence([self.probe], provider='grok', workdir=self.workdir,
                                    read_only=True, env=env, **kw)

    def read(self, prepared, *paths):
        argv, child_env, _meta = prepared
        result = subprocess.run([*argv, *paths], cwd=self.workdir, env=child_env,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_f6_rules_symlink_chain_and_its_final_target_are_unreadable(self):
        target = self.put(os.path.join(self.outside, 'topic.md'))
        link = os.path.join(self.outside, 'link')
        os.symlink(target, link)
        os.makedirs(os.path.join(self.grok_home, 'rules'))
        via_rules = os.path.join(self.grok_home, 'rules', 'topic.md')
        os.symlink(link, via_rules)
        observed = self.read(self.prepare(), via_rules, target)
        self.assertEqual(observed[via_rules], errno.EPERM, observed)
        self.assertEqual(observed[target], errno.EPERM, observed)

    def test_u1_intermediate_hop_does_not_open_the_content(self):
        # rules/topic.md -> outside/link -> outside/topic.md: the hop's own
        # name is not a deny row, but opening it lands on the denied target.
        target = self.put(os.path.join(self.outside, 'topic.md'))
        link = os.path.join(self.outside, 'link')
        os.symlink(target, link)
        os.makedirs(os.path.join(self.grok_home, 'rules'))
        os.symlink(link, os.path.join(self.grok_home, 'rules', 'topic.md'))
        prepared = self.prepare()
        observed = self.read(prepared, link, target, os.path.join(self.grok_home, 'rules', 'topic.md'))
        for path in (link, target, os.path.join(self.grok_home, 'rules', 'topic.md')):
            self.assertEqual(observed[path], errno.EPERM, (path, observed))

    def test_f7_late_case_variant_and_retargeted_chain_stay_unreadable_by_name(self):
        first = self.put(os.path.join(self.outside, 'first.md'))
        second = self.put(os.path.join(self.outside, 'second.md'))
        mid = os.path.join(self.outside, 'mid')
        os.symlink(first, mid)
        linked = os.path.join(self.grok_home, 'CLAUDE.md')
        os.symlink(mid, linked)
        prepared = self.prepare()
        # After policy generation: a new case variant, and the chain retargeted.
        late = self.put(os.path.join(self.grok_home, 'aGeNtS.md'))
        os.remove(mid)
        os.symlink(second, mid)
        observed = self.read(prepared, late, os.path.join(self.grok_home, 'AGENTS.md'), linked)
        for path in (late, os.path.join(self.grok_home, 'AGENTS.md'), linked):
            self.assertEqual(observed[path], errno.EPERM, (path, observed))

    def test_f8_hard_linked_instruction_file_is_refused(self):
        original = self.put(os.path.join(self.grok_home, 'AGENTS.md'))
        os.link(original, os.path.join(self.outside, 'alias.md'))
        with self.assertRaises(h.HostFenceError) as caught:
            self.prepare()
        self.assertEqual(caught.exception.envelope['reason'], 'fence_path_rejected')

    def test_f9_config_dir_must_be_the_child_grok_home(self):
        other = os.path.realpath(self.stack.enter_context(
            tempfile.TemporaryDirectory(prefix='omni_grok_home_b_', dir='/private/tmp')))
        self.put(os.path.join(other, 'AGENTS.md'))
        with self.assertRaises(h.HostFenceError):
            self.prepare(grok_home=other, config_dir=self.grok_home)
        # Same dir both ways is fine; config_dir alone also pins the child.
        _argv, env, _meta = self.prepare(config_dir=self.grok_home)
        self.assertEqual(env['GROK_HOME'], self.grok_home)
        argv, env, meta = h.prepare_host_fence(
            [self.probe], provider='grok', workdir=self.workdir, read_only=True,
            env={k: v for k, v in os.environ.items() if k != 'GROK_HOME'},
            config_dir=self.grok_home)
        self.assertEqual(env['GROK_HOME'], self.grok_home)


class ApiSettingsExposureTests(unittest.TestCase):
    """The pure containment judgement over a composed write set."""

    def setUp(self):
        self.root = os.path.realpath(tempfile.mkdtemp(prefix='omni_apiset_x_'))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.base = os.path.join(self.root, 'a', 'b', 'api-settings')

    def test_subpath_root_at_or_above_the_base_is_exposure(self):
        for root in (self.base, os.path.dirname(self.base), self.root, '/'):
            with self.subTest(root=root):
                self.assertTrue(h.settings_base_exposed(self.base, [('subpath', root)]))

    def test_unrelated_or_deeper_roots_are_not_exposure(self):
        for root in (os.path.join(self.root, 'other'), os.path.join(self.base, 'run')):
            with self.subTest(root=root):
                self.assertIsNone(h.settings_base_exposed(self.base, [('subpath', root)]))

    def test_literal_root_equal_to_an_ancestor_is_exposure(self):
        self.assertTrue(h.settings_base_exposed(
            self.base, [('literal', os.path.join(self.root, 'a'))]))
        self.assertIsNone(h.settings_base_exposed(
            self.base, [('literal', os.path.join(self.root, 'a', 'file'))]))

    def test_other_case_spelling_of_an_ancestor_is_exposure(self):
        # realpath keeps the caller's case; Seatbelt rules match any case.
        ancestor = os.path.join(self.root, 'a')
        other = os.path.join(self.root, 'A')
        self.assertTrue(h.settings_base_exposed(self.base, [('subpath', other)]))
        self.assertTrue(h.settings_base_exposed(self.base, [('literal', other)]))
        self.assertTrue(h.settings_base_exposed(self.base.upper(), [('subpath', ancestor)]))
        self.assertIsNone(h.settings_base_exposed(
            self.base, [('subpath', os.path.join(self.root, 'B'))]))
        with self.assertRaises(h.HostFenceError):
            h._post_write_overlaps_repo([('subpath', other)], [os.path.join(ancestor, 'b')])
        h._post_write_overlaps_repo([('subpath', os.path.join(self.root, 'B'))],
                                    [os.path.join(ancestor, 'b')])

    def test_other_normalization_spelling_of_an_ancestor_is_exposure(self):
        import unicodedata
        nfc = os.path.join(self.root, unicodedata.normalize('NFC', 'caf\u00e9'))
        os.makedirs(os.path.join(nfc, 'api'))
        base = os.path.join(nfc, 'api')
        nfd = os.path.join(self.root, unicodedata.normalize('NFD', 'caf\u00e9'))
        self.assertTrue(h.settings_base_exposed(base, [('subpath', nfd)]))
        self.assertTrue(h.settings_base_exposed(base, [('literal', nfd)]))
        with self.assertRaises(h.HostFenceError):
            h._post_write_overlaps_repo([('subpath', nfd)], [base])
        self.assertTrue(h._reopens(('subpath', nfd), [('literal', os.path.join(nfc, 'AGENTS.md'))]))

    def test_any_case_containment_is_only_case_insensitive(self):
        inside = h.kernel_sandbox._path_inside_any_case
        self.assertTrue(inside(os.path.join(self.root, 'A', 'x'), os.path.join(self.root, 'a')))
        self.assertTrue(inside(os.path.join(self.root, 'a'), os.path.join(self.root, 'A')))
        self.assertFalse(inside(os.path.join(self.root, 'ab'), os.path.join(self.root, 'A')))
        self.assertFalse(inside(self.root, os.path.join(self.root, 'a')))

    def test_unknown_row_kinds_cannot_be_judged_and_count_as_exposure(self):
        self.assertTrue(h.settings_base_exposed(self.base, [('regex', '^/x')]))

    def test_alias_of_an_ancestor_is_exposure(self):
        os.makedirs(os.path.join(self.root, 'a', 'b'))
        alias = os.path.join(self.root, 'alias')
        os.symlink(os.path.join(self.root, 'a'), alias)
        self.assertTrue(h.settings_base_exposed(self.base, [('subpath', alias)]))
        self.assertTrue(h.settings_base_exposed(
            os.path.join(alias, 'b', 'api-settings'), [('subpath', os.path.join(self.root, 'a'))]))

    def test_repo_root_candidates_name_toplevel_and_git_dirs_and_fail_empty(self):
        repo = os.path.join(self.root, 'repo')
        os.makedirs(os.path.join(repo, 'src'))
        subprocess.run(['git', '-C', repo, 'init', '-q'], check=True)
        got = h.repo_root_candidates(os.path.join(repo, 'src'))
        self.assertEqual(got[0], os.path.realpath(repo))
        self.assertIn(os.path.join(os.path.realpath(repo), '.git'), got)
        self.assertEqual(h.repo_root_candidates(os.path.join(self.root, 'missing')), ())
        self.assertEqual(h.repo_root_candidates(self.root), ())


class _SettingsFenceFixture(unittest.TestCase):
    """Shared tree and helpers; API accounts are configured only when asked."""

    configure_api = False

    def setUp(self):
        self.tmpdir = os.environ.get('TMPDIR') or '/private/tmp'
        self.root = os.path.realpath(tempfile.mkdtemp(prefix='omni_apiset_', dir='/private/var/tmp'))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.home = os.path.join(self.root, 'home')
        self.work = os.path.join(self.root, 'work')
        self.state = os.path.join(self.root, 'state')
        for path in (self.home, self.work, self.state):
            os.mkdir(path)
        self.state_base = os.path.join(self.state, 'api-settings')
        self.home_base = os.path.join(self.home, '.cache', 'omni-llm', 'api-settings')
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.dict(os.environ, {'OMNI_LLM_STATE_DIR': self.state}))
        # The temp dir is a write root of every fence; the test tree is not under it.
        os.environ['TMPDIR'] = self.tmpdir
        stack.enter_context(mock.patch.object(h.kernel_sandbox, 'protected_home', return_value=self.home))
        # Declare the API-kind account the way the profile parser reads it.
        stack.enter_context(mock.patch.dict(os.environ))
        for name in ('OMNI_LLM_API_PROFILES_CLAUDE', 'CLAUDE_CONFIG_DIR_API'):
            os.environ.pop(name, None)
        if self.configure_api:
            os.environ['OMNI_LLM_API_PROFILES_CLAUDE'] = 'api'
            os.environ['CLAUDE_CONFIG_DIR_API'] = os.path.join(self.root, 'api-home')
        self.env = {'PATH': '/usr/bin:/bin', 'HOME': self.home}

    def fenced(self, script, workdir=None, provider='claude', **kw):
        return h.prepare_host_fence(['/bin/sh', '-c', script], provider=provider,
                                    workdir=workdir or self.work, env=dict(self.env), **kw)

    def run_fenced(self, script, **kw):
        argv, env, _meta = self.fenced(script, **kw)
        return subprocess.run(argv, cwd=kw.get('workdir') or self.work, env=env,
                              capture_output=True, text=True, timeout=20)

    def mocked_os_layer(self):
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.object(h.kernel_sandbox, 'verify_fence', return_value=None))
        return stack.enter_context(mock.patch.object(
            h.kernel_sandbox, 'wrap_argv',
            side_effect=lambda argv, *a, **kw: [h.kernel_sandbox.SANDBOX_EXEC, '-p', 'x', *argv]))

@unittest.skipUnless(
    h.platform.system() == 'Darwin' and h.kernel_sandbox.available(),
    'requires the live macOS Seatbelt host fence',
)
class NoApiAccountsFenceTests(_SettingsFenceFixture):
    """Without API accounts: base deny-write rows still emitted, no exposure refusal."""

    def assertBaseRows(self, wrap):
        rows = wrap.call_args.kwargs.get('extra_deny_write_final') or ()
        for base in (self.state_base, self.home_base):
            self.assertIn(('subpath', os.path.realpath(base)), rows)

    def assertNoBaseRows(self, wrap):
        rows = wrap.call_args.kwargs.get('extra_deny_write_final') or ()
        for base in (self.state_base, self.home_base):
            self.assertNotIn(('subpath', os.path.realpath(base)), rows)
            self.assertNotIn(('subpath', base), rows)

    def test_final_deny_rows_present_for_any_provider_without_api_env(self):
        self.assertNotIn('OMNI_LLM_API_PROFILES_CLAUDE', os.environ)
        wrap = self.mocked_os_layer()
        for provider in ('claude', 'pi'):
            for read_only in (False, True):
                with self.subTest(provider=provider, read_only=read_only):
                    self.fenced('true', provider=provider, read_only=read_only)
                    self.assertBaseRows(wrap)

    def test_workdir_containing_the_state_dir_or_home_is_not_refused_but_still_denied(self):
        wrap = self.mocked_os_layer()
        for workdir in (self.root, self.state, self.home):
            with self.subTest(workdir=workdir):
                self.fenced('true', workdir=workdir)
                self.assertBaseRows(wrap)

    def test_state_dir_under_the_temp_root_is_not_refused_but_still_denied(self):
        wrap = self.mocked_os_layer()
        state = os.path.realpath(tempfile.mkdtemp(prefix='omni_apiset_tmpstate_'))
        self.addCleanup(shutil.rmtree, state, True)
        os.environ['OMNI_LLM_STATE_DIR'] = state
        self.fenced('true')
        rows = wrap.call_args.kwargs.get('extra_deny_write_final') or ()
        self.assertIn(('subpath', os.path.join(state, 'api-settings')), rows)

    def test_an_unreadable_account_home_skips_the_rows_silently(self):
        wrap = self.mocked_os_layer()
        with mock.patch.object(h, 'api_settings_bases', side_effect=RuntimeError('no home')):
            self.fenced('true')
        self.assertNoBaseRows(wrap)

    def test_blank_declaration_counts_as_not_configured(self):
        self.mocked_os_layer()
        os.environ['OMNI_LLM_API_PROFILES_CLAUDE'] = '  '
        self.assertFalse(h.api_accounts_configured())
        self.fenced('true', workdir=self.root)


@unittest.skipUnless(
    h.platform.system() == 'Darwin' and h.kernel_sandbox.available(),
    'requires the live macOS Seatbelt host fence',
)
class ApiSettingsContainmentTests(_SettingsFenceFixture):
    """With API accounts configured: every fenced child gets the deny, none may reach the bases."""

    configure_api = True

    def test_configured_state_follows_the_declaration(self):
        self.assertTrue(h.api_accounts_configured())
        os.environ['OMNI_LLM_API_PROFILES_CLAUDE'] = ''
        self.assertFalse(h.api_accounts_configured())
        os.environ.pop('OMNI_LLM_API_PROFILES_CLAUDE')
        self.assertFalse(h.api_accounts_configured())
        os.environ['OMNI_LLM_API_PROFILES_CLAUDE'] = 'api,'
        self.assertTrue(h.api_accounts_configured(), 'malformed fails closed')

    def test_an_unreadable_account_home_still_fails_closed(self):
        wrap = self.mocked_os_layer()
        with mock.patch.object(h, 'api_settings_bases', side_effect=RuntimeError('no home')):
            with self.assertRaises(h.HostFenceError) as caught:
                self.fenced('true')
        self.assertEqual(caught.exception.envelope['reason'], 'fence_path_rejected')
        wrap.assert_not_called()

    def test_bases_follow_the_state_dir_and_the_account_home(self):
        self.assertEqual(h.api_settings_bases(), (self.state_base, self.home_base))

    def test_every_fenced_child_carries_final_denies_on_both_bases(self):
        wrap = self.mocked_os_layer()
        for provider in ('claude', 'pi'):
            for read_only in (False, True):
                with self.subTest(provider=provider, read_only=read_only):
                    self.fenced('true', provider=provider, read_only=read_only)
                    rows = wrap.call_args.kwargs['extra_deny_write_final']
                    for base in (self.state_base, self.home_base):
                        self.assertIn(('subpath', os.path.realpath(base)), rows)

    def test_a_workdir_that_contains_a_base_or_its_ancestor_is_refused(self):
        wrap = self.mocked_os_layer()
        for workdir in (self.root, self.state, self.home):
            with self.subTest(workdir=workdir):
                with self.assertRaises(h.HostFenceError) as caught:
                    self.fenced('true', workdir=workdir)
                self.assertEqual(caught.exception.envelope['reason'], 'fence_path_rejected')
                self.assertIn('API settings area', caught.exception.envelope['detail'])
        wrap.assert_not_called()

    def test_repository_root_the_host_fence_adds_is_judged_before_wrapping(self):
        # Z1: the workdir (repo/src) alone does not reach repo/.omni, but the
        # repository root the host fence admits for it does. (Git reports a
        # relative common dir from a subdirectory, which the real resolver
        # refuses on its own; its result is stood in for here.)
        wrap = self.mocked_os_layer()
        repo = os.path.join(self.root, 'repo')
        os.makedirs(os.path.join(repo, 'src'))
        admitted = (h.kernel_sandbox.admit_path(repo, 'subpath', label='repo'),)
        os.environ['OMNI_LLM_STATE_DIR'] = os.path.join(repo, '.omni')
        with mock.patch.object(h, '_repo_write_roots', return_value=(admitted, (), (), (), ())):
            with self.assertRaises(h.HostFenceError) as caught:
                self.fenced('true', workdir=os.path.join(repo, 'src'))
            self.assertEqual(caught.exception.envelope['reason'], 'fence_path_rejected')
            wrap.assert_not_called()
            # Control: the same composition with the state dir elsewhere passes.
            os.environ['OMNI_LLM_STATE_DIR'] = self.state
            self.fenced('true', workdir=os.path.join(repo, 'src'))
            wrap.assert_called_once()

    def test_bare_fence_with_the_grandparent_writable_lets_the_parent_be_renamed(self):
        # Control for the refusal below: without the final check, a write root
        # containing the grandparent lets a child rename the parent away.
        grand = os.path.join(self.work, 'grand')
        parent = os.path.join(grand, 'parent')
        os.makedirs(os.path.join(parent, 'api-settings'))
        argv = h.kernel_sandbox.wrap_argv(
            ['/bin/sh', '-c', 'mv "%s" "%s.moved"' % (parent, parent)], self.work, False)
        res = subprocess.run(argv, cwd=self.work, capture_output=True, text=True, timeout=20)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertTrue(os.path.isdir(parent + '.moved'))

    def test_fence_whose_write_root_contains_the_grandparent_is_refused_before_spawn(self):
        grand = os.path.join(self.work, 'grand')
        parent = os.path.join(grand, 'parent')
        base = os.path.join(parent, 'api-settings')
        os.makedirs(base)
        with mock.patch.object(h, 'api_settings_bases', return_value=(base, self.home_base)):
            with self.assertRaises(h.HostFenceError) as caught:
                self.fenced('mv "%s" "%s.moved"' % (parent, parent))
        self.assertEqual(caught.exception.envelope['reason'], 'fence_path_rejected')
        self.assertTrue(os.path.isdir(base))
        self.assertFalse(os.path.exists(parent + '.moved'))

    def test_live_child_cannot_touch_either_base_or_the_state_dir_above_it(self):
        os.makedirs(self.state_base)
        Path(self.state_base, 'marker').write_text('keep', encoding='utf-8')
        for script in ('mv "%s" "%s.moved"' % (self.state, self.state),
                       'mv "%s" "%s.moved"' % (self.state_base, self.state_base),
                       'echo x >> "%s/marker"' % self.state_base,
                       'rm -f "%s/marker"' % self.state_base,
                       'mkdir -p "%s"' % self.home_base):
            with self.subTest(script=script):
                res = self.run_fenced(script)
                self.assertNotEqual(res.returncode, 0, script)
        self.assertEqual(Path(self.state_base, 'marker').read_text(encoding='utf-8'), 'keep')
        self.assertFalse(os.path.exists(self.home_base))
        control = self.run_fenced('echo ok > "%s/control"' % self.work)
        self.assertEqual(control.returncode, 0, control.stderr)


if __name__ == '__main__':
    unittest.main()
