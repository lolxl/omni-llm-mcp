"""Public-consumer regressions; all vendor, home and health state is synthetic."""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from providers import agent_lane, health, job_runner, jobs, kernel_sandbox


FAKE = r'''import json,os,pathlib,sys
p=pathlib.Path(sys.argv[0]).name
if p=='claude':sys.stdin.read()
a=sys.argv[1:]
def val(k):return a[a.index(k)+1] if k in a else None
k='CLAUDE_CONFIG_DIR' if p=='claude' else 'GROK_HOME'
h=os.environ.get(k) or os.path.join(os.environ['HOME'],'.'+p)
sid=val('--session-id');resume=val('-r') or val('--resume')
found=pathlib.Path(h,'fake-session-'+str(resume)).exists() if resume else None
if not resume:pathlib.Path(h,'fake-session-'+str(sid)).write_text('synthetic')
row={'provider':p,'argv':a,'home':h,'config_set':k in os.environ,
     'config':os.environ.get(k),'cwd':os.getcwd(),'session_id':sid,
     'resume_session':resume,'resume_found':found}
with open(os.environ['REGRESSION_CASE']+'/children.jsonl','a') as f:f.write(json.dumps(row)+'\n')
print(os.environ.get('FAKE_OUTPUT','{"type":"error","error":{"code":"synthetic-failure"}}'))
sys.exit(int(os.environ.get('FAKE_RC','1')))
'''
WRAP = r'''import os,sys
a=sys.argv[1:]
assert a[0]=='-p' and a[1].startswith('(version 1)')
a=a[2:]
while a and a[0]=='-D':a=a[2:]
os.execvpe(a[0],a,os.environ)
'''


def install_seams():
    """Mock OS enforcement only; retain host preparation/env rewrite/real run_cli."""
    case = Path(os.environ['REGRESSION_CASE'])
    patches = [
        mock.patch.object(kernel_sandbox, 'protected_home', return_value=str(case/'home')),
        mock.patch.object(kernel_sandbox, 'SANDBOX_EXEC', str(case/'bin'/'sandbox-exec')),
        mock.patch.object(kernel_sandbox, 'verify_fence', return_value=None),
        mock.patch.object(health, '_account_identity', return_value=None),
    ]
    real_cli = agent_lane.run_cli
    async def recorded_cli(argv, **kwargs):
        job_dir = os.environ.get('REGRESSION_JOB')
        pin = jobs._read_job(job_dir) if job_dir else None
        result = await real_cli(argv, **kwargs)
        row = {'returncode': result.get('returncode'), 'signal': result.get('signal'),
               'timed_out': result.get('timed_out'), 'pin_before_spawn': pin,
               'argv': argv, 'cwd': kwargs.get('cwd'),
               'claude_config': kwargs['env'].get('CLAUDE_CONFIG_DIR'),
               'grok_home': kwargs['env'].get('GROK_HOME')}
        with (case/'processes.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        return result
    real_attest = job_runner.read_successor_attest
    def recorded_attest(provider, sid, env, **kwargs):
        with (case/'attest.jsonl').open('a') as f:
            f.write(json.dumps({'provider': provider, 'session_id': sid,
                'claude_config': env.get('CLAUDE_CONFIG_DIR'),
                'grok_home': env.get('GROK_HOME')})+'\n')
        return real_attest(provider, sid, env, **kwargs)
    patches += [mock.patch.object(agent_lane, 'run_cli', side_effect=recorded_cli),
                mock.patch.object(job_runner, 'read_successor_attest', side_effect=recorded_attest)]
    for patch in patches:
        patch.start()
    return patches


class ProviderReviewRegressions(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='provider-review-')
        self.addCleanup(self.temp.cleanup)
        self.case = Path(self.temp.name).resolve()
        for name in ('home/.claude', 'home/.grok', 'a', 'b', 'grok-a', 'grok-b', 'bin', 'work'):
            (self.case/name).mkdir(parents=True, exist_ok=True)
        env = {'PATH': str(self.case/'bin')+os.pathsep+os.defpath,
               'HOME': str(self.case/'home'), 'TMPDIR': str(self.case),
               'OMNI_LLM_QUOTA_PROBE': '0', 'OMNI_LLM_NOTIFY_QUIET': '1',
               'OMNI_LLM_STATE_DIR': str(self.case/'state'),
               'REGRESSION_CASE': str(self.case),
               # The fenced fake vendor gets an allowlisted environment
               # (#138): its control variables are passed by name.
               'OMNI_LLM_CHILD_ENV_PASS': 'REGRESSION_CASE,FAKE_OUTPUT,FAKE_RC',
               'PROVIDER_REVIEW_ARTIFACT_DIR': os.environ.get('PROVIDER_REVIEW_ARTIFACT_DIR', ''),
               'OMNI_LLM_CLAUDE_BIN': str(self.case/'bin'/'claude')}
        # Registered account homes must be disjoint across providers.
        for key, prefix in (('CLAUDE_CONFIG_DIR', ''), ('GROK_HOME', 'grok-')):
            for label in ('a', 'b'):
                env[key+'_'+label.upper()] = str(self.case/(prefix+label))
        self.env = mock.patch.dict(os.environ, env, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        for name, body in (('claude', FAKE), ('grok', FAKE), ('sandbox-exec', WRAP)):
            path = self.case/'bin'/name
            path.write_text('#!'+sys.executable+'\n'+body, encoding='utf-8')
            path.chmod(0o700)
        seams = install_seams()
        self.addCleanup(lambda: [p.stop() for p in reversed(seams)])
        self.procs = []
        self.addCleanup(self.finish)
        self.real_popen = subprocess.Popen
        root = Path(__file__).resolve().parents[1]
        driver = self.case/'supervisor.py'
        driver.write_text(
            'import os,sys,time,pathlib\n'
            'sys.path.insert(0,'+repr(str(root))+')\n'
            'from tests.test_provider_review_regressions import install_seams\n'
            'install_seams()\nfrom providers import job_runner,jobs\n'
            'os.environ["REGRESSION_JOB"]=sys.argv[1]\n'
            'real=job_runner._execute\n'
            'async def delayed(job_dir,job):\n'
            '    pathlib.Path(job_dir,"runner.active").write_text("test gate")\n'
            '    deadline=time.monotonic()+15\n'
            '    while not pathlib.Path(job_dir,"release").exists():\n'
            '        if time.monotonic()>deadline:raise RuntimeError("release timeout")\n'
            '        time.sleep(.01)\n'
            '    return await real(job_dir,jobs._read_job(job_dir))\n'
            'job_runner._execute=delayed\n'
            'raise SystemExit(job_runner.main())\n', encoding='utf-8')
        self.driver = driver

    def finish(self):
        for proc in self.procs:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)
        # Optional durable evidence; normal test runs keep only temporary state.
        artifact = os.environ.get('PROVIDER_REVIEW_ARTIFACT_DIR')
        if artifact:
            import shutil
            dest = Path(artifact)/self._testMethodName
            shutil.copytree(self.case, dest, dirs_exist_ok=True)

    def supervisor(self, argv, **kwargs):
        if len(argv) == 4 and argv[1:3] == ['-m', 'providers.job_runner']:
            argv = [sys.executable, '-B', str(self.driver), argv[-1]]
            proc = self.real_popen(argv, **kwargs)
            self.procs.append(proc)
            return proc
        return self.real_popen(argv, **kwargs)

    def start(self, provider, **kwargs):
        with mock.patch.object(jobs.subprocess, 'Popen', side_effect=self.supervisor):
            result = jobs.start('synthetic task', provider+'/synthetic-model',
                                str(self.case/'work'), timeout_s=8, **kwargs)
        self.assertNotIn('refused', result)
        return result

    def release(self, job):
        (Path(job['job_dir'])/'release').touch()
        self.assertEqual(self.procs[-1].wait(timeout=15), 0)
        result = jobs._read_result(job['job_dir'])
        self.assertFalse(result['ok'])
        self.assertEqual(result['reason'], 'runtime_error')
        self.assertEqual(self.rows('processes')[-1]['returncode'], 1)
        return jobs._read_job(job['job_dir'])

    def rows(self, name='children'):
        return [json.loads(x) for x in (self.case/(name+'.jsonl')).read_text().splitlines()]

    def resume(self, source):
        with mock.patch.object(jobs.subprocess, 'Popen', side_effect=self.supervisor):
            result = jobs.resume(source['job_dir'], 'synthetic continuation', allow_drift=True)
        self.assertNotIn('refused', result)
        return result

    def selected_pin(self, provider, cooling):
        os.environ['OMNI_LLM_PROFILES_'+provider.upper()] = 'a,b'
        first = self.start(provider)
        before = jobs._read_job(first['job_dir'])
        self.assertEqual(before['account_profile'], 'a')
        if cooling:
            health.flag(provider+'/synthetic-model', 'a')
        after = self.release(first)
        selected = 'b' if cooling else 'a'
        row = self.rows()[0]
        self.assertEqual(row['home'], str(self.case/(('grok-' if provider == 'grok' else '')+selected)))
        self.assertEqual(row['session_id'], before['session_id'])
        self.assertEqual(len(self.rows()), 1)  # No replay after terminal failure.
        self.assertEqual(after['account_profile'], selected)
        self.assertEqual(after['account_home'], row['home'])
        pin = self.rows('processes')[0]['pin_before_spawn']
        self.assertEqual(pin['account_profile'], selected)
        self.assertEqual(pin['account_home'], row['home'])
        key = 'claude_config' if provider == 'claude' else 'grok_home'
        self.assertEqual(self.rows('attest')[0][key], row['home'])
        # Resume stays on the selected account even if it is now cooling.
        health.flag(provider+'/synthetic-model', selected)
        resumed = self.resume(first)
        rb = self.release(resumed)
        self.assertEqual(rb['account_profile'], selected)
        self.assertEqual(self.rows()[1]['home'], row['home'])
        self.assertTrue(self.rows()[1]['resume_found'])
        self.assertEqual(self.rows()[1]['resume_session'], before['session_id'])

    def test_selected_pin_valid_controls(self):
        self.selected_pin('claude', False)

    def test_selected_pin_claude_cooling(self):
        self.selected_pin('claude', True)

    def test_selected_pin_grok_cooling(self):
        self.selected_pin('grok', True)

    def output_cases(self, job_lane, samples):
        for provider in ('claude', 'grok'):
            for managed in (False, True):
                for sample, expected in samples:
                    with self.subTest(provider=provider, managed=managed, sample=sample):
                        health.clear_all()
                        ring = 'OMNI_LLM_PROFILES_'+provider.upper()
                        if managed:
                            os.environ[ring] = 'a,b'
                        else:
                            os.environ.pop(ring, None)
                        os.environ['FAKE_OUTPUT'] = sample
                        os.environ['FAKE_RC'] = '0'
                        result = asyncio.run(agent_lane.run_agent('synthetic task',
                            provider+'/synthetic-model', str(self.case/'work'),
                            timeout_s=5, job_lane=job_lane))
                        self.assertEqual(self.rows('processes')[-1]['returncode'], 0)
                        self.assertEqual(result.get('reason'), expected)
                        self.assertEqual(result['ok'], expected is None)
                        self.assertEqual(health.seat_kind(provider, 'a' if managed else 'default'),
                                         'cooling' if expected in ('rate_limit', 'auth_error') else 'healthy')
                        if expected is None:
                            self.assertEqual(result['text'], sample)

    def test_plain_valid_controls(self):
        self.output_cases(False, [('ordinary successful answer', None),
            ('{"type":"result","is_error":false,"result":"usage_limit_reached"}', None)])

    def test_plain_error_examples_are_answers(self):
        self.output_cases(False, [
            ('Example:\n{"type":"error","error":{"code":"usage_limit_reached"}}', None),
            ('{"type":"error","error":{"code":"authentication_failed"}}', None),
            ('{"type":"turn.failed"}', None)])

    def test_structured_real_errors_remain_failures(self):
        self.output_cases(True, [
            ('{"type":"error","error":{"code":"usage_limit_reached"}}', 'rate_limit'),
            ('{"type":"error","error":{"code":"authentication_failed"}}', 'auth_error')])

    def test_nonzero_plain_real_errors_control(self):
        for provider in ('claude', 'grok'):
            with self.subTest(provider=provider):
                os.environ['FAKE_OUTPUT'] = '{"type":"error","error":{"code":"usage_limit_reached"}}'
                result = asyncio.run(agent_lane.run_agent('synthetic task',
                    provider+'/synthetic-model', str(self.case/'work'), timeout_s=5))
                self.assertFalse(result['ok'])
                self.assertEqual(result['reason'], 'rate_limit')
                self.assertEqual(self.rows('processes')[-1]['returncode'], 1)

    def test_sync_default_unset_control(self):
        result = asyncio.run(agent_lane.run_agent('synthetic task',
            'claude/synthetic-model', str(self.case/'work'), timeout_s=5))
        self.assertFalse(result['ok'])
        self.assertFalse(self.rows()[0]['config_set'])

    def test_direct_default_home_pin_is_explicit_control(self):
        home = str(self.case/'home/.claude')
        result = asyncio.run(agent_lane.run_agent('synthetic task',
            'claude/synthetic-model', str(self.case/'work'), timeout_s=5,
            account_home=home))
        self.assertFalse(result['ok'])
        self.assertTrue(self.rows()[0]['config_set'])
        self.assertEqual(self.rows()[0]['config'], home)

    def test_unset_provenance_rejects_other_home_before_agent_spawn(self):
        for provider in ('claude', 'grok'):
            key = 'CLAUDE_CONFIG_DIR' if provider == 'claude' else 'GROK_HOME'
            for profile in ('b', 'default'):
                with self.subTest(provider=provider, profile=profile):
                    if profile == 'default':
                        os.environ[key] = str(self.case/(('grok-' if provider == 'grok' else '')+'b'))
                    before = len(self.rows()) if (self.case/'children.jsonl').exists() else 0
                    result = asyncio.run(agent_lane.run_agent('synthetic task',
                        provider+'/synthetic-model', str(self.case/'work'), timeout_s=5,
                        account_profile=profile, account_home=str(self.case/(('grok-' if provider == 'grok' else '')+'b')),
                        account_home_explicit=False))
                    self.assertFalse(result['ok'])
                    self.assertEqual(result['reason'], 'account_home_drift')
                    self.assertIn('unset', result['detail'])
                    after = len(self.rows()) if (self.case/'children.jsonl').exists() else 0
                    self.assertEqual(after, before)
                    os.environ.pop(key, None)

    def test_unset_provenance_rejects_other_home_before_job_spawn(self):
        for provider in ('claude', 'grok'):
            with self.subTest(provider=provider):
                with mock.patch.object(jobs.subprocess, 'Popen', side_effect=self.supervisor):
                    result = jobs.start('synthetic task', provider+'/synthetic-model',
                        str(self.case/'work'), timeout_s=5, account_profile='b',
                        account_home=str(self.case/(('grok-' if provider == 'grok' else '')+'b')), account_home_explicit=False)
                self.assertEqual(result.get('refused', {}).get('reason'), 'account_home_drift')
                self.assertIn('unset', result['refused']['detail'])
                self.assertEqual(len(self.procs), 0)

    def default_provenance(self, explicit=False, managed=False, legacy=False):
        home = str(self.case/'home/.claude')
        if explicit:
            os.environ['CLAUDE_CONFIG_DIR'] = home
        if managed:
            os.environ['OMNI_LLM_PROFILES_CLAUDE'] = 'default,b'
        first = self.start('claude')
        if legacy:
            data = jobs._read_job(first['job_dir'])
            data.pop('account_home_explicit', None)
            jobs._write_job(first['job_dir'], data)
        after = self.release(first)
        expected = explicit or legacy
        self.assertEqual(self.rows()[0]['config_set'], expected)
        self.assertEqual(self.rows()[0]['home'], home)
        self.assertEqual(self.rows('attest')[0]['claude_config'], home if expected else None)
        # Current caller environment must not rewrite persisted provenance.
        if expected:
            os.environ.pop('CLAUDE_CONFIG_DIR', None)
        else:
            os.environ['CLAUDE_CONFIG_DIR'] = home
        resumed = self.resume(first)
        rb = self.release(resumed)
        self.assertEqual(rb['account_home'], after['account_home'])
        self.assertEqual(self.rows()[1]['config_set'], expected)
        self.assertTrue(self.rows()[1]['resume_found'])

    def test_default_unset_provenance(self):
        self.default_provenance()

    def test_managed_default_unset_provenance(self):
        self.default_provenance(managed=True)

    def test_default_explicit_provenance_control(self):
        self.default_provenance(explicit=True)

    def test_legacy_default_keeps_explicit_pin_control(self):
        self.default_provenance(legacy=True)

    def test_named_provenance_control(self):
        first = self.start('claude', account_profile='b')
        self.release(first)
        self.assertTrue(self.rows()[0]['config_set'])
        self.assertEqual(self.rows()[0]['config'], str(self.case/'b'))

    def consult(self, account_profile='default', account_pinned=False, direct_home=None):
        from providers import claude
        original = claude.run_cli
        original_rewrite = kernel_sandbox.agent_child_env
        async def observed(argv, **kwargs):
            result = await original(argv, **kwargs)
            env = kwargs['env']
            with (self.case/'consult-processes.jsonl').open('a') as f:
                f.write(json.dumps({'argv': argv, 'returncode': result.get('returncode'),
                    'signal': result.get('signal'), 'timed_out': result.get('timed_out'),
                    'cwd': kwargs['cwd'], 'config_set': 'CLAUDE_CONFIG_DIR' in env,
                    'config': env.get('CLAUDE_CONFIG_DIR'),
                    'api_key_set': 'ANTHROPIC_API_KEY' in env,
                    'sibling_keys': [k for k in env if k.startswith('CLAUDE_CONFIG_DIR_')]})+'\n')
            return result
        def rewrite(base=None, tmpdir=None, keep_config_dir=False):
            result = original_rewrite(base, tmpdir, keep_config_dir)
            with (self.case/'consult-rewrite.jsonl').open('a') as f:
                f.write(json.dumps({'before_set': 'CLAUDE_CONFIG_DIR' in (base or {}),
                    'after_set': 'CLAUDE_CONFIG_DIR' in result,
                    'keep_config_dir': keep_config_dir})+'\n')
            return result
        with mock.patch.object(claude, 'run_cli', side_effect=observed), \
             mock.patch.object(kernel_sandbox, 'agent_child_env', side_effect=rewrite):
            if direct_home:
                call = claude._run_claude_once('synthetic task', 'synthetic-model', 5,
                    account_profile=account_profile, account_home=direct_home)
            else:
                call = claude.run_claude_subprocess('synthetic task', 'synthetic-model', 5,
                    account_profile=account_profile, account_pinned=account_pinned)
            return asyncio.run(call)

    def consult_success(self, expected_explicit, account_profile='default', account_pinned=False,
                        direct_home=None):
        os.environ['FAKE_OUTPUT'] = json.dumps({'type': 'result', 'subtype': 'success',
            'is_error': False, 'result': 'synthetic consult answer'})
        os.environ['FAKE_RC'] = '0'
        result = self.consult(account_profile, account_pinned, direct_home)
        self.assertTrue(result['ok'], result)
        self.assertEqual(result['text'], 'synthetic consult answer')
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(self.rows()[0]['config_set'], expected_explicit)
        process = self.rows('consult-processes')[0]
        self.assertEqual(process['returncode'], 0)
        self.assertIsNone(process['signal'])
        self.assertFalse(process['timed_out'])
        self.assertEqual(process['config_set'], expected_explicit)
        self.assertIn('--output-format', self.rows()[0]['argv'])
        argv = self.rows()[0]['argv']
        self.assertEqual(argv[argv.index('--output-format')+1], 'json')
        self.assertEqual(self.rows()[0]['cwd'], os.path.realpath(process['cwd']))

    def test_consult_no_ring_default_unset_control(self):
        self.consult_success(False)

    def test_consult_managed_default_unset(self):
        os.environ['OMNI_LLM_PROFILES_CLAUDE'] = 'default,b'
        os.environ['ANTHROPIC_API_KEY'] = 'synthetic-key-never-used'
        self.consult_success(False)
        self.assertFalse(self.rows('consult-processes')[0]['api_key_set'])
        self.assertEqual(self.rows('consult-processes')[0]['sibling_keys'], [])
        self.assertFalse(self.rows('consult-rewrite')[0]['after_set'])

    def test_consult_pinned_default_unset(self):
        os.environ['OMNI_LLM_PROFILES_CLAUDE'] = 'default,b'
        self.consult_success(False, account_pinned=True)

    def test_consult_explicit_default_control(self):
        os.environ['OMNI_LLM_PROFILES_CLAUDE'] = 'default,b'
        home = str(self.case/'home/.claude')
        os.environ['CLAUDE_CONFIG_DIR'] = home
        self.consult_success(True)
        self.assertEqual(self.rows()[0]['config'], home)

    def test_consult_named_control(self):
        self.consult_success(True, account_profile='b')
        self.assertEqual(self.rows()[0]['config'], str(self.case/'b'))

    def test_consult_direct_explicit_home_control(self):
        home = str(self.case/'home/.claude')
        self.consult_success(True, direct_home=home)
        self.assertEqual(self.rows()[0]['config'], home)

    def test_consult_keeps_selected_provenance(self):
        from providers import provider_accounts
        os.environ['OMNI_LLM_PROFILES_CLAUDE'] = 'default,b'
        original = provider_accounts.AccountSelection.next
        def changed_environment(selection):
            account = original(selection)
            os.environ['CLAUDE_CONFIG_DIR'] = str(self.case/'home/.claude')
            return account
        with mock.patch.object(provider_accounts.AccountSelection, 'next', changed_environment):
            self.consult_success(False)

    def test_consult_invalid_selected_provenance_stops_before_spawn(self):
        from providers import provider_accounts
        for profile, provenance in (('b', False), ('default', 'false')):
            with self.subTest(profile=profile, provenance=provenance):
                health.clear_all()
                os.environ['OMNI_LLM_PROFILES_CLAUDE'] = 'default,b'
                os.environ['FAKE_OUTPUT'] = json.dumps({'type': 'result', 'is_error': False,
                                                      'result': 'synthetic success'})
                os.environ['FAKE_RC'] = '0'
                before = len(self.rows()) if (self.case/'children.jsonl').exists() else 0
                with mock.patch.object(provider_accounts, 'home_explicit', return_value=provenance):
                    result = self.consult(account_profile=profile)
                self.assertFalse(result['ok'], result)
                self.assertEqual(result['reason'], 'account_home_drift')
                after = len(self.rows()) if (self.case/'children.jsonl').exists() else 0
                self.assertEqual(after, before)

    def test_consult_structured_failures_do_not_replay_control(self):
        os.environ['OMNI_LLM_PROFILES_CLAUDE'] = 'default,b'
        for code, reason in (('usage_limit_reached', 'rate_limit'),
                             ('authentication_failed', 'auth_error')):
            for rc in (0, 1):
                with self.subTest(code=code, rc=rc):
                    health.clear_all()
                    os.environ['FAKE_OUTPUT'] = json.dumps({'type': 'result',
                        'subtype': 'error_during_execution', 'is_error': True,
                        'error': {'code': code}})
                    os.environ['FAKE_RC'] = str(rc)
                    before = len(self.rows()) if (self.case/'children.jsonl').exists() else 0
                    result = self.consult()
                    self.assertFalse(result['ok'], result)
                    self.assertEqual(result['reason'], reason)
                    self.assertIs(result['account_retry_safe'], False)
                    self.assertEqual(len(self.rows()), before+1)
                    self.assertEqual(self.rows()[-1]['home'], str(self.case/'home/.claude'))
                    self.assertEqual(health.seat_kind('claude', 'default'), 'cooling')
                    self.assertEqual(self.rows('consult-processes')[-1]['returncode'], rc)


if __name__ == '__main__':
    unittest.main()
