"""Exact-byte AGY file carrier controls; no AGY process or network is used."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import unittest
from unittest import mock
from providers import agy, config
from providers.cli_run import run_cli
from providers.errors import is_local_failure


def payload(size=605743):
    fixed = b'HEAD-SYNTHETIC\r\n' + b'a' * 210000 + b'\xc3\xa9\r\n' + b'MIDDLE-SYNTHETIC\r\n' + b'TAIL-SYNTHETIC\r\n'
    return (fixed[:-len(b'TAIL-SYNTHETIC\r\n')] + b'z' * (size-len(fixed)) + b'TAIL-SYNTHETIC\r\n').decode('utf-8')


def input_file(argv):
    match = re.search(r'^INPUT_FILE: (.+)$', argv[argv.index('-p')+1], re.M)
    return Path(json.loads(match[1])) if match else None


class AgyPromptFileTests(unittest.TestCase):
    def run_case(self, text, runner=None, prepare=None, effort=None, profile='default', deny=None):
        seen={}
        def fence(argv, **kwargs):
            seen['argv']=list(argv);seen['cwd']=kwargs['workdir'];seen['readonly']=kwargs['read_only']
            path=input_file(argv)
            if path:
                seen['data']=path.read_bytes();seen['mode']=stat.S_IMODE(path.stat().st_mode)
                seen['dir_mode']=stat.S_IMODE(Path(kwargs['workdir']).stat().st_mode)
            return (prepare(argv,**kwargs) if prepare else (argv,kwargs['env'],{}))
        async def clipped_vendor(argv, **kwargs):
            path=input_file(argv)
            data=path.read_bytes() if path else argv[argv.index('-p')+1].encode()[:191602]
            seen['observed']=data
            return {'returncode':0,'stdout':json.dumps({'status':'SUCCESS','response':'tail-read' if data.endswith(b'TAIL-SYNTHETIC\r\n') else 'clipped'}),'stderr':'','timed_out':False}
        with mock.patch.object(agy,'find_agy',return_value='/synthetic/agy'), \
             mock.patch.object(agy,'prepare_host_fence',side_effect=fence), \
             mock.patch.object(agy,'run_cli',side_effect=runner or clipped_vendor), \
             mock.patch.object(agy,'refuse_payload',return_value=deny), \
             mock.patch('providers.model_catalog.agy_effort_tokens',return_value=frozenset(['medium'])):
            result=asyncio.run(agy.run_agy_subprocess(text,'synthetic-model',30,profile,effort=effort))
        return result,seen

    def test_single_call_retains_605743_byte_payload_beyond_clipped_argv_preview(self):
        text=payload();result,seen=self.run_case(text)
        self.assertEqual(result.get('text'),'tail-read',result)
        self.assertEqual(seen['observed'],text.encode('utf-8'))
        self.assertLess(len(seen['argv'][seen['argv'].index('-p')+1].encode()),2048)
        self.assertFalse(Path(seen['cwd']).exists())

    def test_stage_is_exact_readonly_and_exists_before_same_workdir_fence(self):
        text=payload();result,seen=self.run_case(text,effort='medium')
        self.assertTrue(result['ok'],result);self.assertEqual(seen.get('data'),text.encode())
        self.assertEqual(seen.get('mode'),0o400);self.assertEqual(seen.get('dir_mode'),0o700);self.assertTrue(seen['readonly'])
        self.assertEqual(seen['argv'][seen['argv'].index('--model')+1],'synthetic-model')
        self.assertEqual(seen['argv'][seen['argv'].index('--effort')+1],'medium')
        directive=seen['argv'][seen['argv'].index('-p')+1]
        self.assertNotIn('HEAD-SYNTHETIC',directive);self.assertNotIn('MIDDLE-SYNTHETIC',directive);self.assertNotIn('TAIL-SYNTHETIC',directive)
        self.assertIn(hashlib.sha256(text.encode()).hexdigest(),directive)

    def test_privacy_denial_occurs_before_staging_or_any_cli_capability_spawn(self):
        with mock.patch.object(agy,'refuse_payload',return_value={'ok':False,'reason':'privacy_denied'}), \
             mock.patch.object(agy.tempfile,'mkdtemp') as stage, \
             mock.patch('providers.model_catalog.agy_effort_tokens') as capability, \
             mock.patch.object(agy,'run_cli') as child:
            result=asyncio.run(agy.run_agy_subprocess(payload(),'synthetic-model',30,'default',effort='medium'))
        self.assertEqual(result['reason'],'privacy_denied');stage.assert_not_called();capability.assert_not_called();child.assert_not_called()

    def test_real_local_child_reads_complete_file_through_run_cli(self):
        with tempfile.TemporaryDirectory() as td:
            child=Path(td)/'reader.py'
            child.write_text("import json,pathlib,re,sys,hashlib\np=sys.argv[sys.argv.index('-p')+1]\nm=re.search(r'^INPUT_FILE: (.+)$',p,re.M)\nparts=[]\nif m:\n f=pathlib.Path(json.loads(m[1])).open('rb')\n while True:\n  data=f.read(65536)\n  if not data:break\n  parts.append(data)\n f.close()\nelse:parts=[p.encode()[:191602]]\ndata=b''.join(parts)\nprint(json.dumps({'status':'SUCCESS','response':json.dumps({'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'head':data.startswith(b'HEAD-SYNTHETIC'),'middle':b'MIDDLE-SYNTHETIC' in data,'tail':data.endswith(b'TAIL-SYNTHETIC\\r\\n')})}))\n")
            def prepare(argv,**kwargs):return [sys.executable,str(child),*argv[1:]],kwargs['env'],{}
            result,seen=self.run_case(payload(),runner=run_cli,prepare=prepare)
        self.assertTrue(result['ok'],result);reply=json.loads(result['text'])
        self.assertEqual(reply,{'bytes':605743,'sha256':hashlib.sha256(payload().encode()).hexdigest(),'head':True,'middle':True,'tail':True})
        self.assertFalse(Path(seen['cwd']).exists())


    def test_exclusive_nofollow_creation_is_0600_before_0400(self):
        real_open,real_chmod=agy.os.open,agy.os.fchmod
        opened,chmods=[],[]
        def opened_file(path,flags,mode=0o777):
            opened.append((str(path),flags,mode));return real_open(path,flags,mode)
        def chmod(fd,mode):chmods.append(mode);return real_chmod(fd,mode)
        with mock.patch.object(agy.os,'open',side_effect=opened_file),mock.patch.object(agy.os,'fchmod',side_effect=chmod):
            result,_=self.run_case(payload())
        opened=[entry for entry in opened if entry[0].endswith('/omni-input.txt')]
        self.assertTrue(result['ok'],result);self.assertEqual(len(opened),1)
        self.assertTrue(opened[0][1]&os.O_EXCL);self.assertTrue(opened[0][1]&os.O_NOFOLLOW)
        self.assertEqual(opened[0][2],0o600);self.assertEqual(chmods,[0o400])

    def test_small_and_threshold_sized_prompts_keep_legacy_inline_argv(self):
        for size in (2,config.AGY_PROMPT_FILE_THRESHOLD_BYTES):
            with self.subTest(size=size):
                text='x'*size;result,seen=self.run_case(text)
                self.assertTrue(result['ok'],result);self.assertIsNone(input_file(seen['argv']))
                self.assertEqual(seen['argv'][seen['argv'].index('-p')+1],text)
                self.assertFalse(Path(seen['cwd']).exists())

    def test_just_over_threshold_routes_to_file_and_preserves_profile(self):
        with mock.patch.dict(os.environ,{'AGY_CONFIG_DIR_SYNTHETIC':'/synthetic/profile'}):
            def prepare(argv,**kwargs):
                self.assertEqual(kwargs['env']['AGY_CONFIG_DIR'],'/synthetic/profile')
                return argv,kwargs['env'],{}
            text='x'*(config.AGY_PROMPT_FILE_THRESHOLD_BYTES+1)
            result,seen=self.run_case(text,prepare=prepare,profile='synthetic',effort='medium')
        self.assertTrue(result['ok'],result);self.assertEqual(seen['data'],text.encode())

    def test_shared_file_cap_accepts_exact_limit_and_rejects_one_more_before_stage(self):
        self.assertEqual(config.SOURCE_PROMPT_MAX_BYTES,12*1024*1024)
        text='x'*config.SOURCE_PROMPT_MAX_BYTES
        result,seen=self.run_case(text)
        self.assertTrue(result['ok'],result);self.assertEqual(len(seen['data']),config.SOURCE_PROMPT_MAX_BYTES)
        with mock.patch.object(agy,'refuse_payload',return_value=None),mock.patch.object(agy.tempfile,'mkdtemp') as stage,mock.patch.object(agy,'run_cli') as child:
            result=asyncio.run(agy.run_agy_subprocess(text+'x','synthetic-model',30,'default'))
        self.assertEqual(result['reason'],'payload_too_large');stage.assert_not_called();child.assert_not_called()

    def test_malformed_unicode_is_a_typed_local_failure_before_stage(self):
        with mock.patch.object(agy.tempfile,'mkdtemp') as stage,mock.patch.object(agy,'run_cli') as child:
            result=asyncio.run(agy.run_agy_subprocess(chr(0xd800),'synthetic-model',30,'default'))
        self.assertEqual(result['reason'],'runtime_error');self.assertIn('UTF-8',result['detail'])
        stage.assert_not_called();child.assert_not_called()

    def test_write_failure_leaves_no_owned_directory_and_never_spawns(self):
        real=agy.tempfile.mkdtemp;paths=[]
        def owned(*a,**kw):p=real(*a,**kw);paths.append(Path(p));return p
        child=mock.AsyncMock()
        with mock.patch.object(agy.tempfile,'mkdtemp',side_effect=owned),mock.patch.object(agy.os,'write',side_effect=OSError('synthetic write failure')):
            result,_=self.run_case(payload(),runner=child)
        self.assertEqual(result['reason'],'runtime_error');self.assertTrue(is_local_failure(result));child.assert_not_called()
        self.assertTrue(paths);self.assertTrue(all(not p.exists() for p in paths))

    def test_fence_refusal_cleans_file_without_spawning(self):
        child=mock.AsyncMock()
        def refuse(*a,**kw):raise agy.HostFenceError({'ok':False,'reason':'fence_unavailable','detail':'synthetic'})
        result,seen=self.run_case(payload(),runner=child,prepare=refuse)
        self.assertEqual(result['reason'],'fence_unavailable');self.assertTrue(is_local_failure(result));child.assert_not_called()
        self.assertFalse(Path(seen['cwd']).exists())

    def test_error_and_timeout_clean_the_owned_input(self):
        for response in ({'returncode':1,'stdout':'','stderr':'synthetic failure','timed_out':False},
                         {'returncode':0,'stdout':'','stderr':'','timed_out':True}):
            async def failed(*a,**kw):return response
            result,seen=self.run_case(payload(),runner=failed)
            self.assertFalse(result['ok']);self.assertFalse(Path(seen['cwd']).exists())

    def test_cancellation_cleans_the_owned_input(self):
        real=agy.tempfile.mkdtemp;paths=[]
        def owned(*a,**kw):p=real(*a,**kw);paths.append(Path(p));return p
        async def cancelled(*a,**kw):raise asyncio.CancelledError()
        with mock.patch.object(agy.tempfile,'mkdtemp',side_effect=owned),self.assertRaises(asyncio.CancelledError):
            self.run_case(payload(),runner=cancelled)
        self.assertTrue(paths);self.assertTrue(all(not p.exists() for p in paths))

    def test_sdk_selection_without_explicit_effort_is_unchanged(self):
        sdk=mock.AsyncMock(return_value={'ok':True,'text':'synthetic SDK response'})
        with mock.patch.object(agy,'HAS_SDK',True),mock.patch.object(agy.platform,'system',return_value='Linux'),mock.patch.object(agy,'run_agy_sdk',sdk),mock.patch.object(agy,'run_agy_subprocess') as cli:
            result=asyncio.run(agy.run_agy(payload(),'synthetic-model',30))
        self.assertTrue(result['ok']);sdk.assert_awaited_once();cli.assert_not_called()


    def test_existing_symlink_is_never_followed_and_target_is_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            target=Path(td)/'synthetic-sentinel';target.write_bytes(b'unchanged')
            owned=Path(td)/'owned';owned.mkdir(mode=0o700)
            (owned/'omni-input.txt').symlink_to(target)
            child=mock.AsyncMock()
            with mock.patch.object(agy.tempfile,'mkdtemp',return_value=str(owned)):
                result,_=self.run_case(payload(),runner=child)
            self.assertEqual(result['reason'],'runtime_error');self.assertTrue(is_local_failure(result));child.assert_not_called()
            self.assertEqual(target.read_bytes(),b'unchanged');self.assertFalse(owned.exists())

    def test_privacy_gate_error_is_local_failure_without_staging(self):
        with mock.patch.object(agy,'refuse_payload',side_effect=RuntimeError('synthetic gate error')),mock.patch.object(agy.tempfile,'mkdtemp') as stage,mock.patch.object(agy,'run_cli') as child:
            result=asyncio.run(agy.run_agy_subprocess(payload(),'synthetic-model',30,'default'))
        self.assertEqual(result['reason'],'runtime_error');self.assertTrue(is_local_failure(result));stage.assert_not_called();child.assert_not_called()

if __name__=='__main__':unittest.main(verbosity=2)
