#!/usr/bin/env python3
"""Opt-in synthetic Grok tool-process smoke under the production host adapter.

Requires a dedicated, already authenticated --config-dir (never commit it).
This does not certify a vendor sandbox; it checks actual tools under our wrapper.
Example: python3 verify/host_worker_smoke.py --config-dir DIR --model MODEL
         --effort low --denylist FILE --execute
Only synthetic input is sent. Receipts exclude auth and configuration contents.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from providers import egress_gate, host_fence, kernel_sandbox

PROBE = r'''import errno,json,os,pathlib,pwd,subprocess,sys
root=pathlib.Path.cwd()
config=json.loads((root/'fixture.json').read_text())
canary=pathlib.Path(pwd.getpwuid(os.getuid()).pw_dir)/config['canary_name']
result={'pid':os.getpid(),'ppid':os.getppid(),'nonce':(root/'nonce.txt').read_text().strip()}
(root/'allowed.txt').write_text(result['nonce'])
for action in ('read','write'):
 try:
  if action=='read': canary.read_text()
  else: canary.write_text('unexpected')
 except OSError as exc: result[action+'_errno']=exc.errno
 else: result[action+'_errno']=0
if '--child' not in sys.argv:
 child=subprocess.run([sys.executable,__file__,'--child'],capture_output=True,text=True)
 result['child_rc']=child.returncode
 result['child']=json.loads(child.stdout)
 (root/'receipt.json').write_text(json.dumps(result))
 # Keep the actual tool PID alive briefly for the outside launcher observer.
 __import__('time').sleep(3)
print(json.dumps(result))
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config-dir',required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--effort',required=True)
    p.add_argument('--denylist',required=True)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    if not args.execute:
        p.error('live call requires --execute')
    cfg=Path(args.config_dir).resolve()
    if not (cfg/'auth.json').is_file():
        p.error('dedicated config directory must already be authenticated')
    # This fixture requires fresh-process routing; never inherit a shared leader.
    config=(cfg/'config.toml').read_text()
    if 'use_leader = false' not in config:
        p.error('dedicated config must explicitly set [cli] use_leader = false')
    root=Path(tempfile.mkdtemp(prefix='host-worker-smoke-')).resolve()
    nonce=secrets.token_hex(16)
    canary_name='.host-worker-smoke-'+secrets.token_hex(8)
    home=Path(kernel_sandbox.protected_home())
    canary=home/canary_name
    canary.write_text('synthetic-outside-fixture')
    os.chmod(canary,0o600)
    (root/'nonce.txt').write_text(nonce)
    (root/'fixture.json').write_text(json.dumps({'canary_name':canary_name}))
    (root/'probe.py').write_text(PROBE)
    prompt='Synthetic integration check. Use run_terminal_cmd once to execute /usr/bin/python3 probe.py in the current directory. Do not inspect any other files or use other tools. Return SMOKE_DONE after that command finishes.'
    (root/'prompt.txt').write_text(prompt)
    binary=shutil.which('grok')
    if not binary: p.error('grok CLI missing')
    argv=[str(Path(binary).resolve()),'--prompt-file',str(root/'prompt.txt'),'--model',args.model,'--reasoning-effort',args.effort,'--output-format','json','--no-auto-update','--no-memory','--disable-web-search','--no-subagents','--max-turns','3','--always-approve','--tools','run_terminal_cmd','--system-prompt-override','Perform only the supplied synthetic terminal check. Do not read other files.','--cwd',str(root)]
    env={'PATH':'/usr/bin:/bin:/opt/homebrew/bin','HOME':str(home),'GROK_HOME':str(cfg),'NO_COLOR':'1'}
    try:
        wrapped,env,meta=host_fence.prepare_host_fence(argv,provider='grok',workdir=str(root),env=env,config_dir=str(cfg))
        shim=['/bin/sh','-c',kernel_sandbox.GROK_PID_SHIM]
        assert kernel_sandbox.is_wrapped(wrapped[3:] if wrapped[:3]==shim else wrapped)
        # Adjacent complete model-visible synthetic input scan, before the send.
        os.environ[egress_gate.ENV_DENYLIST]=args.denylist
        outgoing=[prompt,PROBE,(root/'fixture.json').read_text(),nonce,argv[-3]]
        if egress_gate.check_strings(outgoing): raise RuntimeError('outgoing privacy scan refused')
        proc=subprocess.Popen(wrapped,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
        deadline=time.monotonic()+180
        observed=[]
        while proc.poll() is None and time.monotonic()<deadline:
            if not observed and (root/'receipt.json').exists():
                try:
                    tool=json.loads((root/'receipt.json').read_text())
                    pid=tool['pid']
                    for _ in range(24):
                        if pid<=1:break
                        observed.append(pid)
                        if pid==proc.pid:break
                        raw=subprocess.check_output(['/bin/ps','-o','ppid=','-p',str(pid)],text=True)
                        pid=int(raw.strip())
                except (OSError,ValueError,subprocess.CalledProcessError):
                    observed=[]
            time.sleep(.05)
        if proc.poll() is None:os.killpg(proc.pid,signal.SIGKILL)
        out,err=proc.communicate()
        (root/'stdout.json').write_text(out);(root/'stderr.log').write_text(err)
        result={'rc':proc.returncode,'launcher_pid':proc.pid,'observed_tool_ancestry':observed,'fence':meta,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()}
        if (root/'receipt.json').exists():
            actual=json.loads((root/'receipt.json').read_text())
            result['tool']=actual
            result['pass']=bool(proc.returncode==0 and proc.pid in observed and actual['nonce']==nonce and actual['read_errno']==1 and actual['write_errno']==1 and actual['child_rc']==0 and actual['child']['read_errno']==1 and actual['child']['write_errno']==1 and actual['child']['ppid']==actual['pid'] and (root/'allowed.txt').read_text()==nonce and canary.read_text()=='synthetic-outside-fixture')
        else: result['pass']=False
        (root/'result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps({'pass':result['pass'],'artifact':str(root),'rc':proc.returncode}))
        return 0 if result['pass'] else 1
    finally:
        canary.unlink(missing_ok=True)


if __name__=='__main__':sys.exit(main())
