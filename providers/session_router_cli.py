"""Local session CLI. No model dispatch, credential loading or vendor binaries."""
import argparse
import json
import os
import sys

from providers.session_router import INSTRUCTIONS, MAX_BODY_BYTES, Registry, RouterError


def _parser():
    parser = argparse.ArgumentParser(prog='omni session', description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('attach', 'update', 'observe', 'peers', 'get', 'leave', 'send', 'hook', 'receive'):
        p = sub.add_parser(name)
        if name in ('get', 'send'):
            p.add_argument('presence_id')
        if name not in ('peers', 'get'):
            p.add_argument('--session-key', default=os.environ.get('OMNI_SESSION_KEY'))
            p.add_argument('--incarnation', default=os.environ.get('OMNI_SESSION_INCARNATION'))
        if name not in ('get', 'leave', 'send'):
            p.add_argument('--cwd', default=os.getcwd())
        if name == 'attach':
            p.add_argument('--adapter', default=os.environ.get('OMNI_SESSION_ADAPTER', 'manual'))
            p.add_argument('--route', default=os.environ.get('OMNI_SESSION_ROUTE'))
            p.add_argument('--harness', default='unknown')
            p.add_argument('--role', default='session')
        if name == 'update':
            p.add_argument('--expected-revision', type=int, required=True)
            p.add_argument('--task', required=True)
            p.add_argument('--work-state', choices=['working', 'blocked', 'done', 'unknown'], required=True)
        if name == 'observe':
            p.add_argument('--sequence', type=int, required=True)
            p.add_argument('--runtime-state', choices=['busy', 'idle', 'unknown'], default='unknown')
        if name == 'receive':
            p.add_argument('--timeout', type=float, default=30)
            p.add_argument('--harness', default='unknown')
            p.add_argument('--role', default='session')
    return parser


def _read_stdin(limit):
    # Read a bounded binary payload; never pass intentional messages in argv.
    stream = getattr(sys.stdin, 'buffer', sys.stdin)
    data = stream.read(limit + 1)
    if isinstance(data, str):
        data = data.encode('utf-8')
    if len(data) > limit:
        raise RouterError('invalid_body')
    try:
        return data.decode('utf-8')
    except UnicodeError:
        raise RouterError('invalid_body') from None


def _hook(router, ns):
    try:
        event = json.loads(_read_stdin(8192))
    except (ValueError, TypeError):
        raise RouterError('invalid_event') from None
    allowed = {'event', 'session_key', 'incarnation', 'cwd', 'adapter', 'route',
               'harness', 'role', 'sequence', 'runtime_state'}
    if not isinstance(event, dict) or set(event) - allowed:
        raise RouterError('invalid_event')
    if event.get('event') not in ('start', 'resume', 'checkpoint', 'exit'):
        raise RouterError('invalid_event')
    for key in ('adapter', 'harness', 'role'):
        if key in event and not isinstance(event[key], str):
            raise RouterError('invalid_metadata')
    if 'route' in event and event['route'] is None:
        raise RouterError('invalid_route')
    binding = {'session_key': event.get('session_key', ns.session_key),
               'incarnation': event.get('incarnation', ns.incarnation)}
    row = router.lifecycle(
        **binding, event=event['event'], sequence=event.get('sequence'),
        cwd=event.get('cwd', ns.cwd), runtime_state=event.get('runtime_state', 'unknown'),
        adapter=event.get('adapter', os.environ.get('OMNI_SESSION_ADAPTER')),
        route=event.get('route', os.environ.get('OMNI_SESSION_ROUTE')),
        harness=event.get('harness'), role=event.get('role'))
    return {'ok': True, 'session': row, 'warnings': router.warnings(row),
            'instructions': INSTRUCTIONS, 'needs_task_update': bool(router.warnings(row))}


def main(argv=None):
    ns = _parser().parse_args(argv)
    try:
        router = Registry()
        binding = {'session_key': getattr(ns, 'session_key', None),
                   'incarnation': getattr(ns, 'incarnation', None)}
        if ns.command == 'attach':
            row = router.attach(**binding, cwd=ns.cwd, adapter=ns.adapter,
                                route=ns.route, harness=ns.harness, role=ns.role)
            result = {'ok': True, 'session': row, 'instructions': INSTRUCTIONS,
                      'warnings': router.warnings(row)}
        elif ns.command == 'update':
            row = router.update(**binding, expected_revision=ns.expected_revision,
                                task=ns.task, work_state=ns.work_state)
            result = {'ok': True, 'session': row, 'warnings': router.warnings(row)}
        elif ns.command == 'observe':
            row = router.observe(**binding, sequence=ns.sequence,
                                 runtime_state=ns.runtime_state, cwd=ns.cwd)
            result = {'ok': True, 'session': row, 'warnings': router.warnings(row)}
        elif ns.command == 'peers':
            result = {'ok': True, 'sessions': router.peers(cwd=ns.cwd)}
        elif ns.command == 'get':
            result = {'ok': True, 'session': router.get(ns.presence_id)}
        elif ns.command == 'leave':
            result = {'ok': True, 'session': router.leave(**binding)}
        elif ns.command == 'send':
            result = router.send(**binding, target_id=ns.presence_id,
                                 body=_read_stdin(MAX_BODY_BYTES))
        elif ns.command == 'receive':
            from providers.session_router_unix import receive_once
            result = receive_once(router, **binding, cwd=ns.cwd, timeout_s=ns.timeout,
                                  harness=ns.harness, role=ns.role)
            if result.get('ok'):
                result['instructions'] = INSTRUCTIONS
        else:
            result = _hook(router, ns)
    except RouterError as exc:
        result = {'ok': False, 'reason': exc.reason}
    except Exception:
        # No exception text: an adapter may include an intentional message body.
        result = {'ok': False, 'reason': 'router_failure'}
    sys.stdout.write(json.dumps(result, ensure_ascii=False) + '\n')
    return 0 if result.get('ok') else 1
