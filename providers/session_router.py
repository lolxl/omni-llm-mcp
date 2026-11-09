"""Local metadata-only session registry and ephemeral, guarded adapter forwarding.

A binding is (session_key, incarnation), not a public presence ID. This prevents
accidental impersonation; a same-UID process is outside the security boundary.
Adapters own atomic live identity validation AND forwarding in guarded_forward.
"""
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import os
import re
import stat
import subprocess
import time
import uuid

MAX_BODY_BYTES = 65536
WORK_STATES = frozenset(('working', 'blocked', 'done', 'unknown'))
RUNTIME_STATES = frozenset(('busy', 'idle', 'unknown'))
INSTRUCTIONS = (
    'Keep OMNI_SESSION_KEY and OMNI_SESSION_INCARNATION stable for this existing '
    'session; a replacement needs a new incarnation. Before work or scope changes, '
    'run omni session update --expected-revision N --task SUMMARY --work-state working. '
    'Declare blocked/done explicitly with update; idle and leave do not mean done. '
    'Check omni session peers, then send intentional peer-authored data with '
    'omni session send PRESENCE_ID < message.txt. At an explicit checkpoint run '
    'omni session receive --timeout 30 to accept one message; sending requires a waiting receiver. '
    'A handoff means transport transfer only, not read, saved or replied. Receivers own saved messages. '
    'Use project-local lifecycle hooks/checkpoints to refresh observations.'
)


class RouterError(Exception):
    """Only a fixed reason code, never an adapter exception or message body."""
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def _text(value, limit=512, reason='invalid_metadata', empty=False):
    if (not isinstance(value, str) or len(value.encode('utf-8')) > limit
            or (not value and not empty) or any(ord(c) < 32 for c in value)):
        raise RouterError(reason)
    return value


def _binding(session_key, incarnation):
    _text(session_key, reason='invalid_binding')
    _text(incarnation, reason='invalid_binding')
    return hashlib.sha256(session_key.encode('utf-8')).hexdigest()


def _route(adapter, route):
    if adapter == 'unix':
        if not isinstance(route, dict) or set(route) != {'path', 'nonce'}:
            raise RouterError('invalid_route')
        _text(route['path'], 1024, reason='invalid_route')
        _text(route['nonce'], 256, reason='invalid_route')
        if not os.path.isabs(route['path']):
            raise RouterError('invalid_route')
    else:
        _text(route, 2048, reason='invalid_route')
    return copy.deepcopy(route)


def _sequence(value):
    if type(value) is not int or value < 0:
        raise RouterError('invalid_sequence')
    return value


def _git_context(cwd):
    try:
        cwd = os.path.realpath(os.fspath(cwd))
        def git(*args):
            p = subprocess.run(['git', '-C', cwd, *args], capture_output=True,
                               text=True, timeout=5, check=True)
            return p.stdout.strip()
        common = git('rev-parse', '--git-common-dir')
        return {'repo': os.path.realpath(os.path.join(cwd, common)),
                'worktree': os.path.realpath(git('rev-parse', '--show-toplevel')),
                'branch': git('rev-parse', '--abbrev-ref', 'HEAD')}
    except (OSError, ValueError, subprocess.SubprocessError):
        # Unborn Git repositories are valid; their HEAD has no commit yet.
        try:
            p = subprocess.run(['git', '-C', cwd, 'symbolic-ref', '--short', 'HEAD'],
                               capture_output=True, text=True, timeout=5, check=True)
            return {'repo': os.path.realpath(os.path.join(cwd, common)),
                    'worktree': os.path.realpath(git('rev-parse', '--show-toplevel')),
                    'branch': p.stdout.strip()}
        except (OSError, ValueError, UnboundLocalError, subprocess.SubprocessError):
            raise RouterError('not_git_repository') from None


def _open_directory(path):
    """Walk with no-follow descriptors; do not resolve an unsafe path first."""
    path = os.path.abspath(os.path.expanduser(os.fspath(path)))
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in path.split('/')[1:]:
            if not component:
                continue
            try:
                os.mkdir(component, 0o700, dir_fd=fd)
            except FileExistsError:
                pass
            nxt = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                          dir_fd=fd)
            os.close(fd)
            fd = nxt
        info = os.fstat(fd)
        if info.st_uid != os.getuid() or info.st_mode & 0o022:
            raise RouterError('unsafe_state_path')
        return fd
    except BaseException:
        os.close(fd)
        raise


def _safe_file(fd):
    info = os.fstat(fd)
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
            or info.st_mode & 0o077 or info.st_nlink != 1):
        raise RouterError('unsafe_state_path')


class Registry:
    def __init__(self, state_dir=None, *, adapters=None):
        self.state_dir = (state_dir if state_dir is not None else
                          os.environ.get('OMNI_LLM_STATE_DIR') or '~/.local/state/omni-llm')
        if adapters is None:
            from providers.session_router_unix import UnixAdapter
            adapters = {'unix': UnixAdapter()}
        self.adapters = dict(adapters)

    @contextmanager
    def _table(self, write=False):
        root = directory = lock = None
        try:
            root = _open_directory(self.state_dir)
            try:
                os.mkdir('sessions', 0o700, dir_fd=root)
            except FileExistsError:
                pass
            directory = os.open('sessions', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                dir_fd=root)
            info = os.fstat(directory)
            if info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise RouterError('unsafe_state_path')
            try:
                lock = os.open('table.lock', os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW,
                               0o600, dir_fd=directory)
            except FileExistsError:
                lock = os.open('table.lock', os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK,
                               dir_fd=directory)
            _safe_file(lock)
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                fd = os.open('table.json', os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                             dir_fd=directory)
            except FileNotFoundError:
                table = {'version': 1, 'sessions': []}
            else:
                try:
                    _safe_file(fd)
                    with os.fdopen(fd, 'r', encoding='utf-8') as handle:
                        fd = None
                        table = json.load(handle)
                finally:
                    if fd is not None:
                        os.close(fd)
                self._validate_table(table)
            yield table
            if write:
                name = 'table-' + uuid.uuid4().hex + '.tmp'
                fd = os.open(name, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
                             0o600, dir_fd=directory)
                try:
                    with os.fdopen(fd, 'w', encoding='utf-8') as handle:
                        json.dump(table, handle, ensure_ascii=False, sort_keys=True)
                        handle.write('\n')
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(name, 'table.json', src_dir_fd=directory, dst_dir_fd=directory)
                    os.fsync(directory)
                finally:
                    try:
                        os.unlink(name, dir_fd=directory)
                    except FileNotFoundError:
                        pass
        except RouterError:
            raise
        except (json.JSONDecodeError, UnicodeError, TypeError, KeyError):
            raise RouterError('invalid_registry') from None
        except OSError:
            raise RouterError('unsafe_state_path') from None
        finally:
            for fd in (lock, directory, root):
                if fd is not None:
                    os.close(fd)

    @staticmethod
    def _validate_table(table):
        if (not isinstance(table, dict) or set(table) != {'version', 'sessions'}
                or table['version'] != 1 or not isinstance(table['sessions'], list)):
            raise RouterError('invalid_registry')
        expected = {'presence_id', 'binding', 'incarnation', 'adapter', 'route', 'harness',
                    'role', 'repo', 'worktree', 'branch', 'task', 'task_revision',
                    'work_state', 'runtime_state', 'observation_sequence', 'last_seen',
                    'last_task_update', 'state', 'closed_at', 'task_context'}
        ids, bindings, active = set(), set(), set()
        for row in table['sessions']:
            if not isinstance(row, dict) or set(row) != expected:
                raise RouterError('invalid_registry')
            strings = expected - {'route', 'task_revision', 'observation_sequence', 'last_seen',
                                  'last_task_update', 'closed_at', 'task_context'}
            if any(not isinstance(row[k], str) for k in strings):
                raise RouterError('invalid_registry')
            if (not re.fullmatch('[0-9a-f]{32}', row['presence_id'])
                    or not re.fullmatch('[0-9a-f]{64}', row['binding'])
                    or row['presence_id'] in ids
                    or (row['binding'], row['incarnation']) in bindings
                    or row['state'] not in ('attached', 'closed')
                    or row['work_state'] not in WORK_STATES
                    or row['runtime_state'] not in RUNTIME_STATES
                    or type(row['task_revision']) is not int or row['task_revision'] < 0
                    or type(row['observation_sequence']) is not int or row['observation_sequence'] < -1
                    or type(row['last_seen']) not in (int, float)
                    or (row['last_task_update'] is not None
                        and type(row['last_task_update']) not in (int, float))
                    or (row['closed_at'] is not None and type(row['closed_at']) not in (int, float))
                    or not isinstance(row['task_context'], dict)
                    or set(row['task_context']) != {'repo', 'worktree', 'branch'}):
                raise RouterError('invalid_registry')
            if row['state'] == 'attached':
                if row['binding'] in active:
                    raise RouterError('invalid_registry')
                active.add(row['binding'])
            try:
                _route(row['adapter'], row['route'])
            except RouterError:
                raise RouterError('invalid_registry') from None
            ids.add(row['presence_id'])
            bindings.add((row['binding'], row['incarnation']))

    @staticmethod
    def _public(row):
        return copy.deepcopy({k: v for k, v in row.items() if k != 'binding'})

    @staticmethod
    def warnings(row):
        warnings = []
        if not row['task']:
            warnings.append('task_empty')
        if row['task_context'] != {key: row[key] for key in ('repo', 'worktree', 'branch')}:
            warnings.append('task_context_changed')
        return warnings

    @staticmethod
    def _caller(table, session_key, incarnation):
        binding = _binding(session_key, incarnation)
        for row in table['sessions']:
            if row['binding'] == binding and row['incarnation'] == incarnation:
                if row['state'] != 'attached':
                    raise RouterError('session_closed')
                return row
        raise RouterError('caller_not_attached')

    def attach(self, *, session_key, incarnation, cwd, adapter='manual', route,
               harness='unknown', role='session'):
        binding = _binding(session_key, incarnation)
        for value in (adapter, harness, role):
            _text(value)
        route = _route(adapter, route)
        context = _git_context(cwd)
        with self._table(write=True) as table:
            row = self._attach_record(table, binding, incarnation, context,
                                      adapter, route, harness, role)
            return self._public(row)

    def _attach_record(self, table, binding, incarnation, context, adapter, route, harness, role):
        for row in table['sessions']:
            if row['binding'] != binding:
                continue
            if row['incarnation'] == incarnation:
                if row['state'] != 'attached':
                    raise RouterError('session_closed')
                if any(row[k] != v for k, v in (('adapter', adapter), ('route', route),
                                                ('harness', harness), ('role', role))):
                    raise RouterError('binding_conflict')
                return row
            if row['state'] == 'attached':
                raise RouterError('incarnation_conflict')
        row = dict(context, presence_id=uuid.uuid4().hex, binding=binding,
                   incarnation=incarnation, adapter=adapter, route=route,
                   harness=harness, role=role, task='', task_revision=0,
                   work_state='unknown', runtime_state='unknown',
                   observation_sequence=-1, last_seen=time.time(),
                   last_task_update=None, state='attached', closed_at=None,
                   task_context=context.copy())
        table['sessions'].append(row)
        return row

    def lifecycle(self, *, event, session_key, incarnation, sequence, cwd,
                  runtime_state='unknown', adapter=None, route=None, harness=None, role=None):
        """Apply one normalized hook atomically; failures never persist registration.

        Hook exit is an ordered observation. Manual leave remains an explicit,
        sequence-free owner action. Optional attachment metadata is validated but
        cannot replace an existing transport or task declaration.
        """
        if event not in ('start', 'resume', 'checkpoint', 'exit'):
            raise RouterError('invalid_event')
        binding = _binding(session_key, incarnation)
        _sequence(sequence)
        if not isinstance(runtime_state, str) or runtime_state not in RUNTIME_STATES:
            raise RouterError('invalid_runtime_state')
        for value in (adapter, harness, role):
            if value is not None:
                _text(value)
        if not isinstance(cwd, (str, os.PathLike)):
            raise RouterError('not_git_repository')
        # Resolve Git before any mutation. Exit need not inspect a vanished cwd.
        context = _git_context(cwd) if event != 'exit' else None
        with self._table(write=True) as table:
            try:
                row = self._caller(table, session_key, incarnation)
            except RouterError as exc:
                if exc.reason != 'caller_not_attached' or event == 'exit':
                    raise
                # All attachment and observation fields validate in this one
                # transaction; no attach is published before observation succeeds.
                selected_adapter = adapter if adapter is not None else 'manual'
                selected_route = _route(selected_adapter, route)
                row = self._attach_record(table, binding, incarnation, context,
                                          selected_adapter, selected_route,
                                          harness if harness is not None else 'unknown',
                                          role if role is not None else 'session')
            else:
                if route is not None:
                    # Bootstrap metadata describes its own transport, which can
                    # predate an explicit receive/set_route upgrade. Validate it
                    # without deriving its schema from or changing the live route.
                    bootstrap_adapter = adapter if adapter is not None else (
                        'unix' if isinstance(route, dict) else 'manual')
                    _route(bootstrap_adapter, route)
            if sequence <= row['observation_sequence']:
                raise RouterError('stale_observation')
            now = time.time()
            row.update(observation_sequence=sequence, runtime_state=runtime_state, last_seen=now)
            if event == 'exit':
                row.update(state='closed', closed_at=now)
            else:
                row.update(context)
            return self._public(row)

    def lookup(self, *, session_key, incarnation):
        with self._table() as table:
            return self._public(self._caller(table, session_key, incarnation))

    def set_route(self, *, session_key, incarnation, adapter, route):
        _text(adapter)
        route = _route(adapter, route)
        with self._table(write=True) as table:
            row = self._caller(table, session_key, incarnation)
            row.update(adapter=adapter, route=route)
            return self._public(row)

    def update(self, *, session_key, incarnation, expected_revision, task, work_state):
        _text(task, 2048, empty=True)
        if work_state not in WORK_STATES:
            raise RouterError('invalid_work_state')
        _sequence(expected_revision)
        with self._table(write=True) as table:
            row = self._caller(table, session_key, incarnation)
            if row['task_revision'] != expected_revision:
                raise RouterError('task_revision_conflict')
            row.update(task=task, work_state=work_state, task_revision=expected_revision + 1,
                       last_task_update=time.time(),
                       task_context={k: row[k] for k in ('repo', 'worktree', 'branch')})
            return self._public(row)

    def observe(self, *, session_key, incarnation, sequence, runtime_state='unknown', cwd):
        _sequence(sequence)
        if runtime_state not in RUNTIME_STATES:
            raise RouterError('invalid_runtime_state')
        context = _git_context(cwd)
        with self._table(write=True) as table:
            row = self._caller(table, session_key, incarnation)
            if sequence <= row['observation_sequence']:
                raise RouterError('stale_observation')
            row.update(context, observation_sequence=sequence, runtime_state=runtime_state,
                       last_seen=time.time())
            return self._public(row)

    def leave(self, *, session_key, incarnation):
        with self._table(write=True) as table:
            row = self._caller(table, session_key, incarnation)
            row.update(state='closed', closed_at=time.time())
            return self._public(row)

    def peers(self, *, cwd):
        context = _git_context(cwd)
        with self._table() as table:
            return [self._public(row) for row in table['sessions']
                    if row['repo'] == context['repo'] and row['state'] == 'attached']

    def get(self, presence_id):
        with self._table() as table:
            for row in table['sessions']:
                if row['presence_id'] == presence_id:
                    return self._public(row)
        raise RouterError('unknown_target')

    def send(self, *, session_key, incarnation, target_id, body):
        if not isinstance(body, str):
            raise RouterError('invalid_body')
        try:
            size = len(body.encode('utf-8'))
        except UnicodeError:
            raise RouterError('invalid_body') from None
        if not 0 < size <= MAX_BODY_BYTES:
            raise RouterError('invalid_body')
        # Serializes local replacement/leave with handoff. External replacement is
        # guarded atomically by the adapter itself, never a get-then-prompt probe.
        with self._table() as table:
            sender = self._caller(table, session_key, incarnation)
            target = next((r for r in table['sessions'] if r['presence_id'] == target_id), None)
            if target is None:
                raise RouterError('unknown_target')
            if target['state'] != 'attached':
                raise RouterError('target_closed')
            adapter = self.adapters.get(target['adapter'])
            if adapter is None:
                return {'ok': False, 'reason': ('identity_binding_unsupported'
                        if target['adapter'] == 'herdr' else 'adapter_unavailable')}
            try:
                destination = self._public(target)
                destination['_sender_id'] = sender['presence_id']
                result = adapter.guarded_forward(destination, body,
                                                 expected_incarnation=target['incarnation'])
                if result.get('ok') is True and result.get('status') == 'handoff':
                    return {'ok': True, 'status': 'handoff'}
                allowed = {'target_replaced', 'target_unavailable', 'target_busy',
                           'identity_binding_unsupported', 'adapter_unavailable',
                           'handoff_unknown', 'adapter_failure', 'identity_changed',
                           'endpoint_unavailable', 'invalid_route', 'invalid_message',
                           'protocol_error', 'timeout'}
                reason = result.get('reason')
                return {'ok': False, 'reason': reason if reason in allowed else 'adapter_failure'}
            except Exception:
                return {'ok': False, 'reason': 'adapter_failure'}
