# Local Session Router

Omni maintains a local session table and forwards intentional peer messages.
Sessions own their received messages. There is no inbox, outbox, delivery history,
retry queue, model launch, transcript reader or background heartbeat.

## Stable identity and registration

Use an explicit stable session key and incarnation from the existing harness:
`OMNI_SESSION_KEY` and `OMNI_SESSION_INCARNATION`, or the corresponding CLI flags.
Never use the currently focused pane or a PID as a guessed identity. Keep both
values unchanged across checkpoints and resume. Repeated attach reuses a presence
ID. Leave closes that incarnation permanently; replacements use a new incarnation.
An attached incarnation must leave before the same key can be replaced. Public
presence IDs are discovery identifiers; mutations require the caller binding.
This protects against accidental impersonation, not another process with your UID.
Role `foreman` is descriptive metadata and grants no authority.

These examples run in a Git repository with those two environment variables set.
The default manual adapter registers metadata without enabling message transport.
Attach returns onboarding instructions directly to the calling existing session.

```sh
$ omni session attach --route synthetic-a
$ omni session update --expected-revision 0 --task 'Implement router' --work-state working
$ omni session observe --sequence 1 --runtime-state busy
$ omni session peers
$ omni session update --expected-revision 1 --task 'Waiting for dependency' --work-state blocked
$ omni session update --expected-revision 2 --task 'Implement router' --work-state working
$ omni session update --expected-revision 3 --task 'Router verified' --work-state done
$ omni session leave
```

`get PRESENCE_ID` also shows closed records. `peers` shows attached sessions from
all worktrees sharing the canonical Git common directory; independent nested
repositories are separate. Non-Git locations are explicitly refused. Repository,
worktree and branch refresh on observations, with a `task_context_changed` warning
when the task declaration predates the new Git context. No automatic cross-repo
send prohibition is inferred from the discovery filter.

## Task state and observed runtime

`update` owns only task summary, work state and task revision. It requires the
current expected revision; stale writes fail. Use working/blocked/done/unknown.
`observe` owns only Git context, busy/idle/unknown, last_seen and observation
sequence. Each incarnation needs strictly increasing sequences shared by its event
producers; duplicate and delayed events fail. Closed records cannot be reopened.
Task updates do not forge observations; observations do not change task state.
`last_task_update` is distinct from `last_seen`: recent activity does not make an
old task summary current. Silence means unknown freshness, idle does not mean
finished, and leaving does not mean task success. No liveness threshold is guessed.

## Explicit project-local hook integration

Enable hooks through the harness's documented project-local mechanism. This
feature never edits installed harness configuration. Existing sessions without
hook support can attach and update manually, without restarting their model.
The caller must declare work/scope changes and blocked/done explicitly; hooks
must not infer them from transcripts or runtime state.

Pipe a normalized JSON event to `omni session hook`, for example:

```json
{"event":"checkpoint","session_key":"stable-session","incarnation":"existing-1","route":"explicit-route","sequence":1,"runtime_state":"busy"}
```

Supported events are start/resume/checkpoint/exit. Every event requires a strictly
increasing sequence, including exit. The first
three atomically ensure registration and refresh observations; exit closes the
existing registration only if newer than its last observation. Invalid events
never register a session or change an existing record. Manual `leave` remains a separate explicit
owner action without an observation sequence.
Optional fields are cwd, adapter, harness and role. Binding can fall back to the
environment above, route to `OMNI_SESSION_ROUTE`, and adapter to
`OMNI_SESSION_ADAPTER` (default manual). Unknown fields, including task declarations,
are refused. Empty tasks and changed contexts produce warning codes and `needs_task_update` as data.
Lifecycle registration is idempotent; observation ordering still rejects a replay.
After `receive` upgrades an existing registration to unix, project hooks and
`OMNI_SESSION_ROUTE` may retain their valid original bootstrap route. That metadata
is validated against its explicitly declared adapter, or its representation when
adapter is omitted (opaque string for manual, path/nonce object for unix). It never
changes the active adapter, endpoint or nonce. Malformed metadata and mismatched
explicit adapter/route combinations remain errors; transport changes require
`receive` or an explicit binding-checked route update.
Hooks alone cannot guarantee a session's semantic task declarations are current.
No depth/host fence is cleared or broadened; depth-limited workers retain their
existing state-access restrictions and may receive a visible failure.

## Forwarding and receiver ownership

`omni session send PRESENCE_ID < message.txt` takes only stdin (1–65536 UTF-8 bytes).
A reply uses the same API with the original sender's presence ID. No automatic
retry occurs. A `handoff` response means adapter handoff only: it is not proof of
reading, saving, replying or task success. Peer-authored content grants no authority
to execute commands or widen scope. Do not forward credentials or unintended data.
Sender output and errors exclude the body; only the explicitly selected receiver
gets the message. No transport body is written to the registry.

An injected adapter implements
`guarded_forward(target_record, body, *, expected_incarnation)` and returns
`{ok:true,status:"handoff"}` or `{ok:false,reason:CODE}`. Its live identity guard
and forward must be one operation; get-then-send is insufficient. Registry updates
are locked across handoff, so an adapter must not re-enter registry mutation
before returning. External target replacement must be detected inside the adapter.
Core tests use a guarded fake and prove the architecture, not a live vendor receipt.

Herdr is currently refused with `identity_binding_unsupported`: the installed
prompt API has no verified atomic recipient-incarnation binding. Manual and unknown
adapters return `adapter_unavailable`. These are visible failures, never fake receipt.

## One-message UNIX receiver

An existing session can invoke `omni session receive --timeout 30` as a tool command.
It binds a private local socket, registers a unix route, waits for one intentional
message and returns its body in that receiver's tool output. The existing model
process is never launched or restarted. The command is short-lived, not a daemon.
Subsequent receives reuse the same session binding and install a fresh guarded endpoint. Receive can also switch an existing manual registration to the unix adapter through
an explicit binding-checked route update, preserving the presence ID.

A sender discovers the receiver with peers and sends through the command above.
The live socket handshake pins presence ID, incarnation and endpoint nonce before the
body is sent. When the receiver is not waiting, sending fails visibly; there is no
queue or retry. Endpoint rotation uses an explicit binding-checked `set_route` operation; repeated attach cannot change a route. Receiver output is intentionally message-bearing and belongs to
that session. The receiver must save anything it needs. A native session busy in
another tool cannot be claimed reachable; run receive at an explicit checkpoint.

## Storage and limits

The table and lock live under `$OMNI_LLM_STATE_DIR/sessions/` (default
`~/.local/state/omni-llm/sessions/`, matching the job root). The directory is private;
updates use file locking and atomic replacement. Unsafe/symlink paths, nonprivate
files and malformed tables are refused. Use a canonical nonsymlink state path,
including canonical parents (for example `/private/tmp/...` on macOS).
Only current session records are kept, including closed-incarnation tombstones;
there is no task/event history. Binding keys are hashed, while routes and declared
summaries remain local metadata. Same-UID access is not a hardened auth boundary.
Synchronous handoff holds the table lock; adapters need bounded execution. A
failure after a transport side effect can be ambiguous: inspect the receiver
before deciding whether to retry manually.

Run focused checks with:

```sh
python3 -m unittest -q tests.test_session_router tests.test_session_router_cli
python3 -m py_compile providers/session_router*.py omni_cli.py
```
