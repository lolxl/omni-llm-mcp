# Model catalog maintenance

A v2 catalog record stores model identifiers, display names, advertised effort
values, source (`live`), observation time, CLI version, and whether availability
is provider advertised or unknown. A provider listing is not an inference test
or proof of account entitlement. Records do not contain profile names, home
paths, endpoints, credentials, or diagnostics.

The legacy `omni models [--timeout N]` and MCP `list_models` remain available.
That view still runs the fenced `agy models` command and adds `routable_models`
for explicit `agy/<id>` selection plus a local `catalogs` status object.

## Local operations

```sh
omni models status --json
omni models status --provider cursor --details --json
omni models status --provider cursor --profile work --json
omni models refresh --provider all --profile default --timeout 120 --json
omni models check --provider cursor --profile default --json
omni models schedule --print --provider codex --profile work --interval 86400
```

`status` and `check` read the v2 record only. Status reports, per provider,
`observed_at`, age, stale state (48 hours by default), and the model count.
Add `--details` to list each model's id and name. A missing record has null
freshness and a model count of zero. Check admits the shipped defaults against
that record. It exits 0 when every default is admitted (`effort_required` is
reported and is not a failure) and 2 when any default is refused.

Refresh starts each provider independently and writes only the v2 record.
Success and the process exit follow that write. On macOS, each automatic
discovery child must receive verified `seatbelt-fs` metadata from the shared
host fence and uses an empty temporary working directory. Off macOS, discovery
uses the `metadata-unfenced` posture in the next section. Cursor, agy, Codex,
Claude, and Grok use metadata-only listing operations; no model prompt or
inference request is sent. A listing is not proof that an account may use
every model. Grok's registry listing remains `availability=unknown`. An
unauthenticated grok listing is `catalog_unreadable` and does not replace the
v2 record. Output and deadlines are capped. A fence refusal, malformed or
incomplete output, pagination loop, timeout, or output cap does not replace
that provider's v2 record. Refresh records an installed CLI version when the
bounded fenced version query succeeds; otherwise the valid model listing is
kept with an unknown version. Refresh exits 2 when any selected provider
fails, and it still writes the providers that succeeded.

v2 records live under `$OMNI_LLM_STATE_DIR/model-catalogs-v2` (default
`~/.local/state/omni-llm/model-catalogs-v2`). The identity key stays under
`$OMNI_LLM_STATE_DIR/model-catalogs`. Cache identity includes the selected
profile, the canonical home selected by the provider's effective child
environment, and every supported endpoint binding that survives the child's
account scrubbing and environment allowlist. `OMNI_LLM_CATALOG_ENDPOINT_*` is a
separate metadata-only cache tag; it does not configure a provider child. The
identity is stored as a locally salted keyed fingerprint; the record does not
contain the home or endpoint values. Profile-specific records do not cross
over when the effective child home or endpoint changes. A name of the form
`agy-opus-<version>@<effort>` is refused as `unsupported_model`. Pass the
catalog id as `agy/<id>`.

## Discovery posture off macOS

On macOS, catalog discovery still requires verified `seatbelt-fs` metadata.
Off macOS, discovery runs `metadata-unfenced`: no filesystem containment, the
same posture inference already uses on those hosts. There is no new sandbox.
The child argv is a fixed per-provider metadata command, the environment is
the catalog child environment used today, the working directory is an empty
temporary directory, and the run stays inside the refresh timeout. A discovery
record stores `discovery.posture` (`seatbelt-fs` or `metadata-unfenced`) and
`discovery.platform`. Scheduled refresh starts those metadata CLIs on
non-macOS hosts. Dispatch-time checks, including agy effort capability, still
require `seatbelt-fs` on every platform.

## Optional scheduling

`omni models schedule --print --profile default --interval 86400` prints an
all-provider launchd plist and does not install it. Add `--provider codex` (or
another provider) to pin a road-specific job when only that road has the named
profile. The plist pins an absolute Python and entry path, the selected provider
and profile, an explicit executable-search `PATH`, non-secret profile home/config
and endpoint bindings, a bounded refresh timeout, and a local log path. The plist
retains every registered Codex, Claude, Grok, and Cursor config home as a
canonical absolute path so later profile checks keep the same fence conflict
registry. It also pins the selected provider's canonical effective home; relative
home values therefore keep their meaning if the scheduler starts in another
working directory. Native Claude, Codex, and Grok defaults remain implicit in the
plist; their canonical resolved roots scope the cache without exporting an
explicit home override. Endpoint entries are limited to values the selected
child actually receives and preserve those values exactly. When a supported
endpoint depends on `OMNI_LLM_CHILD_ENV_PASS`, the plist retains only the needed
endpoint variable names in that selector; read-only Claude plugin selectors are
scrubbed. A URL containing a known secret, URL userinfo, a credential-like query,
or a fragment is refused with `endpoint_secret_refused` instead of being rewritten
or printed. One job can be printed per provider/profile pair.

Before installing or loading a printed job, create the parent directories for
`StandardOutPath` and `StandardErrorPath` with permissions that allow the job
user to write logs. `--print` only emits the plist; it does not provision these
directories or install the job.

The plist never stores API keys, access tokens, passwords, or secret URL parts.
For key-only profiles, provide credentials to the scheduled process through the
operator's separate launchd/Keychain environment mechanism; this command does not
persist or claim to configure those secrets. launchd may not have the same
interactive Keychain session as a terminal, so a job can report a named
authentication failure; inspect `omni models status --profile <name>` after each
run.

The cron and systemd examples below run the same metadata CLIs. On macOS the
shared host fence must verify `seatbelt-fs`. Off macOS the job runs
`metadata-unfenced` (no filesystem containment), as described above.

The equivalent cron command on a supported host is:

```cron
0 3 * * * /absolute/path/python3 /absolute/path/omni_cli.py models refresh --provider all --profile default --timeout 120
```

For systemd on a supported host, use the same command in a user service and add
a timer with the desired interval. Refresh only updates local cache state: it
never installs CLIs or schedules, runs inference, or changes defaults.

```ini
# ~/.config/systemd/user/omni-model-catalog.service
[Service]
Type=oneshot
Environment=PATH=/usr/local/bin:/usr/bin:/bin
ExecStart=/absolute/path/python3 /absolute/path/omni_cli.py models refresh --provider all --profile default --timeout 120

# ~/.config/systemd/user/omni-model-catalog.timer
[Timer]
OnCalendar=daily
Persistent=true

[Install]
WantedBy=timers.target
```

## Activating a release

Install the candidate in a separate directory, and keep the old release for
running processes. Run `models refresh` once for each configured ring profile,
in the MCP client launch environment. Run `models check` for each of those
profiles. Switch to the candidate only if every command exits zero.
