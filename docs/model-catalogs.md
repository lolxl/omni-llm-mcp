# Model catalog maintenance

The catalog records model identifiers, display names, advertised effort values,
source, observation time, CLI version, and whether availability is provider
advertised or unknown. A provider listing is not an inference test or proof of
account entitlement. The bundled snapshot contains sanitized metadata only; it
never contains profile names, home paths, endpoints, credentials, or diagnostics.

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
omni models import --provider claude --profile default --file ./claude-catalog.json
omni models snapshot --output /tmp/model-catalogs.json --profile default
omni models schedule --print --provider codex --profile work --interval 86400
```

`status` and `check` read local data only. Status reports the source, observation
time, age, stale state (48 hours by default), last refresh result, and drift from
the bundled snapshot. Add `--details` to status to include the normalized model
rows and exact IDs. `source` identifies who produced the record (`live` or
`manual`); `origin` identifies where this copy came from (`local` or `bundled`).
If no valid local cache can be read, status can show bundled rows with
`origin=bundled` and a `cache_diagnostic`; that is not a fresh live query. Check
exits 0 when there is no drift, 3 when it finds drift,
and 2 when a usable last-good catalog is missing or invalid.

Refresh starts each provider independently. Each automatic discovery child must
receive verified `seatbelt-fs` metadata from the shared host fence and uses an
empty temporary working directory. On platforms without that verified fence,
automatic discovery fails closed with `discovery_unavailable`; use manual import
to add catalog data there. Cursor, agy, Codex, Claude, and Grok use metadata-only
listing operations; no model prompt or inference request is sent. Automatic
adapter availability describes whether a listing method exists, not whether an
account is entitled to use every model.
Grok's authenticated registry listing remains `availability=unknown`; an
unauthenticated banner is an `auth_error` and leaves last-good data unchanged.
Output and deadlines are capped. A fence refusal,
authentication failure, malformed or incomplete output, pagination loop, timeout,
or output cap leaves that provider's last-good file unchanged. Refresh records an
installed CLI version when the bounded fenced version query succeeds; otherwise
the valid model listing is kept with an unknown version. Refresh exits 2
when any selected provider fails while retaining successful providers.

Caches live under `$OMNI_LLM_STATE_DIR/model-catalogs` (default
`~/.local/state/omni-llm/model-catalogs`). Cache identity includes the selected
profile, the canonical home selected by the provider's effective child
environment, and every supported endpoint binding that survives the child's
account scrubbing and environment allowlist. `OMNI_LLM_CATALOG_ENDPOINT_*` is a
separate metadata-only cache tag; it does not configure a provider child. The
identity is stored as a locally salted keyed fingerprint; the cache and exported
snapshot do not contain the home or endpoint values. Profile-specific caches do
not cross over when the effective child home or endpoint changes.

Manual import accepts a normalized JSON object. Import sets `source` to `manual`
and `availability` to `unknown`; it keeps a supplied observation timestamp but
does not invent one. A minimal example is:

```json
{
  "provider": "grok",
  "models": [
    {"id": "grok-example", "name": "Grok example", "supported_efforts": []}
  ]
}
```

The provider-specific `routable_id` is generated when it is omitted. Catalog
entries are validated and atomically replaced only after the complete import is
valid. Manual import works for disconnected providers and providers without a
currently usable automatic adapter.

## Snapshot promotion and rollback

Snapshot requires an explicit output file and exports validated records while
preserving each provider's original source kind, observation time, and CLI
version. It preserves unselected providers already in the destination. Review the
result before promoting it to `providers/model_catalog.json` on a work branch:

```sh
omni models snapshot --provider cursor --profile default \
  --output providers/model_catalog.json
git diff -- providers/model_catalog.json
```

Run the affected tests and review the source evidence before committing a
promotion. To roll back a committed snapshot, revert that promotion commit. For
an uncommitted manual export, keep the prior file copy and replace the export only
after verifying the backup. Refresh never edits repository files.

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

The cron and systemd examples below are for hosts where the shared host fence
can verify `seatbelt-fs`. On hosts without that fence, including the current
non-Darwin implementation, scheduled refresh fails closed with
`discovery_unavailable`; use `omni models import` for manual catalog data instead.

The equivalent cron command on a supported host is:

```cron
0 3 * * * /absolute/path/python3 /absolute/path/omni_cli.py models refresh --provider all --profile default --timeout 120
```

For systemd on a supported host, use the same command in a user service and add
a timer with the desired interval. Refresh only updates local cache state: it
never installs CLIs or schedules, runs inference, changes defaults or aliases,
or promotes a source snapshot.

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
