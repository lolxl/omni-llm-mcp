# Account selection

This document specifies the shared Codex, Claude and Grok account selector and
its consult, probe, synchronous execution and detached-job consumers. Account
lists have no two-account limit. Cursor retains its existing consult profile ring.

Configure independent directories and an ordered list for each provider:

| Provider | Registry variables | Child environment | Ordered list example |
| --- | --- | --- | --- |
| Codex | `CODEX_HOME_A`, `CODEX_HOME_B`, `CODEX_HOME_C`, ... | `CODEX_HOME` | `OMNI_LLM_PROFILES_CODEX=a,b,c` |
| Claude | `CLAUDE_CONFIG_DIR_A`, `CLAUDE_CONFIG_DIR_B`, `CLAUDE_CONFIG_DIR_C`, ... | `CLAUDE_CONFIG_DIR` | `OMNI_LLM_PROFILES_CLAUDE=a,b,c` |
| Grok | `GROK_HOME_A`, `GROK_HOME_B`, `GROK_HOME_C`, ... | `GROK_HOME` | `OMNI_LLM_PROFILES_GROK=a,b,c` |

These are examples, not maximum pool sizes. Each directory must satisfy the
existing profile-path admission rules: a dedicated `~/.<provider>-<label>` home
or an admitted directory outside the protected home tree. Launch/login with the
provider's native environment variable; shell aliases do not apply to subprocesses.
Login, authentication, configuration and session state belong to each directory.
Do not copy or symlink authentication, configuration or sessions between accounts.
omni does not copy or synchronize configuration files between homes. Shared
settings consist of explicit omni argv overrides (model, effort, sandbox, etc.)
and inherited non-authentication UI/runtime environment settings. For example,
the native allowlisted `GROK_CONFIG` UI/model-effort overlay and standard transport
proxy settings remain available; they do not select a sibling account's home.
Managed account children and quota probes scrub sibling registry variables and
inherited vendor API-key, token and endpoint overrides without changing the parent
environment. Grok's external-auth command, chat proxy/WebSocket and model/API
endpoint overrides are included in that scrub policy. An explicit profile remains
the caller's account choice. When the provider's ring variable is set, an explicit
non-default profile is auth-isolated exactly like a managed child, whether the
request is a consult, a synchronous agent, a detached job or a resume: the child
copy drops the inherited vendor credentials, endpoint overrides and sibling
registrations, the account's own native home applies, and jobs persist the
isolation. With no ring configured, an explicit profile keeps its legacy
environment, including inherited API keys.

`omni profiles add <provider> <label> --exec --persist` prepares the independent
directory, runs the vendor login in a foreground terminal and prints registration
instructions. Login uses `codex login`, `claude auth login`, or `grok login` with
the matching home variable. Browser account selection and authorization remain
interactive. `--persist` updates only the existing local profiles environment
file, not live seats or harness configuration. `profiles list` and `check` include
Grok; `check --live` tests the specified account, not a fallback account.

A non-default `account_profile` pins that account. Without a configured ring,
existing execution behavior remains. `auto` uses the existing opt-in registry
discovery. Rings use priority order, with no rotating pointer. Each candidate is
visited at most once per request, within one total timeout. Duplicate canonical
homes do not become extra attempts. Missing profiles and cooling accounts are
skipped. An exhausted ring returns `account_exhausted` with only validated labels
and typed reasons. Homes remain internal execution pins, not public diagnostics.

Codex quota comes from the existing profile/home-scoped TTL-cached RPC. Exhausted
primary or secondary windows and unknown quota are skipped; unknown is never
declared healthy. `OMNI_LLM_QUOTA_PROBE=0` explicitly disables probing and allows
unverified Codex candidates; cooling accounts remain skipped.

Claude and Grok have no verified numeric quota oracle in this implementation.
Their eligibility uses admitted homes, known cooldown and runtime quota/auth
failures. Managed results expose `quota_verified: false`; Grok registry entries
show `quota_unknown`, not an invented percentage or a promise of remaining quota.
A managed Claude/Grok rate limit whose failure text carries a vendor reset time
cools that account until the reset (ledger source `vendor_reset`); otherwise the
default cooldown applies.
A newly selected account can still fail at runtime. Existing half-open health
recovery remains available. Adding Grok to the registry does not add it to
`probe_covered_roads`: that set describes actual quota oracle coverage, which
remains Codex only.

Codex, Claude and Grok never re-run a failed attempt within the same request:
not on another account, not on the same account after a backoff, and not on a
later seat of a fallback chain. (The other providers keep their bounded backoff
and their chain fall-through.) Codex used to do all three after a definite
startup quota or authentication failure, judged from the child's own JSONL
stream; that stream is written by the child (and anything it ran), so it cannot
prove that no side effect happened, and the automatic re-run was removed. Any
Codex failure now ends the request exactly as it does for Claude and Grok,
whether the account is managed by a ring, pinned by a profile, or the default
with no ring, and including a failure before the child starts (a missing binary
or a fence refusal): one spawn at most, the failure envelope with its typed
reason and `account_retry_safe: false`. A quota (`rate_limit`), `auth_error` or
`account_unavailable` failure also cools the account (source `account_startup`
when no tool activity was seen), so the next request selects the next account;
any other failure only follows the health crash streak and does not cool the
account. The classifier still reads a
complete known-event JSONL stream with a structured error code, or a definite
quota/auth message inside a structurally error-typed event (such as Codex 0.154
error and turn.failed), never arbitrary stderr or assistant messages, to name the
failure reason and the cooldown source. Accepted codes are
`rate_limit_exceeded`, `usage_limit_reached`, `quota_exceeded`,
`authentication_failed`, `unauthorized`, and `account_unavailable`. After tool
activity, a quota failure stops the job and flags the account for every provider;
a new request skips it. A fallback chain does not continue to a later seat
(another Codex seat or another provider) after any Codex failure, whatever
`account_retry_safe` says; the one exception is `account_exhausted`, returned
before any task child was started because no ring account was eligible (quota
probing may still have run `codex app-server` for each account). Claude and Grok
stop on tool activity or on `account_retry_safe: false`. Generic backoff must not
bypass this boundary; a quota-probe process started after a no-ring rate limit
is not a re-run of the task.

Detached jobs persist a concrete profile and canonical home before supervisor
spawn, and update that pin before launching the selected account. If a pending
Claude/Grok account becomes cooling before execution, the next eligible account
becomes the persisted job, attestation and resume pin. This pre-execution selection
does not replay a failed attempt. A Codex pre-execution reselection updates the
pin the same way; a Codex failure after the spawn does not switch accounts.
Claude/Grok
jobs persist `account_home`, managed/auth-isolated provenance, candidates and
deadline; existing Codex fields remain compatible. Managed jobs
persist an absolute wall-clock deadline including supervisor startup delay; the
runner converts the remaining budget to a monotonic attempt deadline. Expired
persisted jobs cannot start a backend after a host reboot. The existing job
lifecycle has no automatic reboot queue replay; explicit resume creates a new
job and budget. Session capture
starts at the selected attempt's stream offset. Resume uses the original profile
and home, skips automatic selection and refuses registry/default-home mapping
drift even if workspace drift is allowed. Legacy jobs without a home pin cannot
prove the original account and are refused for these providers' resume.

Account isolation in the host fences. A vendor child (and its descendants) on
one account cannot read or write another configured account's home, and cannot
rewrite its own home's configuration for later runs:

- Read isolation covers every provider: the fence deny-reads every registered
  `CLAUDE_CONFIG_DIR_<P>`, `CODEX_HOME_<P>` and `GROK_HOME_<P>` home, wherever
  it lives, for every child (including pi and cursor). Only the run's own home
  and the RUNNING provider's default home are exempt; another provider's default
  that lives outside the protected home is denied too. A registered home that
  contains the run's home cannot be denied without removing the run's own
  access, so the spawn is refused (`fence_path_rejected`, naming the label, not
  the path). A Codex consult on a named account takes the Codex agent fence's
  profile rows: the default `~/.codex` is neither readable nor writable, the
  selected home is write-denied whole with only vendor state (sessions, logs,
  caches, sqlite state, `auth.json` for token refresh) re-allowed, and
  `config.toml`, `hooks.json`, `AGENTS*.md` and the agents/skills/plugins/rules
  trees stay read-only.
- A named Claude home gets the default home's write denies, re-rooted: settings,
  settings.local, hooks, plugins, commands, agents, skills, `CLAUDE.md` and
  `.claude.json`. The per-run write allowances (`todos`, `shell-snapshots`,
  `debug`, the session directory under `projects/`) are lexical paths; a
  symlink at any component under the claude root refuses the spawn
  (`fence_path_rejected`) instead of resolving into another home.
- Registered homes must be disjoint. Two homes that are equal across providers,
  or where one contains the other, are refused at resolution with
  `profile_config` naming the two labels, and `profile_check` reports
  `nested_home`. Cursor's config directory (`OMNI_LLM_CURSOR_CONFIG_DIR` and
  `OMNI_LLM_CURSOR_CONFIG_DIR_<P>`) is part of that check against every
  `CLAUDE_CONFIG_DIR_*`, `CODEX_HOME_*` and `GROK_HOME_*` home, in both
  directions. At fence build, an allowed config root (the run's home, or the
  running provider's default root) that contains a registered home, or a cursor
  config root equal to one, refuses the spawn (`fence_path_rejected`, labels
  only). Two labels of the same provider that resolve to the same directory are
  one account listed twice and keep the ring's dedup.
- Row order. Seatbelt is last-match-wins, so the protected-configuration write
  denies (Codex: `config.toml`, `hooks.json`, `AGENTS.md`, `AGENTS.override.md`
  and the agents/skills/plugins/rules trees; Claude: settings, settings.local,
  hooks, plugins, commands, agents, skills, `CLAUDE.md`, `.claude.json`) and the
  sibling-home read denies are emitted a second time as FINAL rows, after every
  allow in the profile (`extra_deny_write_final`, `extra_deny_read_final`), so a
  later state allowance or config-root re-allow cannot reopen them. The earlier
  rows stay. Separately, a protected name that is a symlink, or resolves
  anywhere but its own location, refuses the spawn (`fence_path_rejected`,
  naming the home kind and the protected name, never a path): a link such as
  `settings.json -> debug/payload` would otherwise make a state leaf the write
  path to the configuration the next run executes. A NAMED home refuses any such
  link. The DEFAULT home refuses it only when the target is inside the vendor
  home or inside a root the run can write (workdir, temp roots, state leaves);
  a dotfile-managed `CLAUDE.md` or `AGENTS.md` pointing at a directory the child
  cannot write keeps working, and its target is write-denied by a final row.
  Symlinks nested inside the protected trees (for example a link under
  `skills/`) are not scanned.
- Default-account consult. The Codex consult on the default `~/.codex` builds
  its own policy, and the shared write allowance covers the whole home, so it
  takes the named-account structure for the default home
  (`codex_consult_default_fence`, the same rows as `codex_consult_account_fence`
  without a profile): `~/.codex` is write-denied whole and only the agent
  fence's validated state set is re-allowed (`sessions/`, `shell_snapshots/`,
  `thread-writer-locks/`, `cache/`, `tmp/`, `log/`, existing `.sqlite`/`.db`
  stores with `-wal`/`-shm` siblings, and the named state files including
  `auth.json`). Everything else there is read-only for the child, including
  hook scripts, `*.config.toml` layers and `.tmp/`. The protected
  configuration names also get FINAL write denies (each name as written and as
  it resolves, so an unlink or replacement is refused as well) and the same
  default-home alias rule as the agent lane. Reads are unchanged.
- `models_cache.json` in a Codex home is read-only for every fenced child (consult and agent, default and named accounts) via a final write deny; Codex refetches it when it cannot use or write it.

Codex account state is an accepted residual risk (issue 141). A fenced Codex
child has to write parts of its own account's Codex home to do its job, and
`providers/kernel_sandbox.py` re-allows exactly this set after the whole-home
write deny: the `sessions/` (rollout JSONL), `shell_snapshots/`,
`thread-writer-locks/`, `cache/`, `tmp/` and `log/` trees, existing `.sqlite`
and `.db` stores with their `-wal` and `-shm` siblings, and the state files
`auth.json` (rewritten on token refresh), `history.jsonl`, `session_index.jsonl`,
`version.json`, `installation_id` and the two migration markers. A child steered
by untrusted content can therefore alter state that later runs of the SAME
account read back: `exec resume` replays prior rollout files as earlier turns,
the cache and `tmp/` helpers feed back as tool state, the SQLite state carries
dynamic tool definitions, and `auth.json` integrity rests on the child leaving
the credential alone. The child must also be able to read `auth.json`, and the
fence has no network rules, so a hijacked child could read the credential and
send it out. Other registered account homes are denied reads; their write
isolation depends on keeping them outside the run's writable roots (for
example, not inside the workdir).
`config.toml`, `hooks.json`, `AGENTS.md`, `AGENTS.override.md`, profile layers,
the agents/skills/plugins/rules trees, `.tmp/` and `models_cache.json` are not
writable.

- The boundary this fence provides is account isolation (reads of other
  registered account homes are denied; their writes are isolated only while
  they sit outside the run's writable roots). It does not promise clean
  subsequent runs within one account, and a resume trusts whatever prior state
  its account home holds.
- After a suspected hijack: stop every codex process using that home. Remove
  the `sessions/` tree and `session_index.jsonl`, the `cache/`, `tmp/` and
  `shell_snapshots/` trees, `history.jsonl`, and every top-level `.sqlite` /
  `.db` store together with its `-wal` and `-shm` files. Then initialize clean
  databases with one trusted, operator-run (unfenced) codex invocation on that
  home before returning to fenced runs, because the fence only admits database
  files that exist at spawn. If credentials are suspect, run
  `CODEX_HOME=<that home> codex login`; treat the credential as exposed, since
  logging in again replaces the local file but may not invalidate a token that
  was already copied. Start a new session; never resume one recorded before the
  suspected hijack. Leave `config.toml`, `AGENTS*.md`, `hooks.json` and
  `installation_id` alone.

Claude agent and job write-lane runs pass `--setting-sources project` and `--read-only`
runs pass `--setting-sources ""` (non-foreman lanes), so
a Claude home that authenticates through user-level settings, for example an
`apiKeyHelper`, works on the consultation lane only. A detached job on such a home
fails with `auth_error` and cools that label. Supporting those homes in the agent
lane is tracked as follow-up issue 136.

Verification: `TMPDIR=/private/tmp/oq python3 -B -m unittest -q tests.test_codex_accounts tests.test_provider_accounts`.
All fixtures use temporary homes and state, mocked quota or local fake binaries;
no vendor login, credentials, live quota or network calls are needed.

Managed subscription authentication isolation is persisted separately from
permission to switch accounts. Resume inherits that isolation even with no ring
currently configured, while preserving its original profile/home and refusing
cross-account replay. Direct synchronous resume with a configured account ring
also scrubs API-key overrides; internal callers can pass original authentication
provenance explicitly. Jobs that originally used no ring retain their API-key
authentication environment on resume.

Jobs also persist `account_home_explicit`. A default Claude account whose native
`CLAUDE_CONFIG_DIR` was unset keeps it unset in the detached child and on resume;
an explicit default or named directory remains explicit. This preserves the
vendor's original default-login lookup rather than treating the canonical home
pin as a request to change its configuration environment. Legacy jobs without
this field retain their previously explicit home pin. The Claude consult and
pinned live-check consumers also honor the selected native override presence,
including a managed default account whose configuration variable was unset.

Successful synchronous Claude/Grok plain output is assistant text, even when it
contains JSON error examples. Structured job and consult output modes provide
machine error events; nonzero process failures retain their failure handling.

Without a ring, normal argv/environment behavior remains. Codex has no generic
backoff retry: every Codex failure, with or without a ring and for a pinned
profile, carries `account_retry_safe: false` and ends the request, so the same
account is not run again either. Claude/Grok managed requests carry the same
terminal no-retry provenance even when the failure has no structured
tool-activity proof. Explicit
and resumed agent quota failures also flag the original account for new requests.

## API-key accounts

A Claude ring label can authenticate with an API key instead of a subscription
login. This is an opt-in last resort that spends credits, so it is off by
default and never part of ordinary rotation. Codex and Grok are unaffected.

- Kind declaration: `OMNI_LLM_API_PROFILES_CLAUDE=<label>[,<label>...]` marks the
  ring labels whose home uses an API key (kind `api`). Every other label is kind
  `subscription`. Do not look for a `CLAUDE_CONFIG_DIR_<LABEL>_KIND` spelling: the
  registry regex would read it as another profile. A malformed, duplicate or
  unknown label, or `default`, returns `profile_config`.
- Default off: `OMNI_LLM_API_FALLBACK_CLAUDE=1` is required for every use of an
  api label. Automatic selection without it skips the label with tried reason
  `api_disabled` and never spawns. An explicit pin (`account_profile=<api label>`
  or `--profile`) or a resume of a job that ran on an api label needs the same
  opt-in: without it the request is refused with `api_disabled` before any child
  is created. A pin bypasses only the cooldown gate below, never the opt-in.
- Selection gate: even when enabled, an api label is used automatically only
  after at least one subscription label before it in the ring was skipped for
  cooldown, and no earlier subscription label was skipped for another reason
  (missing, invalid or duplicate home) or is eligible. Otherwise the tried
  reason is `api_not_permitted`. Startup cooldown does not record whether a
  subscription was cooled by quota or by an authentication failure, so an
  auth-cooled subscription also opens the API path. A detached job persists its
  preselection, but the gate is re-read from current health when the runner
  starts: if an earlier subscription recovered meanwhile, the job runs on the
  earliest eligible account and its persisted account, kind and tried list are
  updated before the spawn; if the gate no longer holds, the api label is refused
  with `api_not_permitted`. The gate is then checked once more as the last step
  before the child process is created (after fence preparation and the workspace
  snapshot), for consults, agents and jobs alike. If an earlier subscription
  recovered during preparation, no api child is spawned: the request returns
  `api_not_permitted` ("earlier subscription account recovered") without
  re-selecting, and the next request picks the subscription. Explicit pins and
  resumes are not subject to the cooldown gate.
- Provenance: managed results (success and failure), every `tried` entry and the
  job state carry `account_kind` (`subscription` or `api`) next to
  `account_profile`. An explicit pin of an api label reports `account_kind: api`.
  Legacy jobs without the field read as `subscription`. Resume refuses with
  `account_home_drift` when the label's current kind differs from the persisted
  kind.
- Spend backstop: every api Claude child (consult, agent and job, including
  resume) gets `--max-budget-usd <v>`. `v` comes from `OMNI_LLM_API_MAX_BUDGET_USD`
  (default `1.00`) and must be a positive decimal, otherwise `profile_config`.
  Subscription children never receive the flag. The flag is a cooperative
  backstop enforced by the vendor CLI, not a security boundary: a hostile child
  that holds the key and has network access can bypass it. The spend boundary is
  the Console workspace spend limit together with no purchased credits and no
  auto-reload (see the Console prerequisites below).
- Error classification, from genuine error envelopes only (a successful answer
  that quotes the text is never an error): "credit balance is too low" becomes
  `credit_exhausted` (see "Credit exhaustion" below). The text is read from `result` or `errors` of a result with `is_error: true`,
  from the `message`, `error` or `errors` of a typed error event, and from a bare
  `{"error": {...}}` body. "not scoped to a workspace" becomes `account_unavailable`. Every other 400 is
  unchanged. Claude never replays within a request. A success never clears a
  health flag raised after its own request began, so a slower earlier request
  cannot erase a newer credit cooldown. A flag never shortens an active cooldown
  either: when a seat is already cooling until a later time, a newer flag with an
  earlier `until` keeps the later `until` and its source, while a later `until`
  extends it. This holds for `omni health flag --until` too; a cooldown ends
  early only by being cleared (`providers.health.clear`, a successful run, or a
  half-open probe). The one exception is a credit hold: no success, including an
  explicit pin or resume that bypasses the cooldown check, clears it. Credit
  evidence is stored in its own ledger field (`credit_hold`: `until` and
  `flagged_at`), apart from the generic `until`/`source`, so a longer rate limit
  neither hides a credit raise nor overwrites an earlier hold.
- Job-lane key delivery: the write lane loads only project settings
  (`--setting-sources project`) and `--read-only` loads none
  (`--setting-sources ""`), so an api home's `apiKeyHelper` would not load.
  For api accounts omni additionally passes `--settings <file>`. The file is
  written per spawn (including resume) into a fresh private 0700 directory,
  created inside a validated base and removed when the run ends. The validated
  base is opened `O_DIRECTORY|O_NOFOLLOW` and its device, inode, owner and mode
  must equal what validation saw (a base swapped in between is refused with
  `fence_path_rejected`); the directory is made relative to that descriptor with
  a random name, checked through a no-follow open of its own, and the settings
  file is created in it `O_CREAT|O_EXCL|O_NOFOLLOW`, mode 0600, relative to the
  directory descriptor. The base is one of two fixed areas, tried in order:
  the state dir's `api-settings` area, then `<real home>/.cache/omni-llm/api-settings`.
  A base is used only after every ancestor from the filesystem root is a
  directory owned by the user or root and not group/other-writable (unless
  sticky and root-owned, like `/tmp`), no component below the first user-owned
  one is a symlink, and the base itself is a real directory owned by the user
  with mode exactly 0700; any mismatch refuses that base (missing components
  are created 0700, one by one, after their parent was judged). On every
  fenced spawn of any provider, per-run directories older than 24 hours in
  whichever base already exists and validates are removed on a best-effort
  basis (omni prefix only, regular files only, symlinks never followed; a base
  is never created for this), which cleans up after a SIGKILL. The file has mode 0600 and
  exactly `{"apiKeyHelper": <helper>}`. The fence carries a
  final-position deny-write for the exact file and for its directory, so the child
  can neither rewrite the file, unlink it, rename over it nor rename the directory.
  Containment is judged against the FINAL composed write set, after the host fence
  has added every root (repository root and Git directories included), immediately
  before wrapping: a spawn is refused with `fence_path_rejected` when any write
  root (or literal write row) contains either base or an ancestor of it, because
  such a root lets the child rename the ancestor and substitute its own tree. The
  earlier selection step is only a pre-filter (it also counts the repository roots
  the host fence is expected to add). This final check and a final deny-write
  (subpath) on both bases apply to every fenced child of every provider, so a run
  of another provider or another job cannot reach a pending settings file either;
  the owning API run keeps read of its own file. The deny-write rows are always
  emitted (they cover only omni's own private areas, so they cannot break an
  ordinary run; when the account home cannot be resolved they are skipped, and
  that is a refusal only while API accounts are configured). The refusal for a
  write root that reaches a base applies only while API-kind accounts are
  configured (`OMNI_LLM_API_PROFILES_CLAUDE` set and non-empty; a malformed
  value counts as configured so the guard fails closed). With no API account
  configured there is no refusal, whatever the state dir or workdir is.
  With API accounts configured, a workdir that
  contains or equals either base or an ancestor of it (for example the home
  directory), or a state dir inside the temp roots every fence admits, refuses
  fenced spawns; keep the state dir outside the child's write roots. If no valid
  base exists, or the composed policy makes one reachable, the spawn
  is refused with `fence_path_rejected`. Residual: removing the API account
  declaration turns the exposure refusal off, so a writable root covering the
  state dir is accepted again for non-API runs, and re-enabling API accounts
  while such a run is still alive is not protected for that run. Keep the state
  dir outside any workdir. The helper is read from `<account home>/settings.json`.
  The home must be the canonical (realpath) selected API home, and
  `settings.json` is opened without following a final symlink, must be a regular
  file owned by the current user and directly inside that home (a symlinked file,
  symlinked home or parent, FIFO or oversized file is refused). The helper must be
  a one-line non-empty command string that starts with a command word and contains
  no key-shaped material (`sk-ant-...`, `sk-...`, `xai-...`, `Bearer ...`, JWT);
  otherwise the spawn is refused with `account_unavailable` naming only the label,
  and nothing is copied. The key-material check is best-effort: shell quoting or
  indirection can hide a key from a pattern match, so it is a guard against honest
  mistakes, not a proof. The helper must reference a secret store (Keychain, a
  password manager, a credential command), never embed a key. The real protection
  is that only the owner can edit the home's `settings.json`: children are
  write-denied there by the fence. omni never reads or writes the key itself, and
  the host fence admits only the written file for reading. The consult lane already
  loads the home's settings and is unchanged.
- Key exposure: the API key is readable by the child it authenticates (inherent
  to CLI authentication: the child runs the helper or reads the credential) and
  by other processes of the same user through the Keychain item or the helper's
  own source. The key is not hidden from same-user code; confine it with a
  dedicated workspace key, a spend limit and no stored payment method.
- Launch-context scrub: every managed or ring-pinned child of Codex, Claude and
  Grok (consult, sync agent, job and resume; both Claude kinds) drops the
  inherited `CLAUDECODE`, `CLAUDE_CODE_ENTRYPOINT`, `CLAUDE_CODE_SESSION_ID`,
  `CLAUDE_CODE_SESSION_ATTENDED`, `CLAUDE_CODE_CHILD_SESSION`,
  `CLAUDE_CODE_MESSAGING_SOCKET`, `CLAUDE_CODE_MESSAGING_TOKEN`,
  `CLAUDE_CODE_EXECPATH` and `CLAUDE_CODE_SSE_PORT`. Other `CLAUDE_CODE_*`
  preferences are kept.

Credit exhaustion: when an api account's credit balance runs out, omni tells
the user and lets them decide; omni never buys credits. Purchased (prepaid)
credits do not come back on their own, so nothing here assumes a monthly refill.

- Hold: a `credit_exhausted` failure on an api account flags it with source
  `credit_exhausted` and holds it until the user clears it. There is no automatic
  expiry (the ledger `until` is set a century ahead), the account is skipped by
  every later request, and half-open probing never reopens it. The hold lives in
  the entry's `credit_hold` field, which a credit failure sets or extends (never
  shortens) even while a longer non-credit flag is active, and which other
  flags and successes never remove or shorten. The seat stays cooling while
  either the generic `until` or the hold is active, and a hold that outlasts a
  rate limit keeps the seat cooling after the rate limit ends. An entry written
  before this field existed (source `credit_exhausted`, no field) is still
  honoured as a hold.
- Optional window: set `OMNI_LLM_CREDIT_COOLDOWN_S` to a positive integer number
  of seconds to retry automatically after that long instead. Use it only for keys
  backed by monthly credits. Any other value (zero, negative, fractional,
  non-numeric, empty) fails closed to the manual hold and is logged once.
  Non-api accounts that report the same text keep a 21600 second window unless
  this variable is set.
- Envelope: the failing request returns reason `credit_exhausted` with a detail
  that names only the account label: "API credits for account '<label>' are
  exhausted. To keep using it, add credits in the Claude Console (Settings >
  Billing) - purchased credits are charged - then run `omni health clear claude
  --profile <label>`. Until then omni skips this account." When a later request
  ends as `account_exhausted` because an api account is held for this reason, its
  `tried` entry keeps `skipped_cooldown` and the envelope detail carries the same
  hint, so the calling agent can ask the user whether to add credits.
- Notification: the first time the hold is raised (decided inside the locked
  ledger update, so overlapping failures notify once; not when a credit hold is
  already active, and not on later skips) omni emits one `api_credit_exhausted` event through
  the notify seam with `provider`, `account_profile` (the label) and an `action`
  text. It carries no path, key or identity. See `OMNI_LLM_NOTIFY_CMD`.
- Clear: `omni health clear <model> [--profile P]` removes the flag for the
  model's pool key (the key `omni health flag` writes) and, when the model is a
  road name (`claude`) or `<road>/<model>`, the bare road key of that profile
  and no other road's. It prints `{"ok": true, "cleared": [<ledger keys>]}` and
  exits 0 even when nothing was flagged. If the ledger lock cannot be taken the
  removal did not happen: it prints `{"ok": false, "reason":
  "health_lock_unavailable"}` and exits 1, so retry. A success never ends this
  hold; only this command (or the optional window) does. For this hold use `omni health clear claude --profile <label>` after
  adding credits; the next request selects the api account again.

Console prerequisites: link API credits to the organization (claude.ai Settings,
Billing, API credits); create the key inside a dedicated workspace with a spend
limit, because a key not scoped to a workspace is rejected. omni never buys
credits and purchased credits are charged. If spend must stay bounded, keep no
payment method, no purchased credits and auto-reload off so exhaustion stops
requests instead of charging.

Limits: omni does not verify how the vendor classifies a `claude -p` run for
billing, so confirm usage in the Console. An auth-cooled subscription opens the
API path, as described above. The budget flag caps one child, not the account.
