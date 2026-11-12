# no-opencc
"""User alias file: three runtime retargets over the shipped built-ins.

``<state dir>/aliases.json`` is optional. Absent means every value is the
built-in. ``effective()`` reads the file on every call. A symlink, a
malformed document, or an unknown key refuses the whole file.
"""
import json
import os
import stat

ENV_STATE_DIR = "OMNI_LLM_STATE_DIR"
DEFAULT_STATE_DIR = "~/.local/state/omni-llm"
_FILE_NAME = "aliases.json"


class AliasError(Exception):
    """The alias file cannot be used. ``detail`` names the file."""

    def __init__(self, detail):
        self.detail = detail
        super().__init__(detail)


class Effective(object):
    def __init__(self, values, origins):
        self.values = values
        self.origins = origins

    def get(self, key):
        return self.values[key]

    def rows(self):
        return [
            {"key": key, "value": self.values[key], "origin": self.origins[key]}
            for key in sorted(self.values)
        ]


def state_dir():
    return os.path.abspath(os.path.expanduser(
        os.environ.get(ENV_STATE_DIR) or DEFAULT_STATE_DIR))


def alias_path():
    return os.path.join(state_dir(), _FILE_NAME)


def builtins():
    """Closed key set and the values the release ships."""
    from providers.admission import _BUILTIN_EFFORT
    from providers.config import FAMILIES
    from providers.grok_image_cli import GROK_IMAGE_MODEL
    values = {}
    for name, table in FAMILIES.items():
        for provider, model_id in table:
            values["family.%s.%s" % (name, provider)] = model_id
    values["effort.codex"] = _BUILTIN_EFFORT
    values["image.grok"] = GROK_IMAGE_MODEL
    return values


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _refuse_path(path, detail):
    raise AliasError("%s %s" % (path, detail))


def _load_overrides(path):
    """File object, or ``{}`` when the file is absent. Raises ``AliasError``."""
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return {}
    except OSError as exc:
        _refuse_path(path, "cannot be read (%s)" % (exc.strerror or type(exc).__name__))
    if stat.S_ISLNK(info.st_mode):
        _refuse_path(path, "is a symlink")
    if not stat.S_ISREG(info.st_mode):
        _refuse_path(path, "is malformed")
    if info.st_nlink > 1:
        _refuse_path(path, "is a hard link")
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except OSError as exc:
        _refuse_path(path, "cannot be read (%s)" % (exc.strerror or type(exc).__name__))
    try:
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs)
    except (UnicodeError, ValueError):
        _refuse_path(path, "is malformed")
    if not isinstance(data, dict):
        _refuse_path(path, "is malformed")
    closed = builtins()
    for key, value in data.items():
        if key not in closed:
            _refuse_path(path, "has unknown key %r" % key)
        if not _acceptable_value(value):
            _refuse_path(path, "is malformed")
    return data


def _acceptable_value(value):
    """True for a value both the reader and ``set_alias`` accept."""
    return (isinstance(value, str) and value.strip() != ""
            and not any(ch in value for ch in "\r\n\x00"))


def effective():
    """Built-ins overlaid with the file. Reads the file on every call."""
    path = alias_path()
    overrides = _load_overrides(path)
    values = builtins()
    origins = {key: "built-in" for key in values}
    for key, value in overrides.items():
        values[key] = value
        origins[key] = "file"
    return Effective(values, origins)


def show():
    return effective().rows()


def _write_overrides(overrides):
    from pathlib import Path

    from providers.model_catalog import CatalogError, _atomic_write
    path = alias_path()
    if not overrides:
        try:
            info = os.lstat(path)
        except FileNotFoundError:
            return
        if stat.S_ISLNK(info.st_mode):
            _refuse_path(path, "is a symlink")
        os.unlink(path)
        return
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    payload = (json.dumps(overrides, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    try:
        _atomic_write(Path(path), payload)
    except CatalogError as exc:
        _refuse_path(path, "cannot be written (%s)" % exc.reason)


def set_alias(key, value):
    """Store one override. Unknown keys and a bad file refuse; nothing is inferred."""
    closed = builtins()
    if key not in closed:
        raise AliasError("unknown alias key %r" % key)
    if not _acceptable_value(value):
        raise AliasError("alias value for %r must be a non-empty string" % key)
    overrides = _load_overrides(alias_path())
    overrides[key] = value
    _write_overrides(overrides)


def unset_alias(key):
    if key not in builtins():
        raise AliasError("unknown alias key %r" % key)
    overrides = _load_overrides(alias_path())
    overrides.pop(key, None)
    _write_overrides(overrides)


def admission_warning(key, value):
    """Warning text when the default profile's record would refuse ``value``.

    ``None`` when the record admits it. This does not read the alias file
    and does not block the write.
    """
    import datetime

    from providers import admission, catalog_v2
    from providers.config import DEFAULT_CODEX_MODEL
    if key not in builtins():
        return None
    if key == "effort.codex":
        provider, model, effort = "codex", DEFAULT_CODEX_MODEL, value
    elif key == "image.grok":
        provider, model, effort = "grok", value, None
    else:
        provider = key.split(".")[2]
        provider, model, effort = provider, value, None
    record, reason = catalog_v2.load(provider, "default")
    now = datetime.datetime.now(datetime.timezone.utc)
    if reason:
        decision = {"refusal": reason}
    else:
        decision = admission.decide(
            {"provider": provider, "model": model, "effort": effort}, record, now=now)
    refusal = decision.get("refusal") if isinstance(decision, dict) else "no_catalog"
    if not refusal:
        return None
    return ("warning: %s is not admitted on the default profile (%s); "
            "dispatch will refuse it" % (key, refusal))
