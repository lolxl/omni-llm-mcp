"""Second-generation catalog cache.

Refresh writes one file per provider profile. Status, check, and admission
read it. The file name uses the scope fingerprint. The identity key stays
in the catalog root.
"""
from __future__ import annotations

import json

from providers import catalog_record
from providers.model_catalog import (
    CatalogError,
    _atomic_write,
    _provider_lock,
    _scope_fingerprint,
    _state_dir,
)


def _root(create=False):
    root = _state_dir() / "model-catalogs-v2"
    if create:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise CatalogError("cache_path_invalid")
    return root


def _paths(provider, profile, create=False):
    root = _root(create=create)
    fingerprint = _scope_fingerprint(provider, profile, create=create)
    stem = "%s-%s" % (provider, fingerprint)
    return root / (stem + ".json"), root / (stem + ".lock")


def save(provider, profile, record):
    """Atomically write ``record`` for this provider profile. Mode 0600."""
    path, lock_path = _paths(provider, profile, create=True)
    encoded = (json.dumps(record, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":")) + "\n").encode("utf-8")
    with _provider_lock(lock_path):
        _atomic_write(path, encoded, 0o600)


def load(provider, profile):
    """Return ``(record, None)`` or ``(None, reason)``.

    A missing file is ``(None, None)``. Any read or ``validate_strict``
    failure is ``catalog_unreadable``.
    """
    try:
        path, _lock_path = _paths(provider, profile, create=False)
    except CatalogError as exc:
        # No identity key yet means refresh has never named this scope.
        if exc.reason == "cache_identity_unavailable":
            return None, None
        return None, "catalog_unreadable"
    except Exception:
        return None, "catalog_unreadable"
    try:
        if not path.exists() and not path.is_symlink():
            return None, None
        if path.is_symlink() or not path.is_file():
            return None, "catalog_unreadable"
        value = json.loads(path.read_text(encoding="utf-8"))
        record, _warnings = catalog_record.validate_strict(provider, value)
        return record, None
    except Exception:
        return None, "catalog_unreadable"
