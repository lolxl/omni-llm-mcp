# no-opencc
"""``omni contract``: the interfaces this installed omni offers, as one JSON object.

Pure introspection. It reads names out of live objects (the argparse tree, the MCP
tool list, the dispatch-exec constants) so it cannot drift from them, and it spawns
nothing, opens no network connection and creates no directory.
"""
import argparse
import os

from providers import dispatch_exec, dispatch_result
from providers.dispatch import TOOLS
from providers.health import DEFAULT_STATE_DIR, ENV_STATE_DIR
from providers.version import CONTRACT_VERSION, SERVER_NAME, __version__


def subcommand_names(parser):
    """Sorted top-level subcommand names registered on ``parser``."""
    names = set()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            names.update(action.choices)
    return sorted(names)


def state_dir():
    """The effective state dir, resolved as providers.health resolves it. Not created."""
    return os.path.abspath(os.path.expanduser(
        os.environ.get(ENV_STATE_DIR) or DEFAULT_STATE_DIR))


def build_contract(parser):
    return {
        "name": SERVER_NAME,
        "version": __version__,
        "contract_version": CONTRACT_VERSION,
        "cli": {"subcommands": subcommand_names(parser)},
        "mcp": {"server_name": SERVER_NAME,
                "tools": sorted(t["name"] for t in TOOLS)},
        "dispatch_exec": {"result_record_version": dispatch_result.VERSION,
                          "providers": [dispatch_exec.PROVIDER],
                          "exit_codes": dict(dispatch_exec.EXIT_CODES)},
        "state_dir": state_dir(),
    }
