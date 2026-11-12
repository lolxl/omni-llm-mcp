# no-opencc
"""Single source of the omni release identity.

``__version__`` is the release; ``CONTRACT_VERSION`` is the version of the interface
surface that ``omni contract`` reports and bumps only on a breaking change to
anything that report lists. The plugin manifests repeat __version__ because each harness
reads its own file; tests/test_plugin_manifests.py pins every copy to it.
"""
__version__ = "0.11.0"
CONTRACT_VERSION = 1
SERVER_NAME = "omni-llm-mcp"
