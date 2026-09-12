"""Opt-in, fail-closed resource isolation for all user computation.

Keep package startup minimal: the privileged launcher must not import the web
application or configuration before joining its resource group.
"""

__all__ = [
    "async_call",
    "describe",
    "enabled",
    "initialize",
    "prepare",
    "prepare_async",
    "prepare_helper",
    "queue_request",
    "shutdown",
    "spawn",
    "spawn_async",
    "spawn_prepared",
    "stop",
]


def __getattr__(name):
    if name in __all__:
        from . import manager

        return getattr(manager, name)
    raise AttributeError(name)
