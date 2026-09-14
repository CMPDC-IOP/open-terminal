"""Focused contract tests for the durable identity registry."""

from __future__ import annotations

import multiprocessing
import os
import threading
import time
from pathlib import Path

import pytest

from open_terminal.utils.identity_store import (
    IdentityConflictError,
    IdentityStore,
    IdentityStoreError,
    initialize_directory,
)


def _hold_operation(path: str, home_root: str, entered, release):
    store = IdentityStore(path=path, home_root=home_root)
    with store.operation():
        entered.set()
        release.wait(5)


@pytest.fixture
def store(tmp_path):
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    registry = home / ".open-terminal"
    registry.mkdir(mode=0o700)
    return IdentityStore(path=registry / "identity-map.json", home_root=home)


def test_operation_serializes_independent_processes(store):
    context = multiprocessing.get_context("spawn")
    entered = context.Event()
    release = context.Event()
    process = context.Process(
        target=_hold_operation,
        args=(str(store.path), str(store.home_root), entered, release),
    )
    process.start()
    waiter_entered = threading.Event()

    def wait_for_operation():
        with store.operation():
            waiter_entered.set()

    waiter = threading.Thread(target=wait_for_operation)
    try:
        assert entered.wait(timeout=5)
        waiter.start()
        assert not waiter_entered.wait(timeout=0.15)
        release.set()
        waiter.join(timeout=5)
        assert waiter_entered.is_set()
    finally:
        release.set()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
    assert process.exitcode == 0


def test_read_only_calls_do_not_create_registry_or_lock(tmp_path, monkeypatch):
    import tempfile

    def reject_write_probe():
        raise AssertionError("read-only lookup must not probe for writable temporary directories")

    monkeypatch.setattr(tempfile, "gettempdir", reject_write_probe)
    store = IdentityStore(
        path=tmp_path / "missing" / "identity-map.json", home_root=tmp_path
    )

    assert store.get("unknown") is None
    assert store.all_records() == {}
    assert store.usernames() == {}
    assert not store.path.parent.exists()


def test_store_instances_do_not_lose_updates_when_map_mtime_is_coarse(store, monkeypatch):
    monkeypatch.setenv("OPEN_TERMINAL_UID_MIN", "60000")
    other = IdentityStore(path=store.path, home_root=store.home_root)
    first = store.allocate("first", "first-user")
    # Make this instance cache the first snapshot at a coarse timestamp.
    os.utime(store.path, (1, 1))
    assert store.get("first") == first
    second = other.allocate("second", "second-user")
    # A one-second-resolution filesystem can now report the cached timestamp
    # for the changed map.  Marking first must retain second's allocation.
    os.utime(store.path, (1, 1))
    store.mark("first", "ready")

    assert first.uid != second.uid
    records = store.all_records()
    assert set(records) == {"first", "second"}
    assert records["first"].state == "ready"


def test_adoption_rejects_duplicate_identity_targets(store, monkeypatch):
    monkeypatch.setenv("OPEN_TERMINAL_UID_MIN", "61000")
    store.allocate("first", "first-user", adopt_uid=61001, adopt_gid=61001)
    with pytest.raises(IdentityConflictError):
        store.allocate("first", "other-user", adopt_uid=61001, adopt_gid=61001)
    with pytest.raises(IdentityConflictError):
        store.allocate("second", "second-user", adopt_uid=61001, adopt_gid=61002)


def test_allocation_starts_after_the_highest_occupied_identifier(store, monkeypatch):
    monkeypatch.setenv("OPEN_TERMINAL_UID_MIN", "62000")
    monkeypatch.setattr(store, "_occupied", lambda _document: ({62005}, {62005}))
    store.allocate("existing", "existing-user", adopt_uid=62005, adopt_gid=62005)

    allocated = store.allocate("new", "new-user")
    assert (allocated.uid, allocated.gid) == (62006, 62006)


def test_invalid_state_and_unsafe_registry_path_fail_closed(store):
    store.allocate("user", "test-user")
    with pytest.raises(IdentityStoreError):
        store.mark("user", "invalid")

    os.chmod(store.path.parent, 0o755)
    with pytest.raises(IdentityStoreError):
        store.allocate("second", "second-user")


def test_initialize_directory_never_repairs_an_existing_unsafe_target(tmp_path):
    target = tmp_path / "registry"
    initialize_directory(target, os.geteuid(), os.getegid())
    os.chmod(target, 0o755)

    with pytest.raises(IdentityStoreError):
        initialize_directory(target, os.geteuid(), os.getegid())
    assert (target.stat().st_mode & 0o777) == 0o755


def test_image_service_account_can_own_the_persisted_home_root(monkeypatch):
    from types import SimpleNamespace
    from open_terminal.utils import identity_store

    monkeypatch.setattr(identity_store.os, "geteuid", lambda: 0)
    monkeypatch.setattr(identity_store.pwd, "getpwnam", lambda name: SimpleNamespace(pw_uid=1234, pw_dir="/home/user"))
    assert identity_store._trusted_owner(1234)
    assert not identity_store._trusted_owner(4321)
