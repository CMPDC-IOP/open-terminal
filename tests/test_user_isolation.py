"""Unit tests for persistent identity mapping and account recovery.

These tests fake the privileged subprocess layer and the passwd/group
databases so they run on any host.  Ownership is tracked in a virtual table
because unprivileged processes cannot chown to arbitrary identifiers.  Real
useradd/chown behaviour is covered by tests/identity_container_smoke.py in a
disposable container.
"""

from __future__ import annotations

import grp
import json
import os
import pwd
import subprocess
import threading
from pathlib import Path

import pytest

from open_terminal.utils import user_isolation
from open_terminal.utils.identity_store import (
    IdentityConflictError,
    IdentityStore,
    set_identity_store,
)
from open_terminal.utils.user_isolation import ProvisioningError, ensure_os_user


class FakeAccount:
    def __init__(self, name, uid, gid, home):
        self.pw_name = name
        self.pw_uid = uid
        self.pw_gid = gid
        self.pw_dir = home


class FakeGroup:
    def __init__(self, name, gid):
        self.gr_name = name
        self.gr_gid = gid


class FakeSystem:
    """In-memory passwd/group plus recorded privileged commands."""

    def __init__(self, home_root: Path):
        self.home_root = home_root
        self.accounts: dict[str, FakeAccount] = {}
        self.groups: dict[str, FakeGroup] = {}
        self.owners: dict[str, tuple[int, int]] = {}
        self.commands: list[list[str]] = []
        self.fail_commands: set[str] = set()
        self.lock = threading.RLock()

    # passwd emulation -------------------------------------------------
    def getpwnam(self, name):
        with self.lock:
            account = self.accounts.get(name)
        if account is None:
            raise KeyError(f"getpwnam(): name not found: {name}")
        return account

    def getpwuid(self, uid):
        with self.lock:
            for account in self.accounts.values():
                if account.pw_uid == uid:
                    return account
        raise KeyError(f"getpwuid(): uid not found: {uid}")

    def getpwall(self):
        with self.lock:
            return list(self.accounts.values())

    # group emulation --------------------------------------------------
    def getgrnam(self, name):
        with self.lock:
            group = self.groups.get(name)
        if group is None:
            raise KeyError(f"getgrnam(): name not found: {name}")
        return group

    def getgrgid(self, gid):
        with self.lock:
            for group in self.groups.values():
                if group.gr_gid == gid:
                    return group
        raise KeyError(f"getgrgid(): gid not found: {gid}")

    def getgrall(self):
        with self.lock:
            return list(self.groups.values())

    # privileged command emulation --------------------------------------
    def run(self, cmd, check=True, capture_output=True):
        with self.lock:
            self.commands.append(list(cmd))
        program = cmd[0]
        if program in self.fail_commands:
            raise subprocess.CalledProcessError(1, cmd)
        if len(cmd) > 2 and cmd[1:3] == ["-m", "open_terminal.utils.identity_paths"]:
            if "chown" in self.fail_commands:
                raise subprocess.CalledProcessError(1, cmd)
            home_root, name, uid, gid = cmd[3:]
            target = str(Path(home_root) / name)
            os.makedirs(target, exist_ok=True)
            self._chown(["chown", "-R", f"{uid}:{gid}", target])
            self._chmod(["chmod", "2770", target])
        elif program == "groupadd":
            self._groupadd(cmd)
        elif program == "useradd":
            self._useradd(cmd)
        elif program == "chown":
            self._chown(cmd)
        elif program == "chmod":
            self._chmod(cmd)
        elif program == "mkdir":
            os.makedirs(cmd[-1], exist_ok=True)
            self.owners[os.path.abspath(cmd[-1])] = self.owners.get(
                os.path.abspath(cmd[-1]), (os.getuid(), os.getgid())
            )
        elif program == "usermod":
            pass
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    def _groupadd(self, cmd):
        gid = int(cmd[cmd.index("-g") + 1])
        name = cmd[-1]
        if name in self.groups or any(g.gr_gid == gid for g in self.groups.values()):
            raise subprocess.CalledProcessError(9, cmd)
        self.groups[name] = FakeGroup(name, gid)

    def _useradd(self, cmd):
        uid = int(cmd[cmd.index("-u") + 1])
        gid = int(cmd[cmd.index("-g") + 1])
        name = cmd[-1]
        if name in self.accounts or any(a.pw_uid == uid for a in self.accounts.values()):
            raise subprocess.CalledProcessError(9, cmd)
        home = str(self.home_root / name)
        if "-m" in cmd:
            os.makedirs(home, exist_ok=True)
            self.owners[home] = (uid, gid)
            os.chmod(home, 0o2770)
        self.accounts[name] = FakeAccount(name, uid, gid, home)

    def _chown(self, cmd):
        uid_text, _, gid_text = cmd[-2].partition(":")
        uid, gid = int(uid_text), int(gid_text)
        target = cmd[-1]
        if not os.path.exists(target):
            raise subprocess.CalledProcessError(1, cmd)
        for dirpath, dirnames, filenames in os.walk(target, followlinks=False):
            for name in [dirpath, *(os.path.join(dirpath, f) for f in filenames)]:
                self.owners[os.path.abspath(name)] = (uid, gid)

    def _chmod(self, cmd):
        mode = int(cmd[-2], 8)
        os.chmod(cmd[-1], mode)

    # virtual ownership -------------------------------------------------
    def set_owner(self, path, uid, gid):
        self.owners[os.path.abspath(str(path))] = (uid, gid)

    def wrapped_stat(self, real_stat):
        def stat(path, *args, **kwargs):
            info = real_stat(path, *args, **kwargs)
            key = os.path.abspath(os.fspath(path))
            owner = self.owners.get(key)
            if owner is not None:
                values = list(info)
                values[4], values[5] = owner
                return os.stat_result(values)
            return info

        return stat


class Fixture:
    def __init__(self, tmp_path: Path):
        self.home = tmp_path / "home"
        self.home.mkdir()
        self.map_path = self.home / ".open-terminal" / "identity-map.json"
        self.system = FakeSystem(self.home)
        self.store = IdentityStore(path=self.map_path, home_root=self.home)
        set_identity_store(self.store)
        self._real_stat = os.stat
        self._originals = []
        patches = [
            (pwd, "getpwnam", self.system.getpwnam),
            (pwd, "getpwuid", self.system.getpwuid),
            (pwd, "getpwall", self.system.getpwall),
            (grp, "getgrnam", self.system.getgrnam),
            (grp, "getgrgid", self.system.getgrgid),
            (grp, "getgrall", self.system.getgrall),
            (user_isolation, "_run_privileged", self.system.run),
            (os, "stat", self.system.wrapped_stat(self._real_stat)),
        ]
        for obj, name, value in patches:
            self._originals.append((obj, name, getattr(obj, name)))
            setattr(obj, name, value)

    def close(self):
        for obj, name, value in self._originals:
            setattr(obj, name, value)
        set_identity_store(None)

    def registry(self):
        with open(self.map_path, encoding="utf-8") as handle:
            return json.load(handle)


@pytest.fixture()
def fixture(tmp_path):
    fx = Fixture(tmp_path)
    yield fx
    fx.close()


def test_concurrent_first_access_allocates_once(fixture):
    results = []
    errors = []

    def worker():
        try:
            results.append(ensure_os_user("user-alpha-1111"))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors, errors
    records = {record.uid: record for record in results}
    assert len(records) == 1
    registry = fixture.registry()
    assert len(registry["users"]) == 1
    assert list(registry["users"].values())[0]["state"] == "ready"
    username = list(registry["users"].values())[0]["username"]
    assert fixture.system.accounts[username].pw_uid == list(records)[0]


def test_distinct_users_get_unique_uids_and_gids(fixture):
    first = ensure_os_user("user-alpha-1111")
    second = ensure_os_user("user-beta-2222")
    third = ensure_os_user("user-gamma-3333")
    uids = {first.uid, second.uid, third.uid}
    gids = {first.gid, second.gid, third.gid}
    assert len(uids) == 3
    assert len(gids) == 3


def test_registry_survives_account_reset_and_access_order(fixture):
    first = ensure_os_user("user-alpha-1111")
    second = ensure_os_user("user-beta-2222")
    # Simulate a container rebuild: accounts vanish, registry and /home persist.
    fixture.system.accounts.clear()
    fixture.system.groups.clear()
    # Access in reverse order.
    again_second = ensure_os_user("user-beta-2222")
    again_first = ensure_os_user("user-alpha-1111")
    assert again_first.uid == first.uid
    assert again_first.gid == first.gid
    assert again_second.uid == second.uid
    assert again_second.gid == second.gid


def test_existing_account_with_incomplete_permissions_is_repaired(fixture):
    record = ensure_os_user("user-alpha-1111")
    home = fixture.home / record.username
    # Corrupt ownership and mode after the account was marked ready.
    fixture.system.set_owner(home, 4242, 4242)
    os.chmod(home, 0o755)
    recovered = ensure_os_user("user-alpha-1111")
    assert recovered.uid == record.uid
    info = os.stat(home)
    assert (info.st_uid, info.st_gid) == (record.uid, record.gid)
    assert (info.st_mode & 0o7777) == 0o2770


def test_interrupted_initialization_is_retried(fixture):
    fixture.system.fail_commands.add("chown")
    with pytest.raises(ProvisioningError):
        ensure_os_user("user-alpha-1111")
    registry = fixture.registry()
    state = list(registry["users"].values())[0]["state"]
    assert state in ("allocated", "provisioning")
    fixture.system.fail_commands.clear()
    record = ensure_os_user("user-alpha-1111")
    assert record.state == "ready"


def test_historical_uid_owned_by_other_account_is_not_reused(fixture):
    # A historical directory owned by a uid that now belongs to another account.
    historical = fixture.home / "histuser"
    historical.mkdir()
    fixture.system.set_owner(historical, 5001, 5001)
    os.chmod(historical, 0o2770)
    fixture.system.accounts["otheracct"] = FakeAccount(
        "otheracct", 5001, 5001, str(fixture.home / "otheracct")
    )
    fixture.system.groups["otheracct"] = FakeGroup("otheracct", 5001)
    record = ensure_os_user("histuser-0001-full")
    assert record.uid != 5001
    info = os.stat(historical)
    assert (info.st_uid, info.st_gid) == (record.uid, record.gid)


def test_registry_account_mismatch_fails_closed(fixture):
    record = ensure_os_user("user-alpha-1111")
    fixture.system.accounts[record.username] = FakeAccount(
        record.username, 987654, record.gid, str(fixture.home / record.username)
    )
    with pytest.raises(IdentityConflictError):
        ensure_os_user("user-alpha-1111")


def test_username_collision_gets_deterministic_disambiguation(fixture):
    first = ensure_os_user("alpha-1234-shared")
    # Different full id that sanitizes to the same 8 characters.
    second = ensure_os_user("alpha-1234-other")
    assert first.username != second.username
    assert first.uid != second.uid
    again = ensure_os_user("alpha-1234-other")
    assert again.username == second.username
    assert again.uid == second.uid


@pytest.mark.parametrize("mode", [0o755, 0o2777, 0o7777])
def test_ready_home_mode_is_repaired(fixture, mode):
    record = ensure_os_user("mode-test-user")
    home = fixture.home / record.username
    os.chmod(home, mode)
    assert ensure_os_user(record.user_id).state == "ready"
    assert os.stat(home).st_mode & 0o7777 == 0o2770


def test_recreated_account_restores_memberships(fixture, monkeypatch):
    record = ensure_os_user("recreated-user")
    fixture.system.accounts.clear()
    fixture.system.groups.clear()
    restored = []
    monkeypatch.setattr(user_isolation, "_prepare_group_memberships", lambda r: restored.append(r.user_id))
    ensure_os_user(record.user_id)
    assert restored == [record.user_id]


def test_existing_account_unexpected_home_fails_before_chown(fixture, tmp_path):
    record = ensure_os_user("bad-home-user")
    outside = tmp_path / "outside"
    outside.mkdir()
    fixture.system.accounts[record.username].pw_dir = str(outside)
    fixture.system.commands.clear()
    with pytest.raises(IdentityConflictError, match="unexpected home"):
        ensure_os_user(record.user_id)
    assert fixture.system.commands == []


def test_ready_registry_is_deep_verified_in_new_process(fixture):
    record = ensure_os_user("deep-verify-user")
    target = fixture.home / record.username / "nested"
    target.mkdir()
    child = target / "file"
    child.write_text("existing data")
    fixture.system.set_owner(child, 4242, 4242)
    user_isolation._VERIFIED_HOMES.clear()
    ensure_os_user(record.user_id)
    assert (os.stat(child).st_uid, os.stat(child).st_gid) == (record.uid, record.gid)


def test_verified_home_fast_path_does_not_walk_tree(fixture, monkeypatch):
    record = ensure_os_user("fast-path-user")
    def unexpected_walk(*args, **kwargs):
        raise AssertionError("verified home must not be scanned on every request")
    monkeypatch.setattr(user_isolation, "open_home", unexpected_walk)
    assert ensure_os_user(record.user_id).state == "ready"


def test_business_identity_cannot_adopt_the_image_service_account(fixture):
    fixture.system.accounts["user"] = FakeAccount("user", 1000, 1000, str(fixture.home / "user"))
    fixture.system.groups["user"] = FakeGroup("user", 1000)
    record = ensure_os_user("user")
    assert record.username != "user"
    assert record.uid != 1000
