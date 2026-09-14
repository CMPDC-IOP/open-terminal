"""Tests for descriptor-anchored managed-home repair."""

from __future__ import annotations

import os
import stat

import pytest

from open_terminal.utils.identity_paths import IdentityStoreError, open_home, repair_home


def test_walk_and_repair_stay_within_expected_home(tmp_path):
    root = tmp_path / "homes"
    home = root / "alice"
    nested = home / "notes"
    nested.mkdir(parents=True)
    todo = nested / "todo.txt"
    todo.write_text("buy tea")

    with open_home(root, "alice") as opened:
        entries = list(opened.walk())
    assert [entry.relative_path for entry in entries] == [".", "notes", "notes/todo.txt"]
    assert all(entry.path.startswith(str(home)) for entry in entries)

    repair_home(root, "alice", os.getuid(), os.getgid())
    assert stat.S_IMODE(home.stat().st_mode) == 0o2770
    assert todo.stat().st_uid == os.getuid()


def test_repair_does_not_follow_a_replaced_parent_symlink(tmp_path):
    root = tmp_path / "homes"
    home = root / "alice"
    nested = home / "nested"
    nested.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "secret.txt"
    target.write_text("outside")
    nested.rename(home / "old-nested")
    os.symlink(outside, nested)
    before = target.stat()

    repair_home(root, "alice", os.getuid(), os.getgid())

    after = target.stat()
    assert (after.st_uid, after.st_gid, stat.S_IMODE(after.st_mode)) == (
        before.st_uid,
        before.st_gid,
        stat.S_IMODE(before.st_mode),
    )


def test_walk_allows_hardlinks_but_chown_rejects_them(tmp_path):
    root = tmp_path / "homes"
    home = root / "alice"
    home.mkdir(parents=True)
    first = home / "first"
    first.write_text("shared")
    os.link(first, home / "second")

    with open_home(root, "alice") as opened:
        assert [entry.relative_path for entry in opened.walk()] == [".", "first", "second"]
        for entry in opened.walk():
            if entry.relative_path == "first":
                with pytest.raises(IdentityStoreError, match="hard-linked regular file"):
                    entry.chown(os.getuid(), os.getgid())
                break


def test_chown_rechecks_for_a_hardlink_added_after_open(tmp_path):
    root = tmp_path / "homes"
    home = root / "alice"
    home.mkdir(parents=True)
    first = home / "first"
    first.write_text("not shared yet")

    with open_home(root, "alice") as opened:
        for entry in opened.walk():
            if entry.relative_path == "first":
                os.link(first, home / "outside-link")
                with pytest.raises(IdentityStoreError, match="hard-linked regular file"):
                    entry.chown(os.getuid(), os.getgid())
                break


def test_repeated_walks_do_not_leak_descriptors(tmp_path):
    root = tmp_path / "homes"
    home = root / "alice"
    (home / "nested").mkdir(parents=True)
    (home / "nested" / "file").write_text("safe")
    before = len(os.listdir("/proc/self/fd"))

    with open_home(root, "alice") as opened:
        for _ in range(20):
            list(opened.walk())

    assert len(os.listdir("/proc/self/fd")) == before


def test_create_validates_scope_and_rejects_root_symlink(tmp_path):
    actual_root = tmp_path / "actual-homes"
    actual_root.mkdir()
    configured_root = tmp_path / "homes"
    os.symlink(actual_root, configured_root)

    with pytest.raises(IdentityStoreError, match="home-root component"):
        with open_home(configured_root, "alice", create=True):
            pass

    with pytest.raises(IdentityStoreError, match="invalid username"):
        with open_home(actual_root, "../alice", create=True):
            pass
    with open_home(actual_root, "alice", create=True) as opened:
        assert opened.path == actual_root / "alice"


def test_rejects_group_or_world_writable_final_home_root(tmp_path):
    root = tmp_path / "homes"
    root.mkdir()
    root.chmod(0o777)

    with pytest.raises(IdentityStoreError, match="group- or world-writable"):
        with open_home(root, "alice", create=True):
            pass
