"""Backend contracts tested against a small fake cgroup filesystem, never the host."""

import argparse
import json
import os
from pathlib import Path

import pytest

from open_terminal.execution import cgroups, launcher
from open_terminal.execution.policy import ExecutionPolicy, Limits, load_policy


@pytest.fixture
def policy(tmp_path):
    return ExecutionPolicy(
        cgroup_root=tmp_path / "cgroup" / "deployment",
        service=Limits(100, 4096, 4),
        helpers=Limits(100, 4096, 4),
        compute=Limits(600, 24576, 24),
        user=Limits(300, 12288, 12),
        max_tasks=6,
        max_user_tasks=3,
    )


def policy_dict(policy):
    return {
        **{
            key: vars(getattr(policy, key))
            for key in ("service", "helpers", "compute", "user")
        },
        "cgroup_root": str(policy.cgroup_root),
        "max_tasks": policy.max_tasks,
        "max_user_tasks": policy.max_user_tasks,
    }


def test_policy_requires_explicit_limits_and_rejects_unknown_keys(policy, tmp_path):
    config = tmp_path / "policy.json"
    value = policy_dict(policy)
    config.write_text(json.dumps(value))
    assert load_policy(config) == policy
    value["max_runtime"] = 60
    config.write_text(json.dumps(value))
    assert load_policy(config).max_runtime == 60
    for mutate in (
        lambda d: d.pop("compute"),
        lambda d: d.update(typo=1),
        lambda d: d["user"].update(cpu_millis=True),
        lambda d: d["user"].update(swap_bytes=0),
        lambda d: d["user"].update(pids=0),
        lambda d: d.update(task=vars(Limits(1, 1, 1))),
        *(
            lambda d, key=key: d.update(**{key: 1})
            for key in (
                "start_timeout",
                "cleanup_timeout",
                "terminate_grace",
                "max_queue",
                "max_user_queue",
                "queue_timeout",
            )
        ),
    ):
        value = json.loads(json.dumps(policy_dict(policy)))
        mutate(value)
        config.write_text(json.dumps(value))
        with pytest.raises(ValueError):
            load_policy(config)
    config.write_text('{"max_tasks": 1, "max_tasks": 2}')
    with pytest.raises(ValueError, match="duplicate"):
        load_policy(config)


@pytest.fixture
def fake_tree(policy, monkeypatch):
    mount = policy.cgroup_root.parent
    mount.mkdir()
    mkdir = Path.mkdir
    rmdir = Path.rmdir
    interfaces = {
        "cgroup.controllers": "cpu memory pids",
        "cgroup.subtree_control": "",
        "cgroup.procs": "",
        "cgroup.type": "domain",
        "cgroup.events": "populated 0\nfrozen 0",
        "cgroup.kill": "",
        "cpu.max": "max 100000",
        "memory.max": "max",
        "pids.max": "max",
        "memory.swap.max": "max",
        "memory.oom.group": "0",
        "memory.current": "0",
        "pids.current": "0",
        "memory.events": "oom 0\noom_kill 0",
        "pids.events": "max 0",
        "cpu.stat": "usage_usec 0",
    }

    def make(path, *args, **kwargs):
        existed = path.exists()
        mkdir(path, *args, **kwargs)
        if path.is_relative_to(mount) and not existed:
            for key, value in interfaces.items():
                (path / key).write_text(value)

    def remove(path):
        if path.is_relative_to(mount):
            if any(p.is_dir() for p in path.iterdir()):
                raise OSError("directory contains child cgroups")
            for p in path.iterdir():
                p.unlink()
        rmdir(path)

    writes = []

    def write(path, value):
        assert path.is_relative_to(mount)
        writes.append((path, value))
        if path.name == "cgroup.subtree_control":
            value = value.replace("+", "")
        path.write_text(value)

    monkeypatch.setattr(Path, "mkdir", make)
    monkeypatch.setattr(Path, "rmdir", remove)
    monkeypatch.setattr(cgroups, "_write", write)
    monkeypatch.setattr(cgroups, "_trusted_path", lambda path: None)
    monkeypatch.setattr(cgroups, "_mount_point", lambda path: mount)
    monkeypatch.setattr(cgroups.os, "geteuid", lambda: 0)
    monkeypatch.setattr(cgroups.os, "sched_getaffinity", lambda pid: {0})
    policy.cgroup_root.mkdir()
    total = policy.total
    for filename, value in {
        "cpu.max": f"{total.cpu_millis * 100} 100000",
        "memory.max": str(total.memory_bytes),
        "pids.max": str(total.pids),
        "memory.swap.max": "0",
    }.items():
        (policy.cgroup_root / filename).write_text(value)
    tree = cgroups.CgroupTree(policy)
    return tree, writes


def test_groups_enforce_aggregate_limits_and_guard_lifecycle(fake_tree, policy):
    tree, writes = fake_tree
    tree.initialize()
    task = tree.create_task("abc", "def")
    sibling = tree.create_task("abc", "fed")
    assert task.parent == sibling.parent
    assert (task.parent / "pids.max").read_text() == str(policy.user.pids)
    assert (
        tree.compute_path / "cpu.max"
    ).read_text() == f"{policy.compute.cpu_millis * 100} 100000"
    # New leaf cgroups receive the kernel defaults from the fake filesystem.
    # User limits are inherited from the common owner cgroup, while each task
    # remains an OOM kill unit and retains its own lifecycle interfaces.
    for leaf in (task, sibling):
        assert (leaf / "cpu.max").read_text() == "max 100000"
        assert (leaf / "memory.max").read_text() == "max"
        assert (leaf / "pids.max").read_text() == "max"
        assert (leaf / "memory.swap.max").read_text() == "max"
        assert (leaf / "memory.oom.group").read_text() == "1"
        assert (leaf / "cgroup.kill").exists()
        assert (leaf / "cgroup.events").exists()
    assert (tree.service_path / "memory.oom.group").read_text() == "0"
    for path in (
        tree.service_path,
        tree.helpers_path,
        tree.compute_path,
        task.parent,
    ):
        assert (path / "memory.swap.max").read_text() == "0"
    service_join = writes.index((tree.service_path / "cgroup.procs", str(os.getpid())))
    enable = writes.index((tree.root / "cgroup.subtree_control", "+cpu +memory +pids"))
    assert service_join < enable
    with pytest.raises(cgroups.CgroupError, match="hexadecimal"):
        tree.create_task("../escape", "def")
    with pytest.raises(cgroups.CgroupError, match="unowned"):
        tree.kill(tree.root)
    (task / "cgroup.events").write_text("populated 1\nfrozen 0")
    with pytest.raises(cgroups.CgroupError, match="populated"):
        tree.remove_task(task)
    tree.kill(task)
    assert (task / "cgroup.kill").read_text() == "1"
    (task / "cgroup.events").write_text("populated 0\nfrozen 0")
    assert tree.task_usage(task)["memory_events"]["oom_kill"] == 0
    tree.remove_task(task)
    assert not task.exists()
    tree.remove_task(sibling)


@pytest.mark.parametrize(
    "filename,value",
    [
        ("pids.max", "1"),
        ("memory.max", "max"),
        ("cpu.max", "1000 100000"),
        ("memory.swap.max", "max"),
        ("memory.oom.group", "1"),
    ],
)
def test_startup_rejects_insufficient_or_unbounded_budgets_before_writes(
    fake_tree, filename, value
):
    tree, writes = fake_tree
    (tree.root / filename).write_text(value)
    with pytest.raises(cgroups.CgroupError):
        tree.initialize()
    assert writes == []


def test_startup_checks_visible_ancestors(fake_tree):
    tree, writes = fake_tree
    (tree.root.parent / "pids.max").write_text("1")
    with pytest.raises(cgroups.CgroupError, match="ancestor"):
        tree.initialize()
    assert writes == []


def test_startup_never_adopts_or_kills_existing_computation(fake_tree):
    tree, writes = fake_tree
    tree.compute_path.mkdir()
    (tree.compute_path / "cgroup.events").write_text("populated 1")
    with pytest.raises(cgroups.CgroupError, match="operator cleanup"):
        tree.initialize()
    assert writes == []


def test_startup_rejects_missing_controller_and_readback_failure(fake_tree):
    tree, _ = fake_tree
    (tree.root / "cgroup.controllers").write_text("cpu memory")
    with pytest.raises(cgroups.CgroupError, match="controllers"):
        tree.initialize()


def test_trusted_path_rejects_user_writable_directory(tmp_path):
    # Independent of the UID running this test: user ownership or mode rejects it.
    tmp_path.chmod(0o777)
    with pytest.raises(cgroups.CgroupError, match="root-owned"):
        cgroups._trusted_path(tmp_path)


def test_launcher_joins_before_identity_change_and_exec_without_server_environment(
    monkeypatch, tmp_path
):
    events = []
    args = argparse.Namespace(
        status_fd=4,
        cgroup_fd=5,
        environment_fd=6,
        uid=1234,
        gid=1234,
        cwd="/work",
        helper=False,
        command=["/bin/echo", "ok"],
    )
    monkeypatch.setattr(launcher.os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        launcher.os,
        "set_inheritable",
        lambda fd, inheritable: events.append(("cloexec", fd, inheritable)),
    )
    monkeypatch.setattr(launcher.os, "open", lambda *a, **kw: 7)
    monkeypatch.setattr(
        launcher.os,
        "write",
        lambda fd, data: events.append(("join", fd, data)) or len(data),
    )
    monkeypatch.setattr(launcher.os, "close", lambda fd: None)
    monkeypatch.setattr(launcher, "_read_environment", lambda fd: {"HOME": "/work"})
    monkeypatch.setattr(
        launcher, "_drop_privileges", lambda uid, gid: events.append(("drop", uid, gid))
    )
    monkeypatch.setattr(launcher.os, "chdir", lambda cwd: events.append(("cwd", cwd)))
    monkeypatch.setattr(
        launcher, "_close_descriptors", lambda fd: events.append(("close", fd))
    )
    monkeypatch.setattr(
        launcher.os,
        "execvpe",
        lambda exe, argv, env: events.append(("exec", exe, argv, env)),
    )
    monkeypatch.setattr(launcher.time, "monotonic_ns", lambda: 100_000_000_000)
    monkeypatch.setattr(launcher.time, "time_ns", lambda: 1000_000_000_000)
    launcher.launch(args)
    assert events == [
        ("cloexec", 4, False),
        ("join", 7, b"0"),
        ("drop", 1234, 1234),
        ("cwd", "/work"),
        ("close", 4),
        ("join", 4, b"READY 100000000000 1000000000000\n"),
        ("exec", "/bin/echo", args.command, {"HOME": "/work"}),
    ]


@pytest.mark.parametrize("value", [[], {"x": 1}, {"a=b": "x"}, {"x": "a\0b"}])
def test_launcher_rejects_invalid_environment(value, tmp_path):
    path = tmp_path / "env.json"
    path.write_text(json.dumps(value))
    with path.open("rb") as stream, pytest.raises(ValueError):
        launcher._read_environment(stream.fileno())


def test_launcher_refuses_root_computation():
    with pytest.raises(ValueError, match="non-root"):
        launcher._drop_privileges(0, 0)


@pytest.mark.parametrize("remaining_caps", ["0", "1"])
def test_launcher_clears_all_capability_sets_and_verifies_result(
    monkeypatch, remaining_caps
):
    calls = []

    class Libc:
        def prctl(self, *args):
            calls.append(("prctl", *args))
            return 0

        def capset(self, header, data):
            calls.append(("capset",))
            return 0

    def read(path):
        if path.name == "cap_last_cap":
            return "2"
        return "\n".join(
            [
                *(
                    f"{key}: {remaining_caps}"
                    for key in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")
                ),
                "NoNewPrivs: 1",
            ]
        )

    import resource

    monkeypatch.setattr(launcher.ctypes, "CDLL", lambda *a, **kw: Libc())
    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(
        launcher.os, "sched_setscheduler", lambda *a: calls.append(("scheduler",))
    )
    monkeypatch.setattr(resource, "setrlimit", lambda *a: None)
    monkeypatch.setattr(
        launcher.os, "setgroups", lambda groups: calls.append(("groups", groups))
    )
    monkeypatch.setattr(
        launcher.os, "setresgid", lambda *ids: calls.append(("gid", *ids))
    )
    monkeypatch.setattr(
        launcher.os, "setresuid", lambda *ids: calls.append(("uid", *ids))
    )
    monkeypatch.setattr(launcher.os, "getresgid", lambda: (1234,) * 3)
    monkeypatch.setattr(launcher.os, "getresuid", lambda: (1234,) * 3)
    monkeypatch.setattr(launcher.os, "getgroups", list)
    if remaining_caps == "1":
        with pytest.raises(RuntimeError, match="privilege cleanup"):
            launcher._drop_privileges(1234, 1234)
    else:
        launcher._drop_privileges(1234, 1234)
    uid_index = calls.index(("uid", 1234, 1234, 1234))
    assert calls.index(("prctl", 38, 1, 0, 0, 0)) < uid_index
    for cap in range(3):
        assert calls.index(("prctl", 24, cap, 0, 0, 0)) < uid_index
    assert calls.index(("groups", [])) < uid_index < calls.index(("capset",))


def test_graceful_signal_uses_pidfd_and_rechecks_membership(fake_tree, monkeypatch):
    tree, _ = fake_tree
    tree.initialize()
    path = tree.create_task("a", "b")
    (path / "cgroup.procs").write_text("123\n456\n")
    closed, signalled = [], []

    def open_pid(pid):
        if pid == 456:
            # Simulate this process exiting between enumeration and pidfd open.
            (path / "cgroup.procs").write_text("123\n")
        return pid + 1000

    monkeypatch.setattr(cgroups.os, "pidfd_open", open_pid)
    monkeypatch.setattr(cgroups.os, "close", closed.append)
    monkeypatch.setattr(
        cgroups.signal, "pidfd_send_signal", lambda fd, sig: signalled.append((fd, sig))
    )
    tree.terminate(path)
    assert signalled == [(1123, cgroups.signal.SIGTERM)]
    assert closed == [1123, 1456]
