"""Admission and lifetime contracts; resource values are controlled fixtures."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from open_terminal.execution import manager as module
from open_terminal.execution.policy import ExecutionPolicy, Limits


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    unit = Limits(100, 4096, 4)
    policy = ExecutionPolicy(
        tmp_path, unit, unit, unit * 4, unit * 2, 6, 3, cleanup_timeout=0.01
    )
    tree = Mock()
    tree.create_task.side_effect = lambda owner, task: tmp_path / owner / task
    tree.is_empty.return_value = True
    manager = module.Manager(policy, tree)
    monkeypatch.setattr(
        "pwd.getpwnam",
        lambda name: SimpleNamespace(
            pw_uid=10000 + int(name),
            pw_gid=10000 + int(name),
            pw_dir=f"/home/{name}",
            pw_name=name,
        ),
    )
    return manager


class FakeLaunch:
    def __init__(self, manager, identity, path, argv, uid, gid, cwd, environment):
        self.manager, self.identity, self.path = manager, identity, path
        self.environment = environment


@pytest.fixture
def fake_launch(monkeypatch):
    monkeypatch.setattr(module, "Launch", FakeLaunch)


def test_one_user_has_one_aggregate_reservation(runtime, fake_launch):
    a = runtime.prepare("full-owner-a", "1", ["true"])
    b = runtime.prepare("full-owner-a", "1", ["true"])
    runtime.prepare("full-owner-a", "1", ["true"])
    assert a.path.parent == b.path.parent
    assert len(runtime._owners) == 1
    with pytest.raises(HTTPException) as error:
        runtime.prepare("full-owner-a", "1", ["true"])
    assert error.value.status_code == 429
    # Another user's full reservation still fits the separate compute pool.
    runtime.prepare("full-owner-b", "2", ["true"])
    with pytest.raises(HTTPException) as error:
        runtime.prepare("full-owner-c", "3", ["true"])
    assert error.value.status_code == 429
    assert error.value.headers == {"Retry-After": "1"}


def test_last_task_releases_user_reservation(runtime, fake_launch):
    a = runtime.prepare("a", "1", ["true"])
    b = runtime.prepare("a", "1", ["true"])
    runtime.prepare("b", "2", ["true"])
    runtime._release(*a.identity)
    with pytest.raises(HTTPException):
        runtime.prepare("c", "3", ["true"])
    runtime._release(*b.identity)
    runtime.prepare("c", "3", ["true"])


def test_uid_alias_and_missing_owner_rejected(runtime, fake_launch):
    first = runtime.prepare("abcdefgh-one", "1", ["true"])
    runtime._release(*first.identity)
    with pytest.raises(HTTPException) as error:
        runtime.prepare("abcdefgh-two", "1", ["true"])
    assert error.value.status_code == 409
    with pytest.raises(HTTPException) as error:
        runtime.prepare("", "2", ["true"])
    assert error.value.status_code == 400


def test_root_or_service_identity_rejected(runtime, fake_launch, monkeypatch):
    monkeypatch.setattr("pwd.getpwnam", lambda _: SimpleNamespace(pw_uid=0, pw_gid=0))
    with pytest.raises(HTTPException) as error:
        runtime.prepare("a", "root", ["true"])
    assert error.value.status_code == 403


def test_user_environment_excludes_service_credentials(
    runtime, fake_launch, monkeypatch
):
    monkeypatch.setenv("OPEN_TERMINAL_API_KEY", "service-only-secret")
    launch = runtime.prepare("a", "1", ["true"], env={"CUSTOM": "ok", "HOME": "/wrong"})
    assert "OPEN_TERMINAL_API_KEY" not in launch.environment
    assert launch.environment["HOME"] == "/home/1"
    assert launch.environment["CUSTOM"] == "ok"


def test_failed_prepare_returns_reservation_only_after_empty(runtime, monkeypatch):
    monkeypatch.setattr(module, "Launch", Mock(side_effect=ValueError("invalid")))
    with pytest.raises(ValueError):
        runtime.prepare("a", "1", ["bad"])
    assert not runtime._tasks
    runtime.tree.remove_task.assert_called_once()
    runtime.tree.is_empty.return_value = False
    with pytest.raises(ValueError):
        runtime.prepare("a", "1", ["bad"])
    assert len(runtime._tasks) == 1


def _real_launch(runtime, tmp_path):
    runtime.tree.create_task.side_effect = lambda owner, task: tmp_path
    return runtime.prepare("a", "1", ["/bin/true"])


def test_launch_keeps_user_env_out_of_privileged_interpreter(runtime, tmp_path):
    runtime.tree.create_task.side_effect = lambda owner, task: tmp_path
    launch = runtime.prepare(
        "a", "1", ["/bin/true"], env={"LD_PRELOAD": "/user/library.so"}
    )
    try:
        assert "LD_PRELOAD" not in launch.kwargs["env"]
        assert launch.argv[1] == "-I"
        assert Path(launch.argv[2]).name == "launcher.py"
        assert launch.kwargs["cwd"] == "/"
        assert len(launch.kwargs["pass_fds"]) == 3
    finally:
        launch.abort()


def test_cleanup_retains_budget_until_descendants_are_gone(runtime, tmp_path):
    launch = _real_launch(runtime, tmp_path)
    runtime.tree.is_empty.return_value = False
    with pytest.raises(RuntimeError, match="reservation retained"):
        launch._cleanup()
    assert len(runtime._tasks) == 1
    runtime.tree.remove_task.assert_not_called()
    runtime.tree.is_empty.return_value = True
    launch._cleanup()
    assert not runtime._tasks
    assert not runtime._owners


def test_start_error_kills_group_and_returns_reservation(runtime, tmp_path):
    import os

    launch = _real_launch(runtime, tmp_path)
    os.write(launch._status_write, b'{"error":"cannot enter cgroup"}')
    process = Mock()
    with pytest.raises(RuntimeError, match="cannot enter cgroup"):
        launch.started(process)
    runtime.tree.kill.assert_called_once()
    assert not runtime._tasks
    assert not launch._fds


def test_unknown_mode_never_falls_back(monkeypatch):
    monkeypatch.setenv("OPEN_TERMINAL_EXECUTION_MODE", "cgroups-typo")
    with pytest.raises(ValueError):
        module.enabled()


def test_initialize_rejects_unprivileged_service(monkeypatch):
    from open_terminal import env

    monkeypatch.setattr(module, "_manager", None)
    monkeypatch.setenv("OPEN_TERMINAL_EXECUTION_MODE", "cgroup")
    monkeypatch.setattr(env, "MULTI_USER", True)
    monkeypatch.setattr(module.os, "geteuid", lambda: 1000)
    with pytest.raises(RuntimeError, match="root service"):
        module.initialize()


def test_async_call_remains_responsive_and_cleans_cancelled_result():
    import asyncio
    import threading

    entered, finish = threading.Event(), threading.Event()
    cleaned = []

    def start():
        entered.set()
        assert finish.wait(2)
        return "owned-process"

    async def exercise():
        task = asyncio.create_task(module.async_call(start, cleanup=cleaned.append))
        try:
            await asyncio.wait_for(asyncio.to_thread(entered.wait), 1)
            task.cancel()
            await asyncio.sleep(0.01)
            task.cancel()
            await asyncio.sleep(0.01)
            assert not task.done()
        finally:
            finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
    assert cleaned == ["owned-process"]


def test_interpreter_failure_without_ready_is_not_success(runtime, tmp_path):
    launch = _real_launch(runtime, tmp_path)
    process = Mock()
    process.poll.return_value = 1
    with pytest.raises(RuntimeError, match="before confirming isolation"):
        launch.started(process)
    assert not runtime._tasks
    assert not launch._fds


def test_shutdown_preserves_pending_fds_until_spawn_acknowledges(runtime, tmp_path):
    import os

    launch = _real_launch(runtime, tmp_path)
    descriptors = tuple(launch._fds)
    runtime.shutdown()
    assert len(runtime._tasks) == 1
    for fd in descriptors:
        os.fstat(fd)
    process = Mock()
    process.poll.return_value = None
    with pytest.raises(RuntimeError, match="stopped during launch"):
        launch.started(process)
    process.kill.assert_called_once()
    assert not runtime._tasks
    assert not launch._fds


def test_shutdown_rejects_new_admission(runtime, fake_launch):
    runtime.shutdown()
    with pytest.raises(HTTPException) as error:
        runtime.prepare("a", "1", ["true"])
    assert error.value.status_code == 503
    runtime.tree.create_task.assert_not_called()


def test_managed_spawn_rejects_execution_overrides(runtime, tmp_path, monkeypatch):
    runtime.tree.create_task.side_effect = lambda owner, task: tmp_path
    monkeypatch.setattr(module, "_runtime", lambda: runtime)
    popen = Mock()
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    with pytest.raises(ValueError, match="unsupported managed launch"):
        module.spawn("a", "1", ["true"], executable="/bin/sh")
    popen.assert_not_called()
    assert not runtime._tasks


def test_concurrent_admission_does_not_exceed_user_limit(runtime, fake_launch):
    from concurrent.futures import ThreadPoolExecutor

    def attempt(_):
        try:
            runtime.prepare("a", "1", ["true"])
            return True
        except HTTPException as error:
            assert error.status_code == 429
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        admitted = list(pool.map(attempt, range(16)))
    assert sum(admitted) == runtime.policy.max_user_tasks


def test_busy_management_worker_does_not_block_file_executor(monkeypatch):
    import asyncio
    import threading
    from concurrent.futures import ThreadPoolExecutor

    entered, finish = threading.Event(), threading.Event()

    def launch():
        entered.set()
        assert finish.wait(2)

    async def exercise():
        pending = asyncio.create_task(module.async_call(launch))
        try:
            await asyncio.wait_for(asyncio.to_thread(entered.wait), 1)
            assert (
                await asyncio.wait_for(asyncio.to_thread(lambda: "file I/O"), 1)
                == "file I/O"
            )
        finally:
            finish.set()
            await pending

    with ThreadPoolExecutor(max_workers=1) as workers:
        monkeypatch.setattr(module, "_management_workers", workers)
        asyncio.run(exercise())
