"""Execution surfaces must obtain admission before starting user code."""

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from jupyter_client.manager import AsyncKernelManager
from jupyter_client.provisioning.local_provisioner import LocalProvisioner

from open_terminal import execution
from open_terminal.utils import notebooks, runner


@pytest.fixture
def hard_mode(monkeypatch):
    monkeypatch.setattr(execution, "enabled", lambda: True)


def test_runner_passes_full_owner_and_only_explicit_environment(hard_mode, monkeypatch):
    monkeypatch.setenv("SERVICE_SECRET", "must-not-leak")
    process = Mock(pid=123)
    spawn = Mock(return_value=process)
    stop = Mock()
    monkeypatch.setattr(execution, "spawn", spawn)
    monkeypatch.setattr(execution, "stop", stop)
    monkeypatch.setattr(
        runner.subprocess, "Popen", Mock(side_effect=AssertionError("unmanaged launch"))
    )
    instance = runner.PtyRunner(
        "echo hi",
        "/home/alice",
        {"SERVICE_SECRET": "must-not-leak"},
        run_as_user="alice",
        user_env={"EXPLICIT": "yes"},
        owner="full-owner-identifier",
    )
    try:
        args, kwargs = spawn.call_args
        assert args == (
            "full-owner-identifier",
            "alice",
            ["/bin/bash", "-c", "echo hi"],
        )
        assert kwargs["cwd"] == "/home/alice"
        assert kwargs["env"]["EXPLICIT"] == "yes"
        assert "SERVICE_SECRET" not in kwargs["env"]
        instance.kill()
        stop.assert_called_once_with(process, force=False)
    finally:
        instance.close()


def test_runner_admission_failure_never_falls_back(hard_mode, monkeypatch):
    monkeypatch.setattr(
        execution, "spawn", Mock(side_effect=RuntimeError("quota full"))
    )
    fallback = Mock(side_effect=AssertionError("unmanaged launch"))
    monkeypatch.setattr(runner.subprocess, "Popen", fallback)
    with pytest.raises(RuntimeError, match="quota full"):
        runner.PtyRunner("true", None, {}, run_as_user="alice", owner="alice-id")
    fallback.assert_not_called()


def test_hard_mode_rejects_pipe_fallback(hard_mode, monkeypatch):
    monkeypatch.setattr(runner, "_PTY_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="PTY"):
        asyncio.run(
            runner.create_runner("true", None, {}, run_as_user="alice", owner="owner")
        )
    with pytest.raises(RuntimeError, match="PTY"):
        asyncio.run(runner.PipeRunner("true", None, {}).start())


def test_notebook_admission_and_handshake(hard_mode, monkeypatch):
    process = Mock(pid=123)
    launch = SimpleNamespace(
        argv=["managed-launch"],
        kwargs={"env": {"SAFE": "1"}, "cwd": None, "pass_fds": (7,)},
        started=Mock(),
        abort=Mock(),
    )
    prepare = AsyncMock(return_value=launch)
    stop = Mock()
    monkeypatch.setattr(execution, "prepare_async", prepare)
    monkeypatch.setattr(execution, "stop", stop)
    monkeypatch.setattr(notebooks, "_transfer_connection_file", lambda *args: None)

    async def base_pre(self, **kwargs):
        return ["python", "-m", "ipykernel_launcher"], {
            "env": {"SERVICE_SECRET": "no", "CUSTOM": "yes"}
        }

    async def base_launch(self, kernel_cmd, **kwargs):
        self.provisioner.process = process

    async def base_cleanup(self, **kwargs):
        pass

    monkeypatch.setattr(AsyncKernelManager, "_async_pre_start_kernel", base_pre)
    monkeypatch.setattr(AsyncKernelManager, "_async_launch_kernel", base_launch)
    monkeypatch.setattr(AsyncKernelManager, "_async_cleanup_resources", base_cleanup)
    cls = notebooks._user_kernel_manager_class(
        "alice",
        "/home/alice",
        "/home/alice/work",
        "/tmp/kernel.json",
        owner="full-owner",
    )
    monkeypatch.setattr(
        cls, "kernel_spec", SimpleNamespace(env={"CUSTOM": "yes"}, metadata={})
    )
    manager = cls()
    manager.provisioner = LocalProvisioner()

    async def scenario():
        command, kwargs = await manager._async_pre_start_kernel()
        assert command == launch.argv
        assert kwargs == launch.kwargs
        args, supplied = prepare.call_args
        assert args[:2] == ("full-owner", "alice")
        assert supplied["env"]["CUSTOM"] == "yes"
        assert "SERVICE_SECRET" not in supplied["env"]
        launch.started.assert_not_called()
        await manager._async_launch_kernel(command, **kwargs)
        launch.started.assert_called_once_with(process)
        await manager._async_cleanup_resources()
        stop.assert_called_once_with(process, reason="completed")

    asyncio.run(scenario())


def test_kernel_launch_failure_aborts_reserved_admission(hard_mode, monkeypatch):
    async def fail(self, *args, **kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(AsyncKernelManager, "_async_launch_kernel", fail)
    cls = notebooks._user_kernel_manager_class(
        "alice", "/home/alice", "/home/alice", "/tmp/kernel.json", owner="owner"
    )
    manager = cls()
    manager._execution_launch = SimpleNamespace(abort=Mock())
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(manager._async_launch_kernel(["managed-launch"]))
    manager._execution_launch.abort.assert_called_once_with()


@pytest.fixture
def main_module(monkeypatch):
    from open_terminal import env

    monkeypatch.setattr(env, "API_KEY", "fixture-key")
    monkeypatch.setattr(env, "MULTI_USER", False)
    monkeypatch.setattr(env, "ENABLE_TERMINAL", True)
    from open_terminal import main

    return main


def test_compute_routes_require_owner_before_filesystem_resolution(
    hard_mode, main_module, monkeypatch
):
    from fastapi import HTTPException, Request

    filesystem = Mock(side_effect=AssertionError("must reject before resolving"))
    monkeypatch.setattr(main_module, "get_filesystem", filesystem)
    request = Request({"type": "http", "headers": []})
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            main_module.execute(
                request, main_module.ExecRequest(command="true"), wait=None, tail=None
            )
        )
    assert error.value.status_code == 403
    with pytest.raises(HTTPException) as error:
        asyncio.run(main_module.create_terminal(request))
    assert error.value.status_code == 403
    filesystem.assert_not_called()


def test_execute_routes_full_owner_to_admission(hard_mode, main_module, monkeypatch):
    from fastapi import Request

    filesystem = SimpleNamespace(username="account-name", home="/home/account-name")
    monkeypatch.setattr(main_module, "get_filesystem", lambda request: filesystem)
    observed = {}
    queue_requests = []

    @contextmanager
    def queue_request(request):
        queue_requests.append(request)
        yield

    async def reject(command, cwd, environment, **kwargs):
        observed.update(kwargs)
        observed["environment"] = environment
        raise RuntimeError("admission denied")

    monkeypatch.setattr(main_module, "create_runner", reject)
    monkeypatch.setattr(execution, "queue_request", queue_request)
    monkeypatch.setenv("SERVICE_SECRET", "not-user-env")
    request = Request(
        {"type": "http", "headers": [(b"x-user-id", b"untruncated-owner-id")]}
    )
    with pytest.raises(RuntimeError, match="admission denied"):
        asyncio.run(
            main_module.execute(
                request,
                main_module.ExecRequest(command="true", env={"CUSTOM": "yes"}),
                wait=None,
                tail=None,
            )
        )
    assert observed["owner"] == "untruncated-owner-id"
    assert observed["run_as_user"] == "account-name"
    assert observed["environment"]["CUSTOM"] == "yes"
    assert "SERVICE_SECRET" not in observed["environment"]
    assert queue_requests == [request]


def test_startup_failure_does_not_enter_lifespan(main_module, monkeypatch):
    monkeypatch.setattr(
        execution, "initialize", Mock(side_effect=RuntimeError("invalid policy"))
    )

    async def scenario():
        async with main_module.lifespan(main_module.app):
            pytest.fail("invalid policy allowed startup")

    with pytest.raises(RuntimeError, match="invalid policy"):
        asyncio.run(scenario())


def test_interactive_terminal_uses_managed_launch(hard_mode, main_module, monkeypatch):
    from fastapi import Request

    process = Mock(pid=123)
    monkeypatch.setattr(
        main_module,
        "get_filesystem",
        lambda request: SimpleNamespace(username="alice", home="/home/alice"),
    )
    monkeypatch.setattr(main_module, "_terminal_sessions", {})
    spawn = AsyncMock(return_value=process)
    monkeypatch.setattr(execution, "spawn_async", spawn)
    monkeypatch.setattr(execution, "stop", Mock())
    request = Request({"type": "http", "headers": [(b"x-user-id", b"full-owner-id")]})
    result = asyncio.run(main_module.create_terminal(request))
    try:
        args, kwargs = spawn.call_args
        assert args == ("full-owner-id", "alice", ["/bin/bash", "-il"])
        assert kwargs["cwd"] == "/home/alice"
        assert "start_new_session" not in kwargs
    finally:
        main_module._cleanup_session(result["id"])


def test_delayed_spawn_keeps_loop_responsive_and_cancellation_cleans(
    hard_mode, monkeypatch
):
    import threading

    process = Mock(pid=123)
    release = threading.Event()
    stop = Mock()
    monkeypatch.setattr(execution, "stop", stop)

    async def scenario():
        loop = asyncio.get_running_loop()
        started = asyncio.Event()

        launch = SimpleNamespace(abort=Mock())

        async def prepare_async(*args, **kwargs):
            return launch

        def delayed_spawn_prepared(*args, **kwargs):
            loop.call_soon_threadsafe(started.set)
            if not release.wait(timeout=2):
                raise TimeoutError("event loop could not release the launch worker")
            return process

        monkeypatch.setattr(execution, "prepare_async", prepare_async)
        monkeypatch.setattr(execution, "spawn_prepared", delayed_spawn_prepared)
        task = asyncio.create_task(
            runner.create_runner(
                "true", None, {}, run_as_user="alice", owner="alice-id"
            )
        )
        try:
            await asyncio.wait_for(started.wait(), timeout=1)
            # Request cancellation while its worker still owns an in-flight launch.
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stop.called
        assert all(call.args == (process,) for call in stop.call_args_list)

    asyncio.run(scenario())
