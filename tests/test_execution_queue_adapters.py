"""Queue-aware execution adapters use the async admission APIs."""

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import nbformat
import pytest
from fastapi import HTTPException, Request
from jupyter_client.manager import AsyncKernelManager
from jupyter_client.provisioning.local_provisioner import LocalProvisioner

from open_terminal import execution
from open_terminal.utils import notebooks, runner


def test_create_runner_prepares_async_then_spawns_the_prepared_launch(monkeypatch):
    monkeypatch.setattr(execution, "enabled", lambda: True)
    process = Mock(pid=123)
    launch = SimpleNamespace(abort=Mock())
    observed = {}

    async def prepare_async(owner, username, argv, **kwargs):
        observed["prepare"] = (owner, username, argv, kwargs)
        return launch

    def spawn_prepared(prepared, **kwargs):
        observed["spawn"] = (prepared, kwargs)
        return process

    monkeypatch.setattr(execution, "prepare_async", prepare_async)
    monkeypatch.setattr(execution, "spawn_prepared", spawn_prepared)
    monkeypatch.setattr(execution, "stop", Mock())

    instance = asyncio.run(
        runner.create_runner(
            "echo ready",
            "/home/alice",
            {"SERVICE_SECRET": "must-not-leak", "OMP_NUM_THREADS": "8"},
            run_as_user="alice",
            user_env={"EXPLICIT": "yes"},
            owner="owner-id",
        )
    )
    try:
        owner, username, argv, kwargs = observed["prepare"]
        assert (owner, username, argv) == (
            "owner-id",
            "alice",
            ["/bin/bash", "-c", "echo ready"],
        )
        assert kwargs["cwd"] == "/home/alice"
        assert kwargs["env"]["EXPLICIT"] == "yes"
        assert kwargs["env"]["OMP_NUM_THREADS"] == "8"
        assert "SERVICE_SECRET" not in kwargs["env"]
        prepared, popen_kwargs = observed["spawn"]
        assert prepared is launch
        assert set(popen_kwargs) == {"stdin", "stdout", "stderr"}
    finally:
        instance.close()


def test_notebook_kernel_prestart_prepares_async(monkeypatch):
    monkeypatch.setattr(execution, "enabled", lambda: True)
    launch = SimpleNamespace(argv=["managed-launch"], kwargs={}, abort=Mock())
    prepare_async = Mock()

    async def prepare(owner, username, argv, **kwargs):
        prepare_async(owner, username, argv, **kwargs)
        return launch

    async def base_pre(self, **kwargs):
        return ["python", "-m", "ipykernel_launcher"], {"env": {"CUSTOM": "yes"}}

    monkeypatch.setattr(execution, "prepare_async", prepare)
    monkeypatch.setattr(notebooks, "_transfer_connection_file", lambda *args: None)
    monkeypatch.setattr(AsyncKernelManager, "_async_pre_start_kernel", base_pre)
    manager_class = notebooks._user_kernel_manager_class(
        "alice", "/home/alice", "/home/alice/work", "/tmp/kernel.json", owner="owner-id"
    )
    monkeypatch.setattr(
        manager_class,
        "kernel_spec",
        SimpleNamespace(env={"CUSTOM": "yes"}, metadata={}),
    )
    manager = manager_class()
    manager.provisioner = LocalProvisioner()

    command, kwargs = asyncio.run(manager._async_pre_start_kernel())

    assert command == ["managed-launch"]
    assert kwargs["env"]["CUSTOM"] == "yes"
    assert prepare_async.call_args.args[:2] == ("owner-id", "alice")
    assert prepare_async.call_args.kwargs["cwd"] == "/home/alice/work"
    assert prepare_async.call_args.kwargs["env"]["CUSTOM"] == "yes"


def test_interactive_terminal_uses_async_spawn(monkeypatch):
    from open_terminal import env

    monkeypatch.setattr(env, "API_KEY", "fixture-key")
    monkeypatch.setattr(env, "MULTI_USER", False)
    monkeypatch.setattr(env, "ENABLE_TERMINAL", True)
    from open_terminal import main

    monkeypatch.setattr(execution, "enabled", lambda: True)
    process = Mock(pid=123)
    spawn_async = Mock()
    queue_requests = []

    @contextmanager
    def queue_request(request):
        queue_requests.append(request)
        yield

    async def spawn(owner, username, argv, **kwargs):
        spawn_async(owner, username, argv, **kwargs)
        return process

    monkeypatch.setattr(
        main,
        "get_filesystem",
        lambda request: SimpleNamespace(username="alice", home="/home/alice"),
    )
    monkeypatch.setattr(main, "_terminal_sessions", {})
    monkeypatch.setattr(execution, "queue_request", queue_request)
    monkeypatch.setattr(execution, "spawn_async", spawn)
    monkeypatch.setattr(execution, "stop", Mock())
    request = Request({"type": "http", "headers": [(b"x-user-id", b"owner-id")]})

    result = asyncio.run(main.create_terminal(request))
    try:
        assert spawn_async.call_args.args == ("owner-id", "alice", ["/bin/bash", "-il"])
        assert spawn_async.call_args.kwargs["cwd"] == "/home/alice"
        assert queue_requests == [request]
    finally:
        main._cleanup_session(result["id"])


def test_notebook_queue_admission_error_is_not_converted_to_server_error(
    monkeypatch, tmp_path
):
    class Filesystem:
        username = None
        home = str(tmp_path)

        def resolve_path(self, path):
            return path

        async def read_text(self, path):
            return nbformat.writes(nbformat.v4.new_notebook())

    class Client:
        def create_kernel_manager(self):
            pass

        async def async_start_new_kernel(self, **kwargs):
            raise HTTPException(429, "Compute queue is full")

        async def _async_cleanup_kernel(self):
            pass

    monkeypatch.setattr(
        notebooks, "_new_notebook_client", lambda *args, **kwargs: (Client(), None)
    )
    router = notebooks.create_notebooks_router(Mock(), Mock(), multi_user=False)
    endpoint = next(
        route.endpoint
        for route in router.routes
        if route.path == "/notebooks" and "POST" in route.methods
    )
    request = Request({"type": "http", "headers": []})

    with pytest.raises(HTTPException) as error:
        asyncio.run(
            endpoint(
                notebooks.CreateSessionRequest(path="/notebook.ipynb"),
                request,
                Filesystem(),
            )
        )

    assert error.value.status_code == 429
