"""Creation routes reuse the resource, while each caller reads it independently."""

import asyncio
from types import SimpleNamespace

import nbformat
import pytest
from fastapi import HTTPException
from starlette.datastructures import Headers

from open_terminal import env
from open_terminal.utils import idempotency, notebooks


class Request:
    def __init__(self, **headers):
        self.headers = Headers(headers=headers)

    async def is_disconnected(self):
        return False


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    monkeypatch.setattr(idempotency, "_default", idempotency.Registry())


@pytest.fixture
def main_module(monkeypatch):
    monkeypatch.setattr(env, "API_KEY", "fixture-key")
    from open_terminal import main

    return main


@pytest.fixture
def command_route(monkeypatch, main_module):
    started = 0
    stop = asyncio.Event()

    async def create_runner(*args, **kwargs):
        nonlocal started
        started += 1
        return SimpleNamespace(execution_info=None)

    async def log_process(process):
        await stop.wait()

    async def read_log(*args, **kwargs):
        return [], 0, False

    monkeypatch.setattr(main_module, "create_runner", create_runner)
    monkeypatch.setattr(main_module, "log_process", log_process)
    monkeypatch.setattr(main_module, "read_log", read_log)
    monkeypatch.setattr(
        main_module,
        "get_filesystem",
        lambda request: SimpleNamespace(
            home="/work", username=None, resolve_path=lambda path, **_: path
        ),
    )
    monkeypatch.setattr(main_module, "_processes", {})
    try:
        yield main_module, lambda: started
    finally:
        stop.set()
        for process in main_module._processes.values():
            process.log_task.cancel()


def _command_request(
    main_module, key, command="echo one", user="alice", session="chat"
):
    return Request(
        **{
            "idempotency-key": key,
            "x-user-id": user,
            "x-session-id": session,
        }
    ), main_module.ExecRequest(command=command)


def test_execute_retries_share_one_spawn_and_keep_owner_scopes_independent(
    command_route,
):
    main, count_started = command_route

    async def exercise():
        first_request, first_body = _command_request(main, "same-command")
        second_request, second_body = _command_request(main, "same-command")
        first, second = await asyncio.gather(
            main.execute(first_request, first_body, wait=0, tail=None),
            main.execute(second_request, second_body, wait=0, tail=None),
        )
        assert first["id"] == second["id"]
        assert count_started() == 1

        other_request, other_body = _command_request(main, "same-command", user="bob")
        other = await main.execute(other_request, other_body, wait=0, tail=None)
        assert other["id"] != first["id"]
        assert count_started() == 2

    asyncio.run(exercise())


def test_execute_rejects_changed_parameters_and_reports_deleted_resource(command_route):
    main, _ = command_route

    async def exercise():
        request, body = _command_request(main, "resource-key")
        created = await main.execute(request, body, wait=0, tail=None)

        changed_request, changed_body = _command_request(
            main, "resource-key", command="echo changed"
        )
        with pytest.raises(
            HTTPException, match="different creation parameters"
        ) as conflict:
            await main.execute(changed_request, changed_body, wait=0, tail=None)
        assert conflict.value.status_code == 409

        del main._processes[created["id"]]
        retry_request, retry_body = _command_request(main, "resource-key")
        with pytest.raises(HTTPException, match="no longer exists") as deleted:
            await main.execute(retry_request, retry_body, wait=0, tail=None)
        assert deleted.value.status_code == 410

    asyncio.run(exercise())


def test_managed_stopping_command_pins_its_key_after_log_completion(monkeypatch, main_module):
    starts = 0

    async def create_runner(*args, **kwargs):
        nonlocal starts
        starts += 1
        return SimpleNamespace(execution_info={"state": "stopping"})

    async def complete_log(process):
        return None

    async def read_log(*args, **kwargs):
        return [], 0, False

    monkeypatch.setattr(main_module, "create_runner", create_runner)
    monkeypatch.setattr(main_module, "log_process", complete_log)
    monkeypatch.setattr(main_module, "read_log", read_log)
    monkeypatch.setattr(
        main_module,
        "get_filesystem",
        lambda request: SimpleNamespace(home="/work", username=None, resolve_path=lambda path, **_: path),
    )
    monkeypatch.setattr(main_module, "_processes", {})
    monkeypatch.setattr(idempotency, "_default", idempotency.Registry(ttl=0.001))

    async def exercise():
        first_request, first_body = _command_request(main_module, "managed-key")
        first = await main_module.execute(first_request, first_body, wait=0, tail=None)
        await asyncio.sleep(0.01)
        other_request, other_body = _command_request(main_module, "other-key", user="bob")
        await main_module.execute(other_request, other_body, wait=0, tail=None)
        retry_request, retry_body = _command_request(main_module, "managed-key")
        retry = await main_module.execute(retry_request, retry_body, wait=0, tail=None)
        assert retry["id"] == first["id"]
        assert starts == 2

    asyncio.run(exercise())


def test_terminal_create_retries_share_one_session(monkeypatch, main_module):
    if not hasattr(main_module, "create_terminal"):
        pytest.skip("terminal routes are disabled")
    starts = 0
    sessions = {}

    async def create_terminal_once(request):
        nonlocal starts
        starts += 1
        sessions["terminal-id"] = {"backend": "pty"}
        return {"id": "terminal-id"}

    monkeypatch.setattr(main_module, "_create_terminal_once", create_terminal_once)
    monkeypatch.setattr(main_module, "_terminal_sessions", sessions)
    monkeypatch.setattr(main_module, "_session_is_alive", lambda session: True)
    monkeypatch.setattr(main_module, "get_filesystem", lambda request: SimpleNamespace())

    async def exercise():
        first = await main_module.create_terminal(Request(**{"idempotency-key": "terminal-key"}))
        second = await main_module.create_terminal(Request(**{"idempotency-key": "terminal-key"}))
        assert first["id"] == second["id"] == "terminal-id"
        assert starts == 1

    asyncio.run(exercise())


def test_notebook_create_retries_reuse_the_started_kernel(monkeypatch, tmp_path):
    started = 0

    class Filesystem:
        username = None
        home = str(tmp_path)

        def resolve_path(self, path):
            return str(tmp_path / path)

    class Client:
        def create_kernel_manager(self):
            pass

        async def async_start_new_kernel(self, **kwargs):
            nonlocal started
            started += 1

        async def async_start_new_kernel_client(self):
            pass

        async def _async_cleanup_kernel(self):
            pass

    async def read_notebook(*args, **kwargs):
        return nbformat.writes(nbformat.v4.new_notebook())

    monkeypatch.setattr(notebooks, "_read_notebook", read_notebook)
    monkeypatch.setattr(
        notebooks, "_new_notebook_client", lambda *args, **kwargs: (Client(), None)
    )
    router = notebooks.create_notebooks_router(
        lambda: None, lambda request: Filesystem(), multi_user=False
    )
    endpoint = next(
        route.endpoint for route in router.routes if route.path == "/notebooks"
    )

    async def exercise():
        body = notebooks.CreateSessionRequest(path="note.ipynb")
        first = await endpoint(
            body, Request(**{"idempotency-key": "notebook-key"}), Filesystem()
        )
        second = await endpoint(
            body, Request(**{"idempotency-key": "notebook-key"}), Filesystem()
        )
        assert first.id == second.id
        assert started == 1

    try:
        asyncio.run(exercise())
    finally:
        notebooks._sessions.clear()
        if notebooks._cleanup_task is not None:
            notebooks._cleanup_task.cancel()
            notebooks._cleanup_task = None


def test_execute_queue_failure_allows_same_key_retry(monkeypatch, command_route):
    main, count_started = command_route
    create_runner = main.create_runner
    attempts = 0

    async def initially_busy(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise HTTPException(429, "queue full", headers={"Retry-After": "1"})
        return await create_runner(*args, **kwargs)

    monkeypatch.setattr(main, "create_runner", initially_busy)

    async def exercise():
        request, body = _command_request(main, "retry-queue")
        with pytest.raises(HTTPException) as error:
            await main.execute(request, body, wait=0, tail=None)
        assert error.value.status_code == 429
        assert error.value.headers == {"Retry-After": "1"}
        first = await main.execute(request, body, wait=0, tail=None)
        repeated = await main.execute(request, body, wait=0, tail=None)
        assert first["id"] == repeated["id"]
        assert count_started() == 1
        assert attempts == 2

    asyncio.run(exercise())


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_notebook_failed_start_retries_only_after_confirmed_cleanup(
    monkeypatch, tmp_path, cleanup_fails
):
    attempts = []

    class Client:
        def create_kernel_manager(self):
            pass

        async def async_start_new_kernel(self, **kwargs):
            attempts.append("start")
            raise RuntimeError("kernel failed to start")

        async def _async_cleanup_kernel(self):
            attempts.append("cleanup")
            if cleanup_fails:
                raise RuntimeError("kernel still alive")

    async def read_notebook(*args, **kwargs):
        return nbformat.writes(nbformat.v4.new_notebook())

    fs = SimpleNamespace(
        username=None, home=str(tmp_path),
        resolve_path=lambda path: str(tmp_path / path),
    )
    monkeypatch.setattr(notebooks, "_read_notebook", read_notebook)
    monkeypatch.setattr(notebooks, "_new_notebook_client", lambda *a, **kw: (Client(), None))
    router = notebooks.create_notebooks_router(lambda: None, lambda request: fs, multi_user=False)
    endpoint = next(route.endpoint for route in router.routes if route.path == "/notebooks")

    async def exercise():
        for _ in range(2):
            with pytest.raises(HTTPException) as error:
                await endpoint(
                    notebooks.CreateSessionRequest(path="note.ipynb"),
                    Request(**{"idempotency-key": "retry-kernel"}), fs,
                )
            assert error.value.status_code == 500
        assert attempts == ["start", "cleanup"] * (1 if cleanup_fails else 2)

    asyncio.run(exercise())
