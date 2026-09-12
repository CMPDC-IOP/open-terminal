"""Public deadline status remains additive across process execution surfaces."""

import asyncio
from contextlib import suppress
from types import SimpleNamespace

import nbformat
import pytest
from fastapi import HTTPException, Request


class _ManagedRunner:
    pid = 42

    @property
    def execution_info(self):
        return {
            "state": "running",
            "started_at": 100.0,
            "deadline": 160.0,
            "finished_at": None,
            "end_reason": None,
            "max_runtime": 60.0,
        }


def _request() -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/execute",
            "headers": [],
        }
    )


def test_process_responses_include_execution_without_waiting_for_completion(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("OPEN_TERMINAL_API_KEY", "test-key")
    from open_terminal import env

    monkeypatch.setattr(env, "API_KEY", "test-key")
    from open_terminal import main

    async def scenario():
        release = asyncio.Event()

        async def pending_log(_):
            await release.wait()

        async def create(*_args, **_kwargs):
            return _ManagedRunner()

        async def read(*_args, **_kwargs):
            return [], 0, False

        filesystem = SimpleNamespace(
            username=None,
            home=str(tmp_path),
            resolve_path=lambda path, **_: str(tmp_path / path),
        )
        monkeypatch.setattr(main, "get_filesystem", lambda _: filesystem)
        monkeypatch.setattr(main, "create_runner", create)
        monkeypatch.setattr(main, "log_process", pending_log)
        monkeypatch.setattr(main, "read_log", read)
        main._processes.clear()

        legacy = SimpleNamespace(
            id="legacy",
            command="true",
            status="done",
            exit_code=0,
            log_path=None,
            runner=SimpleNamespace(execution_info=None),
        )
        assert "execution" not in main._process_response(legacy)

        response = await main.execute(
            _request(), main.ExecRequest(command="sleep 60"), wait=0, tail=None
        )
        process = main._processes[response["id"]]
        try:
            assert response["status"] == "running"
            assert response["execution"]["deadline"] == 160.0
            assert not process.log_task.done()

            listed = await main.list_processes(_request())
            assert listed == [
                {
                    "id": response["id"],
                    "command": "sleep 60",
                    "status": "running",
                    "exit_code": None,
                    "log_path": process.log_path,
                    "execution": response["execution"],
                }
            ]
            status = await main.get_status(
                response["id"], _request(), wait=0, offset=0, tail=None
            )
            assert status["execution"] == response["execution"]
            assert not process.log_task.done()
        finally:
            process.log_task.cancel()
            with suppress(asyncio.CancelledError):
                await process.log_task
            main._processes.clear()

    asyncio.run(scenario())


def test_expired_notebook_kernel_refuses_new_cell_execution(monkeypatch, tmp_path):
    from open_terminal import execution
    from open_terminal.utils import notebooks

    monkeypatch.setattr(execution, "enabled", lambda: True)
    monkeypatch.setattr(
        execution,
        "describe",
        lambda _: {
            "state": "finished",
            "started_at": 100.0,
            "deadline": 160.0,
            "finished_at": 160.0,
            "end_reason": "timed_out",
            "max_runtime": 60.0,
        },
    )

    class Client:
        km = SimpleNamespace(_execution_process=object())

        async def async_execute_cell(self, *_args):
            pytest.fail("an expired kernel must not receive a new cell")

    async def scenario():
        session_id = "expired"
        filesystem = SimpleNamespace(username="alice", home=str(tmp_path))
        notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("pass")])
        notebooks._sessions[session_id] = notebooks._Session(
            session_id,
            str(tmp_path / "note.ipynb"),
            "note.ipynb",
            notebook,
            Client(),
            user_id="alice",
            context_id="one",
            username="alice",
            home=str(tmp_path),
            runtime_directory=None,
        )
        router = notebooks.create_notebooks_router(
            lambda: None, lambda: filesystem, multi_user=True
        )
        endpoint = next(
            route.endpoint for route in router.routes if route.path.endswith("/execute")
        )
        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": f"/notebooks/{session_id}/execute",
                "headers": [(b"x-user-id", b"alice"), (b"x-session-id", b"one")],
            }
        )
        with pytest.raises(HTTPException) as error:
            await endpoint(
                session_id,
                notebooks.ExecuteCellRequest(cell_index=0),
                request,
                filesystem,
            )
        assert error.value.status_code == 409
        assert "runtime expired" in error.value.detail
        assert session_id in notebooks._sessions
        notebooks._sessions.clear()

    asyncio.run(scenario())
