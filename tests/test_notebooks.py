import asyncio
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from open_terminal.utils import notebook_files, notebooks
from open_terminal.utils.fs import UserFS

if sys.platform == "linux":
    import pwd
else:
    pwd = None


class FakeFilesystem:
    def __init__(self, home: Path, username: str | None):
        self.home = str(home)
        self.username = username

    def resolve_path(self, path: str) -> str:
        return os.path.normpath(
            path if os.path.isabs(path) else os.path.join(self.home, path)
        )


class FakeClient:
    def create_kernel_manager(self):
        pass

    async def async_start_new_kernel(self, **kwargs):
        self.cwd = kwargs["cwd"]

    async def async_start_new_kernel_client(self):
        pass

    async def async_execute_cell(self, cell, cell_index):
        cell.execution_count = cell_index + 1
        cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text="ok\n")]

    async def _async_cleanup_kernel(self):
        self.cleaned = True


def notebook_text() -> str:
    return nbformat.writes(
        nbformat.v4.new_notebook(
            cells=[nbformat.v4.new_code_cell("print('before')")],
            metadata={
                "kernelspec": {
                    "name": "python3",
                    "display_name": "Python 3",
                    "language": "python",
                }
            },
        )
    )


@pytest.fixture(autouse=True)
def reset_sessions():
    notebooks._sessions.clear()
    yield
    notebooks._sessions.clear()
    if notebooks._cleanup_task is not None:
        notebooks._cleanup_task.cancel()
        notebooks._cleanup_task = None


@pytest.fixture
def multi_user_client(tmp_path, monkeypatch):
    homes = {name: tmp_path / name for name in ("alice", "bob")}
    for home in homes.values():
        home.mkdir()
    filesystems = {name: FakeFilesystem(home, name) for name, home in homes.items()}
    app = FastAPI()

    def verify_api_key(request: Request):
        if request.headers.get("Authorization") != "Bearer test-key":
            raise HTTPException(status_code=401, detail="Invalid API key")

    def get_filesystem(request: Request):
        return filesystems.get(
            request.headers.get("X-User-Id"), FakeFilesystem(homes["alice"], None)
        )

    async def read_notebook(*args, **kwargs):
        return notebook_text()

    async def write_notebook(*args, **kwargs):
        return None

    monkeypatch.setattr(notebooks, "_read_notebook", read_notebook)
    monkeypatch.setattr(notebooks, "_write_notebook", write_notebook)
    monkeypatch.setattr(
        notebooks, "_new_notebook_client", lambda *args, **kwargs: (FakeClient(), None)
    )
    app.include_router(
        notebooks.create_notebooks_router(
            verify_api_key, get_filesystem, multi_user=True
        )
    )
    with TestClient(app) as client:
        yield client, homes


def headers(user="alice", context="one"):
    return {
        "Authorization": "Bearer test-key",
        "X-User-Id": user,
        "X-Session-Id": context,
    }


def test_multi_user_requires_user_and_restricts_notebooks_to_home(multi_user_client):
    client, homes = multi_user_client

    assert (
        client.post(
            "/notebooks",
            json={"path": "note.ipynb"},
            headers={"Authorization": "Bearer test-key"},
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/notebooks", json={"path": "/tmp/note.ipynb"}, headers=headers()
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/notebooks",
            json={"path": str(homes["bob"] / "note.ipynb")},
            headers=headers(),
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/notebooks", json={"path": "bad\u0000.ipynb"}, headers=headers()
        ).status_code
        == 400
    )


def test_sessions_require_the_same_user_and_context_for_every_operation(
    multi_user_client,
):
    client, _ = multi_user_client
    created = client.post(
        "/notebooks", json={"path": "note.ipynb"}, headers=headers()
    ).json()
    session_id = created["id"]

    for method, url, payload in (
        (client.post, f"/notebooks/{session_id}/execute", {"cell_index": 0}),
        (client.get, f"/notebooks/{session_id}", None),
        (client.delete, f"/notebooks/{session_id}", None),
    ):
        response = (
            method(url, json=payload, headers=headers("bob"))
            if payload
            else method(url, headers=headers("bob"))
        )
        assert response.status_code == 404

    for method, url, payload in (
        (client.post, f"/notebooks/{session_id}/execute", {"cell_index": 0}),
        (client.get, f"/notebooks/{session_id}", None),
        (client.delete, f"/notebooks/{session_id}", None),
    ):
        response = (
            method(url, json=payload, headers=headers("alice", "other"))
            if payload
            else method(url, headers=headers("alice", "other"))
        )
        assert response.status_code == 404
    assert (
        client.post(
            f"/notebooks/{session_id}/execute",
            json={"cell_index": 0},
            headers=headers(),
        ).json()["status"]
        == "ok"
    )
    assert client.delete(f"/notebooks/{session_id}", headers=headers()).json() == {
        "status": "stopped"
    }


@pytest.mark.skipif(pwd is None, reason="Notebook descriptor helper is Linux-only")
def test_notebook_helper_rejects_symlinks_and_preserves_permission_bits(
    tmp_path, monkeypatch
):
    account = pwd.getpwuid(os.getuid())
    home = tmp_path / "home"
    home.mkdir()
    notebook = home / "note.ipynb"
    notebook.write_bytes(b"old")
    notebook.chmod(0o660)
    monkeypatch.setattr(
        notebook_files.pwd,
        "getpwnam",
        lambda username: SimpleNamespace(pw_dir=str(home)),
    )

    notebook_files.write_notebook(
        home=str(home), username=account.pw_name, path="note.ipynb", content=b"new"
    )

    assert notebook.read_bytes() == b"new"
    assert stat.S_IMODE(notebook.stat().st_mode) == 0o660
    outside = tmp_path / "outside.ipynb"
    outside.write_text("outside")
    (home / "escape.ipynb").symlink_to(outside)
    with pytest.raises(notebook_files.NotebookFileError):
        notebook_files.read_notebook(
            home=str(home), username=account.pw_name, path="escape.ipynb"
        )
    directory = home / "directory"
    directory.mkdir()
    (home / "directory-link").symlink_to(directory, target_is_directory=True)
    with pytest.raises(notebook_files.NotebookFileError):
        notebook_files.read_notebook(
            home=str(home), username=account.pw_name, path="directory-link/note.ipynb"
        )
    with pytest.raises(notebook_files.NotebookFileError):
        notebook_files.write_notebook(
            home=str(home),
            username=account.pw_name,
            path="directory-link/note.ipynb",
            content=b"changed",
        )
    assert notebook.read_bytes() == b"new"


def test_startup_and_save_failures_cleanup_without_reporting_success(
    multi_user_client, monkeypatch, tmp_path
):
    client, _ = multi_user_client

    class FailingClient(FakeClient):
        async def async_start_new_kernel(self, **kwargs):
            raise RuntimeError("start failed")

    runtime_directory = tmp_path / "runtime"
    runtime_directory.mkdir()
    failing = FailingClient()
    monkeypatch.setattr(
        notebooks,
        "_new_notebook_client",
        lambda *args, **kwargs: (failing, str(runtime_directory)),
    )
    assert (
        client.post(
            "/notebooks", json={"path": "note.ipynb"}, headers=headers()
        ).status_code
        == 500
    )
    assert failing.cleaned is True
    assert not runtime_directory.exists()

    monkeypatch.setattr(
        notebooks, "_new_notebook_client", lambda *args, **kwargs: (FakeClient(), None)
    )

    async def save_fails(*args, **kwargs):
        raise HTTPException(status_code=500, detail="Failed to save notebook")

    monkeypatch.setattr(notebooks, "_write_notebook", save_fails)
    session_id = client.post(
        "/notebooks", json={"path": "note.ipynb"}, headers=headers()
    ).json()["id"]
    response = client.post(
        f"/notebooks/{session_id}/execute", json={"cell_index": 0}, headers=headers()
    )
    assert response.status_code == 500
    assert notebooks._sessions[session_id].busy is False


def test_cancelled_execution_removes_the_session_and_starts_cleanup(tmp_path):
    class BlockingClient(FakeClient):
        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def async_execute_cell(self, cell, cell_index):
            self.started.set()
            await self.release.wait()

    async def exercise():
        filesystem = FakeFilesystem(tmp_path, "alice")
        client = BlockingClient()
        notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("pass")])
        session_id = "cancelled"
        notebooks._sessions[session_id] = notebooks._Session(
            session_id,
            str(tmp_path / "note.ipynb"),
            "note.ipynb",
            notebook,
            client,
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
        task = asyncio.create_task(
            endpoint(
                session_id,
                notebooks.ExecuteCellRequest(cell_index=0),
                request,
                filesystem,
            )
        )
        await client.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)
        assert session_id not in notebooks._sessions
        assert client.cleaned is True

    asyncio.run(exercise())


def test_cancelled_startup_cleans_the_kernel_and_runtime_directory(
    tmp_path, monkeypatch
):
    class BlockingStartClient(FakeClient):
        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def async_start_new_kernel(self, **kwargs):
            self.started.set()
            await self.release.wait()

    async def exercise():
        filesystem = FakeFilesystem(tmp_path, "alice")
        runtime_directory = tmp_path / "runtime"
        runtime_directory.mkdir()
        client = BlockingStartClient()

        async def read_notebook(*args, **kwargs):
            return notebook_text()

        monkeypatch.setattr(notebooks, "_read_notebook", read_notebook)
        monkeypatch.setattr(
            notebooks,
            "_new_notebook_client",
            lambda *args, **kwargs: (client, str(runtime_directory)),
        )
        router = notebooks.create_notebooks_router(
            lambda: None, lambda: filesystem, multi_user=True
        )
        endpoint = next(
            route.endpoint for route in router.routes if route.path == "/notebooks"
        )
        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/notebooks",
                "headers": [(b"x-user-id", b"alice"), (b"x-session-id", b"one")],
            }
        )
        task = asyncio.create_task(
            endpoint(
                notebooks.CreateSessionRequest(path="note.ipynb"), request, filesystem
            )
        )
        await client.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert client.cleaned is True
        assert not runtime_directory.exists()
        assert not notebooks._sessions

    asyncio.run(exercise())


def test_single_user_kernel_executes_and_saves_notebook(tmp_path):
    notebook_path = tmp_path / "single.ipynb"
    notebook_path.write_text(notebook_text())
    app = FastAPI()

    def verify_api_key():
        pass

    def get_filesystem(request: Request):
        return UserFS(home=str(tmp_path))

    app.include_router(
        notebooks.create_notebooks_router(
            verify_api_key, get_filesystem, multi_user=False
        )
    )
    with TestClient(app) as client:
        created = client.post("/notebooks", json={"path": str(notebook_path)}).json()
        response = client.post(
            f"/notebooks/{created['id']}/execute",
            json={"cell_index": 0, "source": "print('smoke')"},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert "smoke" in notebook_path.read_text()
        assert client.delete(f"/notebooks/{created['id']}").status_code == 200


def test_busy_connection_helper_preserves_retryable_status(
    multi_user_client, monkeypatch
):
    from open_terminal.utils.service_processes import HelpersBusy

    class BusyClient(FakeClient):
        async def async_start_new_kernel(self, **kwargs):
            raise HelpersBusy()

    client, _ = multi_user_client
    kernel = BusyClient()
    monkeypatch.setattr(
        notebooks, "_new_notebook_client", lambda *args, **kwargs: (kernel, None)
    )
    response = client.post("/notebooks", json={"path": "note.ipynb"}, headers=headers())
    assert response.status_code == 503
    assert response.headers["retry-after"] == "1"
    assert kernel.cleaned is True
    assert not notebooks._sessions
