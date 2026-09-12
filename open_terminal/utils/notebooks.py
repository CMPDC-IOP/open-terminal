"""Jupyter notebook execution endpoints with per-user session isolation."""

import asyncio
import logging
import os
import shutil
import sys
import tempfile
import time
import uuid
from collections.abc import Callable
from typing import Annotated, Any

import nbformat
from fastapi import APIRouter, Depends, HTTPException, Request
from jupyter_client.manager import AsyncKernelManager
from nbclient import NotebookClient
from pydantic import BaseModel, Field

from open_terminal import execution as execution_runtime
from open_terminal.utils.compute_environment import (
    compute_thread_environment,
    with_compute_thread_defaults,
)
from open_terminal.utils.service_processes import HelpersBusy, open_helper, run_helper

_IDLE_TIMEOUT = 30 * 60
_SUDO_ENV = {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LANG": "C.UTF-8"}
_CONTROL_CHARACTERS = frozenset(chr(value) for value in range(32)) | {chr(127)}
log = logging.getLogger(__name__)


class _Session:
    """A kernel and notebook bound to one API user and session context."""

    __slots__ = (
        "busy",
        "client",
        "context_id",
        "created_at",
        "_execution_stop_reason",
        "execution_info",
        "home",
        "id",
        "last_used",
        "nb",
        "path",
        "relative_path",
        "runtime_directory",
        "user_id",
        "username",
    )

    def __init__(
        self,
        session_id: str,
        path: str,
        relative_path: str | None,
        nb: Any,
        client: NotebookClient,
        *,
        user_id: str,
        context_id: str,
        username: str | None,
        home: str,
        runtime_directory: str | None,
    ):
        self.id, self.path, self.relative_path, self.nb, self.client = (
            session_id,
            path,
            relative_path,
            nb,
            client,
        )
        self.user_id, self.context_id, self.username, self.home = (
            user_id,
            context_id,
            username,
            home,
        )
        self.runtime_directory, self.busy = runtime_directory, False
        self._execution_stop_reason = "completed"
        self.execution_info = None
        self.created_at = self.last_used = time.time()


_sessions: dict[str, _Session] = {}
_cleanup_task: asyncio.Task | None = None


async def _idle_cleanup_loop() -> None:
    while True:
        await asyncio.sleep(60)
        now = time.time()
        for session_id in [
            sid
            for sid, item in _sessions.items()
            if now - item.last_used > _IDLE_TIMEOUT and not item.busy
        ]:
            await _destroy_session(session_id)


async def _cleanup_session_resources(session: _Session) -> None:
    try:
        manager = getattr(session.client, "km", None)
        if manager is not None:
            manager._execution_stop_reason = getattr(
                session, "_execution_stop_reason", "completed"
            )
        await session.client._async_cleanup_kernel()
    except Exception:
        log.exception("Failed to clean up notebook kernel %s", session.id)
    finally:
        if session.runtime_directory:
            await asyncio.to_thread(shutil.rmtree, session.runtime_directory, True)


async def _destroy_session(session_id: str, *, reason: str = "completed") -> None:
    """Remove a session before awaiting teardown, so it cannot be reused."""
    session = _sessions.pop(session_id, None)
    if session is None:
        return
    session._execution_stop_reason = reason
    cleanup = asyncio.create_task(_cleanup_session_resources(session))
    try:
        await asyncio.shield(cleanup)
    except asyncio.CancelledError:
        # Keep cleanup alive after a cancelled HTTP request or app shutdown task.
        def consume_cleanup_result(completed: asyncio.Task) -> None:
            if not completed.cancelled():
                completed.exception()

        cleanup.add_done_callback(consume_cleanup_result)
        raise


def _ensure_cleanup_task() -> None:
    global _cleanup_task
    if _cleanup_task is None or _cleanup_task.done():
        _cleanup_task = asyncio.create_task(_idle_cleanup_loop())


class CreateSessionRequest(BaseModel):
    path: str = Field(..., description="Path to the .ipynb file.")


class CreateSessionResponse(BaseModel):
    id: str
    kernel: str
    status: str


class ExecuteCellRequest(BaseModel):
    cell_index: int = Field(description="Zero-based cell index to execute.")
    source: str | None = Field(
        None, description="Override the source already in the notebook."
    )


class ExecuteCellResponse(BaseModel):
    status: str
    execution_count: int | None = None
    outputs: list = Field(default_factory=list)


class SessionStatusResponse(BaseModel):
    id: str
    path: str
    kernel: str
    status: str
    execution: dict[str, Any] | None = None


def _session_execution_info(session: _Session) -> dict | None:
    """Return managed kernel lifetime metadata when a process is available."""
    if not execution_runtime.enabled():
        return None
    manager = getattr(session.client, "km", None)
    process = getattr(manager, "_execution_process", None)
    if process is None:
        execution = getattr(manager, "_execution_info", None)
        if execution is not None:
            session.execution_info = execution
        return session.execution_info
    execution = execution_runtime.describe(process)
    if execution is not None:
        session.execution_info = execution
    return execution


def _expired_kernel_detail(execution: dict) -> str:
    if execution.get("end_reason") == "timed_out":
        return "Notebook kernel is no longer available because its runtime expired"
    return "Notebook kernel is no longer available because it has finished"


def _filesystem_identity(
    filesystem: Any, *, multi_user: bool
) -> tuple[str | None, str]:
    username, home = (
        getattr(filesystem, "username", None),
        getattr(filesystem, "home", None),
    )
    if not isinstance(home, str) or not os.path.isabs(home):
        raise HTTPException(
            status_code=403, detail="Notebook filesystem is unavailable"
        )
    if multi_user and (not isinstance(username, str) or not username):
        raise HTTPException(
            status_code=403, detail="Notebook filesystem is unavailable"
        )
    return username if isinstance(
        username, str
    ) and username else None, os.path.normpath(home)


def _notebook_path(
    path: str, filesystem: Any, *, home: str, multi_user: bool
) -> tuple[str, str | None]:
    if (
        not isinstance(path, str)
        or any(character in _CONTROL_CHARACTERS for character in path)
        or (multi_user and "\\" in path)
    ):
        raise HTTPException(status_code=400, detail="Invalid notebook path")
    try:
        resolved = filesystem.resolve_path(path)
    except (AttributeError, TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid notebook path") from None
    if not isinstance(resolved, str) or not os.path.isabs(resolved):
        raise HTTPException(status_code=400, detail="Invalid notebook path")
    resolved = os.path.normpath(resolved)
    if not multi_user:
        return resolved, None
    relative = os.path.relpath(resolved, home)
    if relative == ".." or relative.startswith(f"..{os.sep}"):
        raise HTTPException(status_code=403, detail="Notebook access denied")
    return resolved, relative


async def _run_notebook_file_helper(
    action: str,
    *,
    home: str,
    username: str,
    relative_path: str,
    content: bytes | None = None,
) -> bytes:
    command = [
        "sudo",
        "-n",
        "-u",
        username,
        "--",
        sys.executable,
        "-I",
        "-m",
        "open_terminal.utils.notebook_files",
        action,
        "--home",
        home,
        "--username",
        username,
        "--path",
        relative_path,
    ]
    async with open_helper(
        *command,
        stdin=asyncio.subprocess.PIPE
        if content is not None
        else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=_SUDO_ENV,
    ) as process:
        stdout, _ = await process.communicate(content)
        if process.returncode:
            raise OSError("Notebook file is unavailable")
        return stdout


async def _read_notebook(
    filesystem: Any,
    path: str,
    relative_path: str | None,
    *,
    username: str | None,
    home: str,
) -> str:
    try:
        if username is None:
            return await filesystem.read_text(path)
        assert relative_path is not None
        data = await _run_notebook_file_helper(
            "read", home=home, username=username, relative_path=relative_path
        )
        return data.decode("utf-8", "strict")
    except (OSError, UnicodeDecodeError, PermissionError):
        raise HTTPException(status_code=404, detail="Notebook not found") from None


async def _write_notebook(
    filesystem: Any,
    path: str,
    relative_path: str | None,
    content: str,
    *,
    username: str | None,
    home: str,
) -> None:
    try:
        if username is None:
            await filesystem.write(path, content)
        else:
            assert relative_path is not None
            await _run_notebook_file_helper(
                "write",
                home=home,
                username=username,
                relative_path=relative_path,
                content=content.encode("utf-8"),
            )
    except (OSError, PermissionError):
        log.exception("Saving notebook failed")
        raise HTTPException(status_code=500, detail="Failed to save notebook") from None


def _connection_runtime_directory() -> tuple[str, str]:
    directory = tempfile.mkdtemp(prefix="open-terminal-kernel-")
    # The service owns this directory; users can traverse it but cannot replace its file.
    os.chmod(directory, 0o711)
    return directory, os.path.join(directory, "kernel.json")


def _transfer_connection_file(path: str, username: str) -> None:
    if os.geteuid() == 0:
        import pwd

        account = pwd.getpwnam(username)
        os.chown(path, -1, account.pw_gid)
    else:
        run_helper(
            ["sudo", "-n", "chgrp", username, "--", path],
            check=True,
            capture_output=True,
            env=_SUDO_ENV,
        )
    # The service owns the file and the target user's primary group can read
    # it.  This lets both Jupyter's manager and its dropped-identity kernel
    # reconcile the same connection data without exposing it to other users.
    os.chmod(path, 0o640)


def _user_kernel_manager_class(
    username: str, home: str, working_directory: str, connection_file: str,
    owner: str | None = None,
) -> type[AsyncKernelManager]:
    class UserKernelManager(AsyncKernelManager):
        def __init__(self, *args: Any, **kwargs: Any):
            kwargs["connection_file"] = connection_file
            super().__init__(*args, **kwargs)
            self._execution_launch = None
            self._execution_process = None
            self._execution_info = None

        async def _async_pre_start_kernel(self, **kwargs: Any):
            if execution_runtime.enabled():
                from jupyter_client.provisioning.local_provisioner import LocalProvisioner

                if not owner:
                    raise ValueError("Hard execution requires a notebook owner")
                metadata = self.kernel_spec.metadata if self.kernel_spec else {}
                provisioner_name = metadata.get("kernel_provisioner", {}).get(
                    "provisioner_name", "local-provisioner"
                )
                if provisioner_name != "local-provisioner":
                    raise RuntimeError("Hard execution requires the local kernel provisioner")
                if self.provisioner is None:
                    self.kernel_id = self.kernel_id or str(uuid.uuid4())
                    self.provisioner = LocalProvisioner(
                        kernel_id=self.kernel_id, kernel_spec=self.kernel_spec, parent=self
                    )
                if type(self.provisioner) is not LocalProvisioner:
                    raise RuntimeError("Hard execution requires the local kernel provisioner")
            # jupyter-client >= 8 dispatches kernel startup through this override.
            kernel_command, launch_kwargs = await super()._async_pre_start_kernel(
                **kwargs
            )
            await asyncio.to_thread(
                _transfer_connection_file, self.connection_file, username
            )
            # Only a trusted environment reaches sudo. Kernelspec variables are
            # installed by env(1) after sudo has switched to the target user.
            kernel_environment = {
                "HOME": home,
                "LOGNAME": username,
                "USER": username,
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "LANG": "C.UTF-8",
            }
            # Only these resolved library defaults are carried through the
            # restricted sudo launch. Other launch variables stay subject to
            # the existing kernelspec allow-list below.
            kernel_environment.update(
                compute_thread_environment(launch_kwargs.get("env"))
            )
            for key in self.kernel_spec.env if self.kernel_spec else ():
                value = launch_kwargs.get("env", {}).get(key)
                if isinstance(value, str):
                    kernel_environment[key] = value
            if execution_runtime.enabled():
                self._execution_launch = await execution_runtime.prepare_async(
                    owner,
                    username,
                    kernel_command,
                    cwd=working_directory, env=kernel_environment,
                )
                launch_kwargs.update(self._execution_launch.kwargs)
                return self._execution_launch.argv, launch_kwargs
            launch_kwargs["env"] = dict(_SUDO_ENV)
            # ``Popen(cwd=...)`` happens before sudo changes identity.  Change
            # directory in this fixed shell fragment after the privilege drop;
            # kernelspec arguments stay separate and are executed through "$@".
            launch_kwargs["cwd"] = None
            assignments = [
                f"{key}={value}" for key, value in kernel_environment.items()
            ]
            return [
                "sudo",
                "-n",
                "-u",
                username,
                "--",
                "env",
                "-i",
                *assignments,
                "/bin/sh",
                "-c",
                'cd -- "$1" || exit 1; shift; exec "$@"',
                "open-terminal-kernel",
                working_directory,
                *kernel_command,
            ], launch_kwargs

        async def _async_launch_kernel(self, kernel_cmd: list[str], **kwargs: Any):
            try:
                await super()._async_launch_kernel(kernel_cmd, **kwargs)
                if self._execution_launch is not None:
                    self._execution_process = self.provisioner.process
                    await execution_runtime.async_call(
                        self._execution_launch.started, self._execution_process,
                        cleanup=lambda _: self._execution_launch.abort(),
                    )
            except BaseException:
                if self._execution_launch is not None:
                    await execution_runtime.async_call(self._execution_launch.abort)
                raise

        async def _async_cleanup_resources(self, restart: bool = False):
            try:
                if self._execution_process is not None:
                    await execution_runtime.async_call(
                        execution_runtime.stop,
                        self._execution_process,
                        reason=getattr(self, "_execution_stop_reason", "completed"),
                    )
                    self._execution_info = execution_runtime.describe(
                        self._execution_process
                    )
                elif self._execution_launch is not None:
                    await execution_runtime.async_call(self._execution_launch.abort)
            finally:
                self._execution_launch = None
                self._execution_process = None
                await super()._async_cleanup_resources(restart=restart)

    return UserKernelManager


def _new_notebook_client(
    notebook: Any,
    kernel_name: str,
    *,
    username: str | None,
    home: str,
    working_directory: str,
    owner: str | None = None,
) -> tuple[NotebookClient, str | None]:
    if execution_runtime.enabled() and (not owner or not username):
        raise ValueError("Hard execution requires a notebook owner and target user")
    if username is None:
        return NotebookClient(notebook, kernel_name=kernel_name, timeout=120), None
    runtime_directory, connection_file = _connection_runtime_directory()
    return (
        NotebookClient(
            notebook,
            kernel_name=kernel_name,
            timeout=120,
            kernel_manager_class=_user_kernel_manager_class(
                username, home, working_directory, connection_file, owner=owner
            ),
        ),
        runtime_directory,
    )


def _require_user_context(request: Request, *, multi_user: bool) -> tuple[str, str]:
    user_id = request.headers.get("x-user-id", "")
    if (multi_user or execution_runtime.enabled()) and not user_id.strip():
        raise HTTPException(status_code=403, detail="X-User-Id is required")
    return user_id, request.headers.get("x-session-id", "")


def _owned_session(
    session_id: str, request: Request, filesystem: Any, *, multi_user: bool
) -> _Session:
    session = _sessions.get(session_id)
    user_id, context_id = _require_user_context(request, multi_user=multi_user)
    username, home = _filesystem_identity(filesystem, multi_user=multi_user)
    if (
        session is None
        or session.user_id != user_id
        or session.context_id != context_id
        or session.username != username
        or session.home != home
    ):
        raise HTTPException(status_code=404, detail="Session not found")
    return session


def create_notebooks_router(
    verify_api_key: Callable[..., Any],
    get_filesystem: Callable[..., Any],
    *,
    multi_user: bool,
) -> APIRouter:
    """Create notebook routes bound to the app's filesystem and identity rules."""
    router = APIRouter(
        prefix="/notebooks", tags=["notebooks"], dependencies=[Depends(verify_api_key)]
    )

    @router.post(
        "",
        response_model=CreateSessionResponse,
        include_in_schema=False,
        operation_id="create_notebook_session",
        summary="Create a notebook session",
        description="Start a Jupyter kernel for the given notebook.",
    )
    async def create_session(
        req: CreateSessionRequest,
        http_request: Request,
        filesystem: Annotated[Any, Depends(get_filesystem)],
    ) -> CreateSessionResponse:
        user_id, context_id = _require_user_context(http_request, multi_user=multi_user)
        username, home = _filesystem_identity(filesystem, multi_user=multi_user)
        path, relative_path = _notebook_path(
            req.path, filesystem, home=home, multi_user=multi_user
        )

        content = await _read_notebook(
            filesystem, path, relative_path, username=username, home=home
        )
        try:
            notebook = nbformat.reads(content, as_version=4)
        except Exception as error:
            raise HTTPException(status_code=400, detail="Invalid notebook") from error
        kernel_name = notebook.metadata.get("kernelspec", {}).get("name", "python3")
        client, runtime_directory = _new_notebook_client(
            notebook,
            kernel_name,
            username=username,
            home=home,
            working_directory=os.path.dirname(path),
            owner=user_id,
        )
        try:
            client.create_kernel_manager()
            with execution_runtime.queue_request(http_request):
                await client.async_start_new_kernel(
                    cwd=os.path.dirname(path), env=with_compute_thread_defaults({} if execution_runtime.enabled() else None)
                )
                await client.async_start_new_kernel_client()
        except BaseException as error:
            log.exception("Notebook kernel startup failed for %s", kernel_name)
            failed = _Session(
                "",
                path,
                relative_path,
                notebook,
                client,
                user_id=user_id,
                context_id=context_id,
                username=username,
                home=home,
                runtime_directory=runtime_directory,
            )
            await _cleanup_session_resources(failed)
            if isinstance(error, (asyncio.CancelledError, HelpersBusy, HTTPException)):
                raise
            raise HTTPException(
                status_code=500, detail=f"Failed to start kernel '{kernel_name}'"
            ) from error
        _ensure_cleanup_task()
        session_id = uuid.uuid4().hex[:12]
        _sessions[session_id] = _Session(
            session_id,
            path,
            relative_path,
            notebook,
            client,
            user_id=user_id,
            context_id=context_id,
            username=username,
            home=home,
            runtime_directory=runtime_directory,
        )
        return CreateSessionResponse(id=session_id, kernel=kernel_name, status="ready")

    @router.post(
        "/{session_id}/execute",
        response_model=ExecuteCellResponse,
        include_in_schema=False,
        operation_id="execute_notebook_cell",
        summary="Execute a notebook cell",
        description="Execute a cell and save its outputs to the notebook.",
    )
    async def execute_cell(
        session_id: str,
        req: ExecuteCellRequest,
        http_request: Request,
        filesystem: Annotated[Any, Depends(get_filesystem)],
    ) -> ExecuteCellResponse:
        session = _owned_session(
            session_id, http_request, filesystem, multi_user=multi_user
        )
        execution = _session_execution_info(session)
        if execution and execution.get("state") in {"stopping", "finished"}:
            raise HTTPException(status_code=409, detail=_expired_kernel_detail(execution))
        if session.busy:
            raise HTTPException(status_code=409, detail="Cell already executing")
        notebook = session.nb
        if req.cell_index < 0 or req.cell_index >= len(notebook.cells):
            raise HTTPException(status_code=400, detail="cell_index out of range")
        cell = notebook.cells[req.cell_index]
        if req.source is not None:
            cell.source = req.source
        session.busy, session.last_used = True, time.time()
        try:
            try:
                await session.client.async_execute_cell(cell, req.cell_index)
            except Exception as error:
                log.debug("Notebook cell execution failed", exc_info=True)
                return ExecuteCellResponse(
                    status="error",
                    outputs=[
                        {
                            "output_type": "error",
                            "ename": type(error).__name__,
                            "evalue": str(error),
                            "traceback": [str(error)],
                        }
                    ],
                )
            outputs = []
            for output in cell.outputs:
                serialized = dict(output)
                if "data" in serialized:
                    serialized["data"] = dict(serialized["data"])
                outputs.append(serialized)
            await _write_notebook(
                filesystem,
                session.path,
                session.relative_path,
                nbformat.writes(notebook),
                username=session.username,
                home=session.home,
            )
            return ExecuteCellResponse(
                status="ok",
                execution_count=cell.get("execution_count"),
                outputs=outputs,
            )
        except asyncio.CancelledError:
            await _destroy_session(session_id)
            raise
        finally:
            session.busy, session.last_used = False, time.time()

    @router.get(
        "/{session_id}",
        response_model=SessionStatusResponse,
        response_model_exclude_none=True,
        include_in_schema=False,
        operation_id="get_notebook_session",
        summary="Get notebook session status",
    )
    async def get_session(
        session_id: str,
        http_request: Request,
        filesystem: Annotated[Any, Depends(get_filesystem)],
    ) -> SessionStatusResponse:
        session = _owned_session(
            session_id, http_request, filesystem, multi_user=multi_user
        )
        execution = _session_execution_info(session)
        status = (
            "stopped"
            if execution and execution.get("state") in {"stopping", "finished"}
            else "busy" if session.busy else "ready"
        )
        return SessionStatusResponse(
            id=session.id,
            path=session.path,
            kernel=session.nb.metadata.get("kernelspec", {}).get("name", "python3"),
            status=status,
            execution=execution,
        )

    @router.delete(
        "/{session_id}",
        include_in_schema=False,
        operation_id="delete_notebook_session",
        summary="Stop a notebook session",
    )
    async def delete_session(
        session_id: str,
        http_request: Request,
        filesystem: Annotated[Any, Depends(get_filesystem)],
    ) -> dict[str, str]:
        _owned_session(session_id, http_request, filesystem, multi_user=multi_user)
        await _destroy_session(session_id, reason="cancelled")
        return {"status": "stopped"}

    return router
