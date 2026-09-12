"""Trusted admission and process lifetime ownership for the cgroup backend.

Only this service process writes cgroups. User code starts through a fixed
launcher, after joining its task cgroup and dropping the service identity.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import selectors
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial
from pathlib import Path

from fastapi import HTTPException

from open_terminal import config

from .admission import FairQueue, Ticket
from .policy import ExecutionPolicy, load_policy

log = logging.getLogger(__name__)
_SAFE_POPEN_OPTIONS = {
    "stdin",
    "stdout",
    "stderr",
    "bufsize",
    "text",
    "encoding",
    "errors",
    "universal_newlines",
    "restore_signals",
}
_BASE_ENV = {"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8"}
_manager: Manager | None = None
_initialization_lock = threading.Lock()
_disconnect_check = ContextVar("execution_queue_disconnect", default=None)
# Launch handshakes must not occupy the default executor used by file I/O.
_management_workers = ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="execution-control"
)


def enabled() -> bool:
    mode = os.environ.get(
        "OPEN_TERMINAL_EXECUTION_MODE", config.get("execution_mode", "legacy")
    )
    if mode not in ("legacy", "cgroup"):
        raise ValueError("execution_mode must be legacy or cgroup")
    return mode == "cgroup"


async def async_call(function, *args, cleanup=None, **kwargs):
    """Run blocking management off-loop, retaining ownership across cancellation."""
    task = asyncio.get_running_loop().run_in_executor(
        _management_workers, partial(function, *args, **kwargs)
    )
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
        except BaseException:  # noqa: BLE001 - task.result() below propagates the exception
            break
    if cancelled:
        if not task.cancelled() and task.exception() is None and cleanup is not None:
            await async_call(cleanup, task.result())
        else:
            # Retrieve a failed thread's exception even if the caller cancelled.
            if not task.cancelled():
                task.exception()
        raise asyncio.CancelledError
    return task.result()


def _trusted_path(path: Path) -> None:
    """Deployment code, policy and lock directories must not be user writable."""
    if not path.is_absolute():
        raise ValueError("execution paths must be absolute")
    for component in (path, *path.parents):
        info = component.lstat()
        if stat.S_ISLNK(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
            raise ValueError(
                f"execution path must be root-owned and not group/world writable: {component}"
            )


def initialize() -> None:
    global _manager
    if not enabled():
        return
    with _initialization_lock:
        if _manager is not None:
            return
        import fcntl

        from open_terminal.env import MULTI_USER

        from .cgroups import CgroupTree

        if sys.platform != "linux" or os.geteuid() != 0 or not MULTI_USER:
            raise RuntimeError(
                "cgroup execution requires Linux, a root service and multi_user=true"
            )
        policy_file = os.environ.get(
            "OPEN_TERMINAL_EXECUTION_POLICY_FILE",
            config.get("execution_policy_file", ""),
        )
        if not policy_file:
            raise ValueError("cgroup execution requires execution_policy_file")
        _trusted_path(Path(policy_file))
        for source in Path(__file__).resolve().parent.glob("*.py"):
            _trusted_path(source)
        _trusted_path(Path(__file__).resolve().parent.parent / "__init__.py")
        _trusted_path(Path(sys.executable).resolve())
        policy = load_policy(policy_file)
        lock_dir = Path("/run/open-terminal-execution")
        lock_dir.mkdir(mode=0o700, exist_ok=True)
        _trusted_path(lock_dir)
        lock_path = lock_dir / (
            hashlib.sha256(str(policy.cgroup_root).encode()).hexdigest() + ".lock"
        )
        lock_fd = os.open(
            lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600
        )
        try:
            _trusted_path(lock_path)
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            tree = CgroupTree(policy)
            tree.initialize()
            runtime = Manager(policy, tree, lock_fd)
            # Test the actual isolated interpreter, cgroup entry and privilege
            # drop before accepting requests. This runs only /bin/true.
            probe = runtime.prepare(
                "internal-startup-probe", "nobody", ["/bin/true"], cwd="/"
            )
            try:
                process = subprocess.Popen(
                    probe.argv,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    **probe.kwargs,
                )
                probe.started(process)
                if process.wait(timeout=policy.start_timeout) != 0:
                    raise RuntimeError("execution startup probe failed")
            finally:
                probe.abort()
            runtime._uid_owners.clear()
            _manager = runtime
        except BaseException:
            os.close(lock_fd)
            raise


def _runtime() -> Manager:
    if not enabled():
        raise RuntimeError("cgroup execution is disabled")
    # Requests outside an ASGI lifespan still fail closed, never launch directly.
    initialize()
    assert _manager is not None
    return _manager


def prepare(
    owner: str, username: str, argv: list[str], *, cwd=None, env=None
) -> Launch:
    return _runtime().prepare(owner, username, argv, cwd=cwd, env=env)


@contextmanager
def queue_request(request):
    """Associate queue waits with a parsed HTTP request's disconnect signal."""
    token = _disconnect_check.set(request.is_disconnected)
    try:
        yield
    finally:
        _disconnect_check.reset(token)


async def prepare_async(
    owner: str, username: str, argv: list[str], *, cwd=None, env=None
):
    runtime = _runtime()
    ticket = await async_call(
        runtime.submit,
        owner,
        username,
        argv,
        cwd=cwd,
        env=env,
        cleanup=runtime.cancel_ticket,
    )
    ready = asyncio.wrap_future(ticket.ready)
    disconnected = _disconnect_check.get()
    monitor = None

    async def watch():
        while True:
            if await disconnected():
                raise HTTPException(
                    499, "Request disconnected while waiting for compute"
                )
            await asyncio.sleep(0.1)

    try:
        if disconnected is not None and not ready.done():
            monitor = asyncio.create_task(watch())
            done, _ = await asyncio.wait(
                (ready, monitor), return_when=asyncio.FIRST_COMPLETED
            )
            if monitor in done:
                monitor.result()
            return ready.result()
        return await asyncio.shield(ready)
    except BaseException:
        ready.add_done_callback(
            lambda future: None if future.cancelled() else future.exception()
        )
        await async_call(runtime.cancel_ticket, ticket)
        raise
    finally:
        if monitor is not None:
            monitor.cancel()
            # No await after selecting a granted launch: cancellation there
            # could discard the result before ownership reaches the caller.
            monitor.add_done_callback(
                lambda future: None if future.cancelled() else future.exception()
            )


async def spawn_async(
    owner: str, username: str, argv: list[str], *, cwd=None, env=None, **kwargs
):
    launch = await prepare_async(owner, username, argv, cwd=cwd, env=env)
    try:
        return await async_call(spawn_prepared, launch, cleanup=stop, **kwargs)
    except BaseException:
        await async_call(launch.abort)
        raise


def spawn(owner: str, username: str, argv: list[str], *, cwd=None, env=None, **kwargs):
    launch = prepare(owner, username, argv, cwd=cwd, env=env)
    return spawn_prepared(launch, **kwargs)


def spawn_prepared(launch, **kwargs):
    try:
        unsupported = set(kwargs) - _SAFE_POPEN_OPTIONS
        if unsupported:
            raise ValueError(
                f"unsupported managed launch options: {sorted(unsupported)}"
            )
        with launch._cleanup_lock:
            if launch._cancel_requested:
                raise RuntimeError("execution manager stopped during launch")
            process = subprocess.Popen(launch.argv, **kwargs, **launch.kwargs)
            launch.started(process)
            return process
    except BaseException:
        launch.abort()
        raise


def prepare_helper(argv: list[str], **kwargs) -> Launch:
    import grp
    import pwd

    runtime = _runtime()
    username = kwargs.pop("user", None)
    if username is None:
        uid, gid = os.geteuid(), os.getegid()
    else:
        account = (
            pwd.getpwnam(username)
            if isinstance(username, str)
            else pwd.getpwuid(username)
        )
        uid, gid = account.pw_uid, account.pw_gid
    group = kwargs.pop("group", None)
    if group is not None:
        gid = grp.getgrnam(group).gr_gid if isinstance(group, str) else group
    if kwargs.pop("extra_groups", None):
        raise ValueError("managed helpers cannot inherit supplementary groups")
    if kwargs.pop("shell", False) or kwargs.pop("preexec_fn", None) or kwargs.pop("pass_fds", None):
        raise ValueError("managed helpers require a fixed argv and no preexec/pass_fds")
    kwargs.pop("start_new_session", None)
    env = kwargs.pop("env", None)
    # Helpers are trusted fixed programs, but still do not need API credentials.
    environment = dict(_BASE_ENV if env is None else env)
    cwd = kwargs.pop("cwd", None) or "/"
    unsupported = set(kwargs) - _SAFE_POPEN_OPTIONS
    if unsupported:
        raise ValueError(f"unsupported managed helper options: {sorted(unsupported)}")
    launch = Launch(
        runtime,
        None,
        runtime.tree.helpers_path,
        argv,
        uid,
        gid,
        cwd,
        environment,
        helper=True,
    )
    launch.kwargs.update(kwargs)
    return launch


def stop(process, *, force=True, reason="cancelled") -> None:
    launch = getattr(process, "_ot_execution_launch", None)
    if launch is None or launch.helper:
        raise RuntimeError("refusing to stop an untracked process as a managed task")
    launch.stop(force=force, reason=reason)


def describe(process):
    launch = getattr(process, "_ot_execution_launch", None)
    return launch.snapshot() if launch is not None and not launch.helper else None


def shutdown() -> None:
    # Keep the lock until process exit: releasing it while helper cleanup is
    # pending would let a second API attach to an incompletely drained tree.
    if _manager is not None:
        _manager.shutdown()


class Manager:
    def __init__(self, policy: ExecutionPolicy, tree, lock_fd: int | None = None):
        self.policy = policy
        self.tree = tree
        self.lock_fd = lock_fd
        self._lock = threading.RLock()
        self._tasks: dict[str, Launch | None] = {}
        self._owners: dict[str, set[str]] = {}
        self._uid_owners: dict[int, str] = {}
        self._stopping = False
        self._queue = FairQueue(self)

    def _has_capacity(self, owner_key):
        return (
            len(self._tasks) < self.policy.max_tasks
            and len(self._owners.get(owner_key, ())) < self.policy.max_user_tasks
            and (
                owner_key in self._owners
                or (self.policy.user * (len(self._owners) + 1)).fits_within(
                    self.policy.compute
                )
            )
        )

    def submit(self, owner, username, argv, *, cwd=None, env=None):
        account, owner_key = self._account(owner, username)
        # Bound retained queue data as well as ticket count. Avoid serializing
        # arbitrarily large input just to discover that it exceeds the bound.
        values = list(argv) + [cwd or ""]
        if env is not None:
            values.extend(env.keys())
            values.extend(env.values())
        if not argv or not all(isinstance(value, str) for value in values):
            raise HTTPException(400, "Invalid queued execution specification")
        if (
            sum(map(len, values)) > 1024 * 1024
            or sum(len(value.encode()) for value in values) > 1024 * 1024
        ):
            raise HTTPException(413, "Queued command and environment exceed 1 MiB")
        with self._lock:
            if self._stopping:
                raise HTTPException(503, "Execution manager is stopping")
            if self._uid_owners.get(account.pw_uid, owner_key) != owner_key:
                raise HTTPException(
                    409, "User identifiers resolve to the same OS account"
                )
            self._uid_owners[account.pw_uid] = owner_key
            return self._queue.submit(
                Ticket(
                    owner,
                    owner_key,
                    username,
                    list(argv),
                    cwd,
                    dict(env) if env is not None else None,
                )
            )

    def cancel_ticket(self, ticket):
        launch = self._queue.cancel(ticket)
        if launch is not None:
            launch.abort()

    def _account(self, owner, username):
        import pwd

        if not isinstance(owner, str) or not owner.strip():
            raise HTTPException(400, "X-User-Id is required for managed execution")
        account = pwd.getpwnam(username)
        if account.pw_uid == 0 or account.pw_uid == os.geteuid() or account.pw_gid == 0:
            raise HTTPException(
                403, "Execution requires a separate unprivileged OS account"
            )
        return account, hashlib.sha256(owner.encode()).hexdigest()

    def prepare(self, owner, username, argv, *, cwd=None, env=None, _queued=False):
        account, owner_key = self._account(owner, username)
        task_id = uuid.uuid4().hex
        with self._lock:
            if self._stopping:
                raise HTTPException(503, "Execution manager is stopping")
            if self._uid_owners.get(account.pw_uid, owner_key) != owner_key:
                raise HTTPException(
                    409, "User identifiers resolve to the same OS account"
                )
            if not self._has_capacity(owner_key) or (not _queued and self._queue.count):
                raise HTTPException(
                    429,
                    "Compute budget is full. Retry later.",
                    headers={"Retry-After": "1"},
                )
            path = self.tree.create_task(owner_key, task_id)
            self._tasks[task_id] = None
            self._owners.setdefault(owner_key, set()).add(task_id)
            self._uid_owners[account.pw_uid] = owner_key
            try:
                environment = dict(_BASE_ENV)
                if env:
                    environment.update(env)
                environment.update(
                    HOME=account.pw_dir, USER=account.pw_name, LOGNAME=account.pw_name
                )
                launch = Launch(
                    self,
                    (owner_key, task_id),
                    path,
                    argv,
                    account.pw_uid,
                    account.pw_gid,
                    cwd or account.pw_dir,
                    environment,
                )
                self._tasks[task_id] = launch
                return launch
            except BaseException:
                # No child exists yet. Only release after the cgroup is empty.
                if self.tree.is_empty(path):
                    self.tree.remove_task(path)
                    self._release(owner_key, task_id)
                raise

    def _release(self, owner, task):
        with self._lock:
            self._tasks.pop(task, None)
            tasks = self._owners.get(owner)
            if tasks is not None:
                tasks.discard(task)
                if not tasks:
                    del self._owners[owner]
            self._queue.condition.notify_all()
            # UID ownership persists for this manager lifetime; UID reuse may
            # otherwise expose a previous user's files to a colliding identity.

    def shutdown(self):
        with self._lock:
            self._stopping = True
            self._queue.close()
            tasks = list(self._tasks.values())
        for task in tasks:
            if task is not None:
                task.request_stop()


class Launch:
    def __init__(
        self, manager, identity, path, argv, uid, gid, cwd, environment, *, helper=False
    ):
        if not argv or not all(
            isinstance(arg, str) and "\0" not in arg for arg in argv
        ):
            raise ValueError("managed launch requires a nonempty string argv")
        if not isinstance(environment, dict) or not all(
            isinstance(k, str)
            and isinstance(v, str)
            and k
            and "=" not in k
            and "\0" not in k + v
            for k, v in environment.items()
        ):
            raise ValueError("invalid execution environment")
        self.manager = manager
        self.identity = identity
        self.path = path
        self.helper = helper
        self.process = None
        self._fds = []
        self._cleanup_lock = threading.RLock()
        self._stopping = False
        self._cancel_requested = False
        self._closed = False
        self._reaper_started = False
        self._environment_file = None
        self.queued_at = None
        self.queue_wait_seconds = 0.0
        self.started_at = None
        self.deadline = None
        self.finished_at = None
        self._deadline_monotonic = None
        self._terminate_at = None
        self.end_reason = None
        try:
            cgfd = os.open(
                path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
            )
            self._fds.append(cgfd)
            readfd, writefd = os.pipe2(os.O_CLOEXEC)
            self._fds.extend((readfd, writefd))
            self._status_read = readfd
            self._status_write = writefd
            self._environment_file = tempfile.TemporaryFile()  # noqa: SIM115 - owned across launch/cleanup
            self._environment_file.write(json.dumps(environment).encode())
            self._environment_file.flush()
            self._environment_file.seek(0)
            envfd = self._environment_file.fileno()
            self.argv = [
                str(Path(sys.executable).resolve()),
                "-I",
                str(Path(__file__).resolve().with_name("launcher.py")),
                "--cgroup-fd",
                str(cgfd),
                "--status-fd",
                str(writefd),
                "--environment-fd",
                str(envfd),
                "--uid",
                str(uid),
                "--gid",
                str(gid),
                "--cwd",
                str(cwd),
            ]
            if helper:
                self.argv.append("--helper")
            self.argv.extend(("--", *argv))
            self.kwargs = {
                "pass_fds": (cgfd, writefd, envfd),
                "env": dict(_BASE_ENV),
                "cwd": "/",
                "start_new_session": True,
            }
        except BaseException:
            self._close_fds()
            raise

    def _close_fds(self):
        with self._cleanup_lock:
            for descriptor in self._fds:
                os.close(descriptor)
            self._fds.clear()
            if self._environment_file is not None:
                self._environment_file.close()
                self._environment_file = None

    def started(self, process):
        with self._cleanup_lock:
            if self._closed:
                process.kill()
                process.wait(timeout=self.manager.policy.cleanup_timeout)
                raise RuntimeError(
                    "managed launch was cancelled before process registration"
                )
            if self._cancel_requested:
                self.process = process
                self.abort()
                raise RuntimeError("execution manager stopped during launch")
            self._started(process)

    def request_stop(self):
        with self._cleanup_lock:
            self._cancel_requested = True
            self.end_reason = self.end_reason or "shutdown"
            # External adapters (Jupyter) may be between prepare and Popen.
            # Keep descriptors and reservation until that adapter acknowledges
            # the launch through started()/abort(), avoiding FD reuse at fork.
            if self.process is not None:
                self.abort()

    def _started(self, process):
        self.process = process
        process._ot_execution_launch = self
        try:
            os.close(self._status_write)
            self._fds.remove(self._status_write)
            deadline = time.monotonic() + self.manager.policy.start_timeout
            with selectors.DefaultSelector() as selector:
                selector.register(self._status_read, selectors.EVENT_READ)
                received = b""
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        raise RuntimeError("managed process startup timed out")
                    message = os.read(self._status_read, 4096)
                    if not message:
                        fields = received.split()
                        if (
                            len(fields) != 3
                            or fields[0] != b"READY"
                            or not all(v.isdigit() for v in fields[1:])
                        ):
                            raise RuntimeError(
                                "managed launcher exited before confirming isolation"
                            )
                        started_mono = int(fields[1]) / 1_000_000_000
                        started_wall = int(fields[2]) / 1_000_000_000
                        if (
                            started_mono <= 0
                            or started_mono > time.monotonic()
                            or started_wall <= 0
                        ):
                            raise RuntimeError("invalid launcher start timestamp")
                        if not self.helper:
                            self.started_at = started_wall
                            self.deadline = (
                                started_wall + self.manager.policy.max_runtime
                            )
                            self._deadline_monotonic = (
                                started_mono + self.manager.policy.max_runtime
                            )
                        break
                    received += message
                    if len(received) > 256 or not (
                        b"READY ".startswith(received) or received.startswith(b"READY ")
                    ):
                        raise RuntimeError(
                            "managed process could not start: "
                            + received.decode(errors="replace")
                        )
        except BaseException:
            self.abort()
            raise
        finally:
            self._close_fds()
        if not self.helper:
            self._ensure_reaper()

    def _ensure_reaper(self):
        with self._cleanup_lock:
            if self._reaper_started or self._closed:
                return
            self._reaper_started = True
        threading.Thread(
            target=self._reap, name="execution-reaper", daemon=True
        ).start()

    def snapshot(self):
        return {
            "state": "finished"
            if self._closed
            else "stopping"
            if self._stopping
            else "running"
            if self.started_at is not None
            else "starting",
            "started_at": self.started_at,
            "deadline": self.deadline,
            "finished_at": self.finished_at,
            "end_reason": self.end_reason,
            "max_runtime": self.manager.policy.max_runtime,
            "queued_at": self.queued_at,
            "queue_wait_seconds": self.queue_wait_seconds,
        }

    def _reason(self, fallback):
        if self.end_reason is not None:
            return
        try:
            events = self.manager.tree.task_usage(self.path)["memory_events"]
            oom = events.get("oom_kill", 0)
            self.end_reason = "oom" if isinstance(oom, int) and oom > 0 else fallback
        except Exception:
            log.debug("Could not read final task memory events", exc_info=True)
            self.end_reason = fallback

    def _begin_stop(self, reason):
        self._reason(reason)
        if self._stopping:
            return
        self._stopping = True
        self._terminate_at = time.monotonic() + self.manager.policy.terminate_grace
        try:
            self.manager.tree.terminate(self.path)
        except Exception:
            log.exception("Graceful task termination failed; escalating to cgroup kill")
            self._terminate_at = time.monotonic()

    def _advance(self):
        with self._cleanup_lock:
            if self._closed:
                return True
            if self._stopping:
                if (
                    self._terminate_at is None
                    or time.monotonic() >= self._terminate_at
                    or self.manager.tree.is_empty(self.path)
                ):
                    self._cleanup()
                    return True
            elif self.process is None or self.process.poll() is not None:
                self._reason(
                    "completed" if self.started_at is not None else "start_failed"
                )
                self._cleanup()
                return True
            elif (
                self._deadline_monotonic is not None
                and time.monotonic() >= self._deadline_monotonic
            ):
                self._begin_stop("timed_out")
            return False

    def _reap(self):
        while True:
            try:
                if self._advance():
                    return
            except Exception:
                log.exception(
                    "Task cleanup incomplete; its resource reservation remains held"
                )
            time.sleep(0.05)

    def stop(self, *, force=True, reason="cancelled"):
        if reason not in {"cancelled", "completed", "shutdown"}:
            raise ValueError("invalid task stop reason")
        with self._cleanup_lock:
            if self._closed:
                return
            self._reason(reason)
            # Output/kernel cleanup can run as soon as the leader exits.
            # Preserve a grace period already offered to its remaining children.
            preserve_grace = (
                reason == "completed"
                and self._stopping
                and self._terminate_at is not None
            )
            if force and not preserve_grace:
                self.abort()
                return
            self._begin_stop(reason)
            self._ensure_reaper()
        while not self._advance():
            time.sleep(0.05)

    def _cleanup(self):
        with self._cleanup_lock:
            if self._closed:
                return
            self._stopping = True
            self._reason("start_failed" if self.started_at is None else "cancelled")
            self._close_fds()
            # A failed launcher may not have entered the cgroup yet. Reap that
            # trusted child as well as any processes already inside the group.
            if self.process is not None and self.process.poll() is None:
                try:
                    self.process.kill()
                except ProcessLookupError:
                    pass
            self.manager.tree.kill(self.path)
            deadline = time.monotonic() + self.manager.policy.cleanup_timeout
            while not self.manager.tree.is_empty(self.path):
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "task cgroup is still populated; compute reservation retained"
                    )
                time.sleep(0.01)
            if self.process is not None:
                self.process.wait(timeout=self.manager.policy.cleanup_timeout)
            with self.manager._lock:
                self.manager.tree.remove_task(self.path)
                self.manager._release(*self.identity)
            self.finished_at = time.time()
            self._closed = True

    def abort(self):
        if self.helper:
            # The helper pool owns child process-group reaping. Never kill the
            # shared helper cgroup, which may contain unrelated file uploads.
            self._close_fds()
            return
        try:
            self._cleanup()
        except BaseException:
            self._ensure_reaper()
            raise
