"""Bounded admission for trusted file/control helpers, separate from computation.

The pool is shared by async routes and blocking worker threads in one service
process. It limits helper launches, not their descendants or memory usage;
those guarantees require the execution isolation layer described in TODO.md.
"""

import asyncio
import concurrent.futures
import logging
import math
import os
import signal
import subprocess
import threading
from collections import deque
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field

from fastapi import HTTPException

log = logging.getLogger(__name__)


class HelpersBusy(HTTPException):
    def __init__(self):
        super().__init__(
            503, "File helpers are busy. Retry later.", headers={"Retry-After": "1"}
        )


@dataclass(eq=False)
class _Ticket:
    ready: concurrent.futures.Future = field(default_factory=concurrent.futures.Future)
    granted: bool = False
    released: bool = False


class HelperPool:
    """A FIFO pool whose tickets can be awaited from threads or event loops."""

    def __init__(self, limit: int, max_queue: int, wait_timeout: float):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("helper limit must be a positive integer")
        if (
            isinstance(max_queue, bool)
            or not isinstance(max_queue, int)
            or max_queue < 0
        ):
            raise ValueError("helper queue length must be a nonnegative integer")
        if not math.isfinite(wait_timeout) or wait_timeout < 0:
            raise ValueError("helper wait timeout must be finite and nonnegative")
        self.limit = limit
        self.max_queue = max_queue
        self.wait_timeout = wait_timeout
        self._lock = threading.Lock()
        self._active = 0
        self._waiting: deque[_Ticket] = deque()

    def _request(self) -> _Ticket:
        with self._lock:
            ticket = _Ticket()
            if self._active < self.limit:
                self._active += 1
                ticket.granted = True
                ticket.ready.set_result(None)
            else:
                if self.wait_timeout == 0 or len(self._waiting) >= self.max_queue:
                    raise HelpersBusy()
                self._waiting.append(ticket)
            return ticket

    def _release(self, ticket: _Ticket) -> None:
        with self._lock:
            if ticket.released:
                return
            ticket.released = True
            if not ticket.granted:
                self._waiting.remove(ticket)
                ticket.ready.cancel()
                return
            if self._waiting:
                next_ticket = self._waiting.popleft()
                next_ticket.granted = True
                next_ticket.ready.set_result(None)
            else:
                self._active -= 1

    @contextmanager
    def sync_slot(self):
        ticket = self._request()
        try:
            try:
                ticket.ready.result(timeout=self.wait_timeout)
            except concurrent.futures.TimeoutError:
                raise HelpersBusy() from None
            yield
        finally:
            self._release(ticket)

    @asynccontextmanager
    async def slot(self):
        ticket = self._request()
        try:
            if not ticket.ready.done():
                try:
                    await asyncio.wait_for(
                        asyncio.shield(asyncio.wrap_future(ticket.ready)),
                        self.wait_timeout,
                    )
                except TimeoutError:
                    raise HelpersBusy() from None
            yield
        finally:
            self._release(ticket)


_pool = HelperPool(limit=8, max_queue=32, wait_timeout=2.0)
_TERMINATE_GRACE = 1.0


def _execution_enabled() -> bool:
    """Read the execution switch only when a helper is about to launch."""
    from open_terminal import execution

    return execution.enabled()


def _prepare_helper_launch(argv: list[str], **popen_kwargs):
    """Return a cgroup-aware execution launch wrapper.

    This import deliberately happens at launch time.  The execution module is
    initialized by the service and must not become an import dependency of the
    routes that merely import this helper pool.
    """
    from open_terminal import execution

    return execution.prepare_helper(argv, **popen_kwargs)


def _helper_argv(popenargs: tuple, popen_kwargs: dict) -> list[str]:
    """Normalize the sequence form used by every trusted helper invocation."""
    if len(popenargs) == 1:
        command = popenargs[0]
    elif not popenargs and "args" in popen_kwargs:
        command = popen_kwargs.pop("args")
    else:
        raise TypeError("run_helper accepts one helper command argument")
    if not isinstance(command, (list, tuple)) or not all(
        isinstance(part, str) for part in command
    ):
        raise TypeError("helper commands must be sequences of strings")
    return list(command)


def _run_prepared_helper(
    launch, *, input, capture_output: bool, timeout, check: bool
) -> subprocess.CompletedProcess:
    """The cgroup-aware equivalent of ``subprocess.run``.

    Keep this close to the stdlib implementation so its input, timeout,
    capture, text, and ``check`` contracts remain unchanged.
    """
    try:
        with subprocess.Popen(launch.argv, **launch.kwargs) as process:
            try:
                # The launcher reports exec failure through a private status
                # pipe.  Do this before communicate: a helper may fill stdout
                # before a synchronous caller starts draining it.
                launch.started(process)
                stdout, stderr = process.communicate(input, timeout=timeout)
            except subprocess.TimeoutExpired:
                _kill_helper_process_group(process)
                # On POSIX communicate records partial output in the raised
                # exception; wait still guarantees the PID has exited before
                # its pool slot can be returned.
                process.wait()
                raise
            except BaseException:
                _kill_helper_process_group(process)
                process.wait()
                raise
            retcode = process.poll()
            if check and retcode:
                raise subprocess.CalledProcessError(
                    retcode, process.args, output=stdout, stderr=stderr
                )
        return subprocess.CompletedProcess(process.args, retcode, stdout, stderr)
    finally:
        launch.abort()


def _kill_helper_process_group(process) -> None:
    """Kill a helper's private session, including sudo's descendants."""
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass


def run_helper(
    *popenargs,
    input=None,
    capture_output: bool = False,
    timeout=None,
    check: bool = False,
    **kwargs,
) -> subprocess.CompletedProcess:
    """Blocking helper: the worker owns its slot until subprocess.run returns.

    Cancelling an outer to_thread await does not release a still-running worker's
    slot. Existing subprocess.run timeout/check/input semantics are preserved.
    """
    with _pool.sync_slot():
        if not _execution_enabled():
            return subprocess.run(
                *popenargs,
                input=input,
                capture_output=capture_output,
                timeout=timeout,
                check=check,
                **kwargs,
            )
        if input is not None:
            if kwargs.get("stdin") is not None:
                raise ValueError("stdin and input arguments may not both be used.")
            kwargs["stdin"] = subprocess.PIPE
        if capture_output:
            if kwargs.get("stdout") is not None or kwargs.get("stderr") is not None:
                raise ValueError(
                    "stdout and stderr arguments may not be used with capture_output."
                )
            kwargs["stdout"] = subprocess.PIPE
            kwargs["stderr"] = subprocess.PIPE
        # prepare_helper receives every Popen option, including pipes added by
        # subprocess.run's public convenience arguments.
        launch = _prepare_helper_launch(_helper_argv(popenargs, kwargs), **kwargs)
        return _run_prepared_helper(
            launch,
            input=input,
            capture_output=capture_output,
            timeout=timeout,
            check=check,
        )


def _signal_helper(
    process: asyncio.subprocess.Process, sig: int, *, process_group: bool = True
) -> None:
    try:
        if os.name == "posix" and process_group:
            os.killpg(process.pid, sig)
        elif sig == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()
    except ProcessLookupError:
        pass


async def _reap(spawn: asyncio.Task, *, process_group: bool = True) -> None:
    try:
        process = await spawn
    except Exception:
        log.debug("Helper failed before it could be reaped", exc_info=True)
        return
    if process.returncode is None:
        # Both legacy and managed launchers give every helper a private POSIX
        # session. SIGTERM can therefore reach sudo's fixed helper child while
        # leaving siblings in the shared helpers cgroup alone.
        _signal_helper(process, signal.SIGTERM, process_group=process_group)
    drain = asyncio.create_task(process.communicate())
    try:
        await asyncio.wait_for(asyncio.shield(drain), _TERMINATE_GRACE)
    except TimeoutError:
        _signal_helper(
            process,
            signal.SIGKILL if os.name == "posix" else signal.SIGTERM,
            process_group=process_group,
        )
        await drain


async def _finish_cleanup(task: asyncio.Task) -> None:
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    task.result()
    if cancelled:
        raise asyncio.CancelledError


@asynccontextmanager
async def open_helper(*command: str, **kwargs):
    """Lease a launched helper and reap it before returning its slot.

    Callers must finish/cancel their own pipe reader tasks before leaving this
    context. Shielding launch handles cancellation between fork and PID return.
    """
    async with _pool.slot():
        launch = (
            _prepare_helper_launch(list(command), **kwargs)
            if _execution_enabled()
            else None
        )
        if launch is None and os.name == "posix":
            kwargs["start_new_session"] = True
        if launch is None:
            argv = command
            launch_kwargs = kwargs
        else:
            argv = launch.argv
            launch_kwargs = launch.kwargs
        spawn = asyncio.create_task(asyncio.create_subprocess_exec(*argv, **launch_kwargs))
        started = None
        try:
            process = await asyncio.shield(spawn)
            if launch is not None:
                started = asyncio.create_task(asyncio.to_thread(launch.started, process))
                await asyncio.shield(started)
            yield process
        finally:
            try:
                await _finish_cleanup(
                    asyncio.create_task(_reap(spawn, process_group=True))
                )
            finally:
                if started is not None and not started.done():
                    # Do not close the launcher's status FDs while the
                    # handshake thread is reading them.  _finish_cleanup keeps
                    # this cancellation-safe and then restores cancellation.
                    try:
                        await _finish_cleanup(started)
                    except asyncio.CancelledError:
                        log.debug("Helper handshake cancelled during cleanup")
                    except Exception:
                        log.debug("Helper handshake failed during cleanup", exc_info=True)
                if launch is not None:
                    launch.abort()
