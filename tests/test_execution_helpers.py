import asyncio
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from open_terminal.utils import service_processes as helpers


@dataclass
class _Launch:
    argv: list[str]
    kwargs: dict
    events: list[str] = field(default_factory=list)
    process: object | None = None

    def started(self, process) -> None:
        self.events.append("started")
        self.process = process

    def abort(self) -> None:
        self.events.append("abort")


def _cgroup_launches(monkeypatch):
    launches = []

    def prepare(argv, **kwargs):
        launch_kwargs = dict(kwargs)
        launch_kwargs["start_new_session"] = True
        launch = _Launch(argv, launch_kwargs)
        launches.append(launch)
        return launch

    monkeypatch.setattr(helpers, "_execution_enabled", lambda: True)
    monkeypatch.setattr(helpers, "_prepare_helper_launch", prepare)
    return launches


def test_run_helper_uses_prepared_launcher_and_preserves_run_conveniences(monkeypatch):
    launches = _cgroup_launches(monkeypatch)

    result = helpers.run_helper(
        [
            sys.executable,
            "-c",
            "import sys; sys.stdout.write(sys.stdin.read().upper())",
        ],
        input="hello",
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout == "HELLO"
    assert result.stderr == ""
    assert launches[0].kwargs["stdin"] is helpers.subprocess.PIPE
    assert launches[0].kwargs["stdout"] is helpers.subprocess.PIPE
    assert launches[0].kwargs["stderr"] is helpers.subprocess.PIPE
    assert launches[0].events == ["started", "abort"]


def test_run_helper_aborts_launcher_after_timeout_and_reaps_child(monkeypatch):
    launches = _cgroup_launches(monkeypatch)

    with pytest.raises(helpers.subprocess.TimeoutExpired):
        helpers.run_helper(
            [sys.executable, "-c", "import time; time.sleep(60)"], timeout=0.01
        )

    launch = launches[0]
    assert launch.events == ["started", "abort"]
    if os.name == "posix":
        with pytest.raises(ProcessLookupError):
            os.kill(launch.process.pid, 0)


@pytest.mark.skipif(
    os.name != "posix", reason="process-group cleanup is POSIX-specific"
)
def test_run_helper_timeout_reaps_managed_helper_descendants(monkeypatch):
    _cgroup_launches(monkeypatch)
    command = (
        "import subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        "print(child.pid, flush=True); time.sleep(60)"
    )

    with pytest.raises(helpers.subprocess.TimeoutExpired) as error:
        helpers.run_helper(
            [sys.executable, "-c", command],
            capture_output=True,
            text=True,
            timeout=0.05,
        )

    child_pid = int(error.value.output.decode().strip())
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            state = Path(f"/proc/{child_pid}/stat").read_text().split()[2]
        except FileNotFoundError:
            break
        if state == "Z":
            break
        time.sleep(0.01)
    else:
        pytest.fail("managed helper descendant survived timeout cleanup")


def test_open_helper_handshakes_before_yield_and_reaps_its_process_group(monkeypatch):
    launches = _cgroup_launches(monkeypatch)
    signal_groups = []
    original_signal = helpers._signal_helper

    def signal(process, sig, *, process_group=True):
        signal_groups.append(process_group)
        original_signal(process, sig, process_group=process_group)

    monkeypatch.setattr(helpers, "_signal_helper", signal)

    async def exercise():
        async with helpers.open_helper(
            sys.executable,
            "-c",
            "import time; time.sleep(60)",
            stdout=asyncio.subprocess.DEVNULL,
        ):
            assert launches[0].events == ["started"]

    asyncio.run(exercise())
    assert launches[0].events == ["started", "abort"]
    assert signal_groups and all(signal_groups)


def test_cancelled_open_helper_finishes_handshake_before_aborting_launcher(monkeypatch):
    launches = _cgroup_launches(monkeypatch)
    handshake_entered = threading.Event()
    allow_handshake_to_finish = threading.Event()

    async def exercise():
        task = asyncio.create_task(
            _open_sleeping_helper(
                launches, handshake_entered, allow_handshake_to_finish
            )
        )
        try:
            await asyncio.wait_for(asyncio.to_thread(handshake_entered.wait), 2)
            task.cancel()
            await asyncio.sleep(0.05)
            assert launches[0].events == ["started"]
            if os.name == "posix":
                with pytest.raises(ProcessLookupError):
                    os.kill(launches[0].process.pid, 0)
        finally:
            allow_handshake_to_finish.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)

    asyncio.run(exercise())
    assert launches[0].events == ["started", "abort"]


async def _open_sleeping_helper(launches, handshake_entered, allow_handshake_to_finish):
    def prepare(argv, **kwargs):
        launch_kwargs = dict(kwargs)
        launch_kwargs["start_new_session"] = True
        launch = _Launch(argv, launch_kwargs)

        def started(process):
            launch.events.append("started")
            launch.process = process
            handshake_entered.set()
            assert allow_handshake_to_finish.wait(2)

        launch.started = started
        launches.append(launch)
        return launch

    # The fixture's prepare function is installed by replacing it for only this
    # operation, after the launch list has been created.
    original_prepare = helpers._prepare_helper_launch
    helpers._prepare_helper_launch = prepare
    try:
        async with helpers.open_helper(
            sys.executable, "-c", "import time; time.sleep(60)"
        ):
            pytest.fail("the blocked status handshake must not yield")
    finally:
        helpers._prepare_helper_launch = original_prepare
