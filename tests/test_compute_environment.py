import asyncio
import json
import os
import shlex
import sys
from types import SimpleNamespace

from jupyter_client.manager import AsyncKernelManager

from open_terminal import config
from open_terminal.utils import compute_environment, notebooks, runner


def test_compute_threads_cannot_be_configured(monkeypatch):
    monkeypatch.setattr(config, "_config", {"compute_threads": 3})
    monkeypatch.setenv("OPEN_TERMINAL_COMPUTE_THREADS", "7")
    resolved = compute_environment.with_compute_thread_defaults(
        {
            "OPEN_TERMINAL_COMPUTE_THREADS": "7",
            "OMP_NUM_THREADS": "9",
            "UNCHANGED": "yes",
        }
    )

    assert resolved["OMP_NUM_THREADS"] == "9"
    assert resolved["OPENBLAS_NUM_THREADS"] == "1"
    assert resolved["MKL_NUM_THREADS"] == "1"
    assert resolved["NUMEXPR_NUM_THREADS"] == "1"
    assert resolved["VECLIB_MAXIMUM_THREADS"] == "1"
    assert resolved["BLIS_NUM_THREADS"] == "1"
    assert resolved["UNCHANGED"] == "yes"


def test_pipe_runner_starts_a_subprocess_with_fixed_compute_defaults(monkeypatch):
    monkeypatch.setenv("OPEN_TERMINAL_COMPUTE_THREADS", "7")
    monkeypatch.setattr(runner, "_PTY_AVAILABLE", False)
    monkeypatch.setattr(runner, "_WINPTY_AVAILABLE", False)

    class Log:
        entries: list[str]

        def __init__(self):
            self.entries = []

        async def write(self, entry: str) -> None:
            self.entries.append(entry)

    script = (
        "import json, os; print(json.dumps({key: os.environ[key] for key in "
        + repr(compute_environment.COMPUTE_THREAD_VARIABLES)
        + "}))"
    )

    async def exercise() -> dict[str, str]:
        process = await runner.create_runner(
            f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}",
            None,
            {"PATH": os.environ.get("PATH", os.defpath)},
        )
        log = Log()
        await asyncio.gather(process.read_output(log), process.wait())
        return json.loads(json.loads(log.entries[0])["data"])

    assert asyncio.run(exercise()) == {
        variable: "1" for variable in compute_environment.COMPUTE_THREAD_VARIABLES
    }


def test_pty_sudo_launch_installs_request_environment_after_privilege_drop(
    monkeypatch,
):
    captured = {}

    class FakeProcess:
        pid = 123

    def fake_popen(*args, **kwargs):
        captured["command"] = args[0]
        captured["env"] = kwargs["env"]
        return FakeProcess()

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    process = runner.PtyRunner(
        "echo ready",
        "/work",
        {"PATH": "/custom", "OMP_NUM_THREADS": "8"},
        run_as_user="isolated-user",
        user_env={"UNTRUSTED_REQUEST_VALUE": "present"},
    )
    process.close()

    assert captured["env"] == {
        "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
        "LANG": "C.UTF-8",
        "TERM": os.environ.get("TERM", "xterm-256color"),
    }
    assert "sudo -H -u isolated-user -- env" in captured["command"]
    assert "OMP_NUM_THREADS=8" in captured["command"]
    assert "OPENBLAS_NUM_THREADS=1" in captured["command"]
    assert "UNTRUSTED_REQUEST_VALUE=present" in captured["command"]


def test_notebook_sudo_kernel_launch_only_forwards_compute_defaults(monkeypatch):
    async def base_pre_start_kernel(self, **kwargs):
        return ["python", "-m", "fake_kernel"], {
            "cwd": "/ignored",
            "env": {"OMP_NUM_THREADS": "8", "UNTRUSTED": "excluded"},
        }

    monkeypatch.setattr(
        AsyncKernelManager, "_async_pre_start_kernel", base_pre_start_kernel
    )
    monkeypatch.setattr(notebooks, "_transfer_connection_file", lambda *args: None)
    manager_class = notebooks._user_kernel_manager_class(
        "alice", "/home/alice", "/home/alice/work", "/tmp/kernel.json"
    )
    monkeypatch.setattr(manager_class, "kernel_spec", SimpleNamespace(env={}))
    manager = manager_class()

    command, launch_kwargs = asyncio.run(manager._async_pre_start_kernel())

    assert launch_kwargs == {"cwd": None, "env": notebooks._SUDO_ENV}
    assert "OMP_NUM_THREADS=8" in command
    assert "OPENBLAS_NUM_THREADS=1" in command
    assert "UNTRUSTED=excluded" not in command
