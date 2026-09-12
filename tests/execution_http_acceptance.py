"""Real HTTP/PTY/notebook acceptance inside execution_docker_bootstrap.py."""

import asyncio
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import httpx
import nbformat
from websockets.asyncio.client import connect

from open_terminal.execution.policy import load_policy

ROOT = Path("/sys/fs/cgroup/ot-test")
MIB = 1024**2
TOKEN = "disposable-acceptance-only"


def limits(cpu, memory, pids):
    return {
        "cpu_millis": cpu,
        "memory_bytes": memory * MIB,
        "pids": pids,
    }


def events(path):
    return {
        k: int(v) for k, v in (line.split() for line in path.read_text().splitlines())
    }


def checked(response):
    assert response.status_code == 200, (response.status_code, response.text)
    return response.json()


async def main():
    assert os.environ.get("OT_EXECUTION_DISPOSABLE_TEST") == "1"
    policy = {
        "cgroup_root": str(ROOT),
        "service": limits(500, 512, 128),
        "helpers": limits(250, 256, 96),
        "compute": limits(750, 768, 160),
        "user": limits(375, 384, 80),
        "max_tasks": 8,
        "max_user_tasks": 4,
        "max_runtime": 30,
    }
    policy_path = Path("/run/ot-acceptance-policy.json")
    policy_path.write_text(json.dumps(policy))
    runtime_policy = load_policy(policy_path)
    env = {
        **os.environ,
        "OPEN_TERMINAL_EXECUTION_MODE": "cgroup",
        "OPEN_TERMINAL_EXECUTION_POLICY_FILE": str(policy_path),
        "OPEN_TERMINAL_MULTI_USER": "true",
        "OPEN_TERMINAL_API_KEY": TOKEN,
        "OPEN_TERMINAL_ENABLE_TERMINAL": "true",
        "OPEN_TERMINAL_ENABLE_NOTEBOOKS": "true",
        "OPEN_TERMINAL_LOG_DIR": "/run/ot-acceptance-logs",
    }
    # Standalone harness: log lifetime is closed in the server cleanup below.
    log = open("/run/ot-acceptance-server.log", "w+")  # noqa: ASYNC230, SIM115
    server = subprocess.Popen(  # noqa: ASYNC220 — start before HTTP load; cleanup is bounded
        [
            sys.executable,
            "-m",
            "uvicorn",
            "open_terminal.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
        env=env,
        stdout=log,
        stderr=log,
    )
    report = {}
    try:
        headers = {
            "Authorization": "Bearer " + TOKEN,
            "X-User-Id": "alice-acceptance",
            "X-Session-Id": "one",
        }
        async with httpx.AsyncClient(
            base_url="http://127.0.0.1:8000", headers=headers, timeout=20
        ) as a:
            for _ in range(100):
                if server.poll() is not None:
                    raise AssertionError("API startup failed")
                try:
                    if (await a.get("/health")).status_code < 500:
                        break
                except httpx.ConnectError:
                    pass
                await asyncio.sleep(0.2)
            else:
                raise AssertionError("API startup timed out")

            # Bounded CPU workers end independently even if runtime enforcement fails.
            worker = (
                "import time\ne=time.monotonic()+35\nwhile time.monotonic()<e: pass\n"
            )
            worker_command = shlex.join([sys.executable, "-c", worker])
            notebook = nbformat.v4.new_notebook(
                cells=[nbformat.v4.new_code_cell("print('ready')")]
            )
            checked(
                await a.post(
                    "/home-files/upload",
                    data={"path": "probe.ipynb"},
                    files={"file": ("probe.ipynb", nbformat.writes(notebook).encode())},
                )
            )
            session = checked(
                await a.post(
                    "/notebooks",
                    json={"path": "probe.ipynb"},
                )
            )
            cell = checked(
                await a.post(
                    f"/notebooks/{session['id']}/execute",
                    json={
                        "cell_index": 0,
                        "source": "import subprocess,sys,os\np=subprocess.Popen([sys.executable,'-c',"
                        + repr(worker)
                        + "])\nprint(os.getpid(),p.pid)",
                    },
                )
            )
            assert cell["status"] == "ok", cell
            terminal = checked(
                await a.post("/api/terminals")
            )
            async with connect(
                f"ws://127.0.0.1:8000/api/terminals/{terminal['id']}",
                additional_headers={
                    "X-User-Id": headers["X-User-Id"],
                    "X-Session-Id": "one",
                },
            ) as ws:
                await ws.send(json.dumps({"type": "auth", "token": TOKEN}))
                await ws.send((worker_command + "\n").encode())
                commands = [checked(await a.post(
                    "/execute?wait=0", json={"command": worker_command}
                ))]
                owner = (
                    ROOT
                    / "compute"
                    / (
                        "owner-"
                        + hashlib.sha256(headers["X-User-Id"].encode()).hexdigest()
                    )
                )
                await asyncio.sleep(1)
                groups = list(owner.glob("task-*"))
                assert len(groups) == 3, groups
                memberships = {
                    p.name: (p / "cgroup.procs").read_text().split() for p in groups
                }
                assert all(memberships.values()), memberships
                assert (owner / "cpu.max").read_text().split()[0] == str(
                    policy["user"]["cpu_millis"] * 100
                )
                assert (owner / "memory.max").read_text().strip() == str(
                    policy["user"]["memory_bytes"]
                )
                assert (owner / "pids.max").read_text().strip() == str(
                    policy["user"]["pids"]
                )
                assert (owner / "memory.swap.max").read_text().strip() == "0"
                leaf_controls = {
                    group.name: {
                        name: (group / name).read_text().strip()
                        for name in (
                            "cpu.max",
                            "memory.max",
                            "pids.max",
                            "memory.swap.max",
                            "memory.oom.group",
                        )
                    }
                    for group in groups
                }
                assert all(
                    controls
                    == {
                        "cpu.max": "max 100000",
                        "memory.max": "max",
                        "pids.max": "max",
                        "memory.swap.max": "max",
                        "memory.oom.group": "1",
                    }
                    for controls in leaf_controls.values()
                ), leaf_controls
                t0 = time.monotonic()
                before = events(owner / "cpu.stat")
                await asyncio.sleep(4)
                after = events(owner / "cpu.stat")
                elapsed = time.monotonic() - t0
                used_cores = (
                    (after["usage_usec"] - before["usage_usec"]) / 1e6 / elapsed
                )
                assert 0.1 < used_cores <= policy["user"]["cpu_millis"] / 1000 * 1.2, (
                    used_cores
                )
                assert after["nr_throttled"] > before["nr_throttled"]
                report["cross_entry_cpu"] = {
                    "cores": used_cores,
                    "elapsed": elapsed,
                    "memberships": memberships,
                    "leaf_controls": leaf_controls,
                    "limit_cores": policy["user"]["cpu_millis"] / 1000,
                }
                bheaders = {**headers, "X-User-Id": "bob-acceptance"}
                bcommand = checked(
                    await a.post(
                        "/execute?wait=0",
                        headers=bheaders,
                        json={"command": worker_command},
                    )
                )
                blob = b"upload-under-compute-load\n" * 32768
                started = time.monotonic()
                checked(
                    await a.post(
                        "/home-files/upload",
                        headers=bheaders,
                        data={"path": "payload.bin"},
                        files={"file": ("payload.bin", blob)},
                    )
                )
                downloaded = await a.get(
                    "/home-files/content",
                    headers=bheaders,
                    params={"path": "payload.bin"},
                )
                assert downloaded.status_code == 200 and downloaded.content == blob
                checked(await a.get("/home-files", headers=bheaders))
                report["upload_under_two_user_load"] = {
                    "bytes": len(blob),
                    "seconds": time.monotonic() - started,
                    "sha256": hashlib.sha256(blob).hexdigest(),
                }

                async def parallel_upload(index):
                    path = f"parallel-{index}.bin"
                    checked(
                        await a.post(
                            "/home-files/upload",
                            headers=bheaders,
                            data={"path": path},
                            files={"file": (path, blob)},
                        )
                    )
                    content = await a.get(
                        "/home-files/content", headers=bheaders, params={"path": path}
                    )
                    assert content.status_code == 200 and content.content == blob

                uploads = asyncio.gather(*(parallel_upload(i) for i in range(12)))
                helper_peak = 0
                while not uploads.done():
                    helper_peak = max(
                        helper_peak,
                        int((ROOT / "helpers" / "pids.current").read_text()),
                    )
                    await asyncio.sleep(0.01)
                await uploads
                assert 0 < helper_peak <= policy["helpers"]["pids"], helper_peak
                assert events(ROOT / "helpers" / "memory.events")["oom_kill"] == 0
                report["parallel_uploads"] = {
                    "count": 12,
                    "bytes_each": len(blob),
                    "observed_helper_pids_peak": helper_peak,
                }
                for result, h in ((commands[0], headers), (bcommand, bheaders)):
                    checked(await a.delete(f"/execute/{result['id']}", headers=h))
                checked(await a.delete(f"/notebooks/{session['id']}"))
                checked(await a.delete(f"/api/terminals/{terminal['id']}"))
            for _ in range(100):
                if events(ROOT / "compute" / "cgroup.events")["populated"] == 0:
                    break
                await asyncio.sleep(0.05)
            assert events(ROOT / "compute" / "cgroup.events")["populated"] == 0
            report["cleanup"] = "compute populated=0 after HTTP stops"
            again = checked(await a.post("/execute?wait=5", json={"command": "true"}))
            assert again["exit_code"] == 0, again
            report["quota_reuse"] = "passed"
            # Real deadlines apply to all three entries, including an idle kernel/PTY.
            deadline_nb = checked(
                await a.post("/notebooks", json={"path": "probe.ipynb"})
            )
            deadline_pty = checked(await a.post("/api/terminals"))
            sleeper = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(45)"
            deadline_cmd = checked(
                await a.post(
                    "/execute?wait=0",
                    json={"command": shlex.join([sys.executable, "-c", sleeper])},
                )
            )
            print(
                "HTTP_PROGRESS: testing actual command, idle PTY and notebook deadlines",
                flush=True,
            )
            stop_by = (
                time.monotonic()
                + runtime_policy.max_runtime
                + runtime_policy.terminate_grace
                + 3
            )
            while time.monotonic() < stop_by:
                cmd_status = checked(
                    await a.get(f"/execute/{deadline_cmd['id']}/status")
                )
                nb_status = checked(await a.get(f"/notebooks/{deadline_nb['id']}"))
                terminal_status = await a.get(f"/api/terminals/{deadline_pty['id']}")
                if (
                    cmd_status["execution"]["state"] == "finished"
                    and nb_status["execution"]["state"] == "finished"
                    and terminal_status.status_code == 404
                ):
                    break
                await asyncio.sleep(0.5)
            else:
                raise AssertionError("Three-entry deadline cleanup timed out")
            for entry in (cmd_status, nb_status):
                execution = entry["execution"]
                assert execution["end_reason"] == "timed_out", execution
                assert (
                    0
                    <= execution["finished_at"] - execution["deadline"]
                    < runtime_policy.terminate_grace + 1
                ), execution
            assert events(ROOT / "compute" / "cgroup.events")["populated"] == 0
            expired_cell = await a.post(
                f"/notebooks/{deadline_nb['id']}/execute", json={"cell_index": 0}
            )
            assert expired_cell.status_code == 409, expired_cell.text
            checked(await a.delete(f"/notebooks/{deadline_nb['id']}"))
            report["deadlines"] = {
                "command": cmd_status["execution"],
                "notebook": nb_status["execution"],
                "terminal": "expired terminal returns 404; compute populated=0",
                "expired_cell_status": 409,
            }
        print("HTTP_ACCEPTANCE=" + json.dumps(report), flush=True)
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        log.seek(0)
        print("API_LOG=" + log.read(), flush=True)
        log.close()


if __name__ == "__main__":
    asyncio.run(main())
