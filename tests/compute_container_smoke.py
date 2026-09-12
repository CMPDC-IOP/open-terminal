"""Run only in a disposable image with sudo and OT_NOTEBOOK_DISPOSABLE_TEST=1."""

import asyncio
import json
import os
import pwd
import shlex
import subprocess
import sys
from pathlib import Path

import httpx
import nbformat

# Remove image-specific library overrides so the test checks the common setting.
VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
for variable in VARIABLES:
    os.environ.pop(variable, None)
os.environ["OMP_NUM_THREADS"] = "3"

from open_terminal import main as api
from open_terminal.utils.service_processes import open_helper
from open_terminal.utils.user_isolation import resolve_user


def result(text):
    for line in text.replace("\r", "").splitlines():
        if "THREAD_RESULT=" in line:
            return json.JSONDecoder().raw_decode(line.split("THREAD_RESULT=", 1)[1])[0]
    return None


async def main():
    assert os.environ.get("OT_NOTEBOOK_DISPOSABLE_TEST") == "1"
    assert os.environ.get("OPEN_TERMINAL_MULTI_USER") == "true"
    username, home = resolve_user("ot-thread-smoke")
    uid = pwd.getpwnam(username).pw_uid
    probe = str(Path(home) / "probe.py")
    notebook_path = str(Path(home) / "probe.ipynb")
    script = (
        'import json,os\nprint("THREAD_RESULT=" + json.dumps({"uid": os.getuid(), '
        '"cwd": os.getcwd(), "threads": {k: os.environ.get(k) for k in '
        + repr(VARIABLES)
        + "}}))\n"
    )
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(script)])
    await asyncio.to_thread(
        subprocess.run,
        [
            "sudo",
            "-n",
            "-u",
            username,
            "--",
            sys.executable,
            "-c",
            (
                "from pathlib import Path; import sys; Path(sys.argv[1]).write_text(sys.argv[2]); "
                "Path(sys.argv[3]).write_text(sys.argv[4])"
            ),
            probe,
            script,
            notebook_path,
            nbformat.writes(notebook),
        ],
        check=True,
    )
    expected_threads = {
        key: "3" if key == "OMP_NUM_THREADS" else "1" for key in VARIABLES
    }
    expected = {"uid": uid, "cwd": home, "threads": expected_threads}
    # Cancellation must reach a helper behind sudo before the lease is returned.
    started = asyncio.Event()
    helper_pid = None

    async def sleeping_helper():
        nonlocal helper_pid
        async with open_helper(
            "sudo",
            "-n",
            "-u",
            username,
            "--",
            sys.executable,
            "-c",
            "import os,time; print(os.getpid(), flush=True); time.sleep(60)",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
        ) as process:
            helper_pid = int(await process.stdout.readline())
            started.set()
            await asyncio.Event().wait()

    helper_task = asyncio.create_task(sleeping_helper())
    await asyncio.wait_for(started.wait(), 5)
    helper_task.cancel()
    try:
        await asyncio.wait_for(helper_task, 5)
    except asyncio.CancelledError:
        pass
    try:
        os.kill(helper_pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise AssertionError("sudo helper child survived cancellation")
    print("PASS: cancellation reaps a real sudo helper child")
    command = shlex.join([sys.executable, probe])
    headers = {
        "Authorization": "Bearer " + os.environ["OPEN_TERMINAL_API_KEY"],
        "X-User-Id": "ot-thread-smoke",
        "X-Session-Id": "thread-smoke",
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api.app),
        base_url="http://test",
        headers=headers,
    ) as client:
        response = await client.post(
            "/execute", params={"wait": 10}, json={"command": command}
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["exit_code"] == 0, data
        assert result("".join(entry["data"] for entry in data["output"])) == expected, (
            data
        )
        response = await client.post(
            "/execute",
            params={"wait": 10},
            json={"command": command, "env": {"MKL_NUM_THREADS": "4"}},
        )
        data = response.json()
        assert (
            result("".join(entry["data"] for entry in data["output"]))["threads"][
                "MKL_NUM_THREADS"
            ]
            == "4"
        ), data
        print(
            "PASS: execute user switch, shared defaults, inherited and request overrides"
        )

        response = await client.post("/api/terminals")
        assert response.status_code == 200, response.text
        terminal = response.json()["id"]
        try:
            fd = api._terminal_sessions[terminal]["master_fd"]
            os.write(fd, (command + "\n").encode())
            output = ""
            async with asyncio.timeout(10):
                while result(output) is None:
                    try:
                        output += os.read(fd, 65536).decode(errors="replace")
                    except BlockingIOError:
                        await asyncio.sleep(0.01)
            assert result(output) == expected, output
            print(
                "PASS: real interactive login terminal receives the same defaults and user identity"
            )
        finally:
            assert (
                await client.delete(f"/api/terminals/{terminal}")
            ).status_code == 200

        response = await client.post("/notebooks", json={"path": notebook_path})
        assert response.status_code == 200, response.text
        notebook_id = response.json()["id"]
        try:
            response = await client.post(
                f"/notebooks/{notebook_id}/execute", json={"cell_index": 0}
            )
            assert response.status_code == 200, response.text
            data = response.json()
            assert data["status"] == "ok", data
            assert (
                result("".join(item.get("text", "") for item in data["outputs"]))
                == expected
            ), data
            print(
                "PASS: real notebook kernel receives the same defaults and user identity"
            )
        finally:
            assert (await client.delete(f"/notebooks/{notebook_id}")).status_code == 200


if __name__ == "__main__":
    asyncio.run(main())
