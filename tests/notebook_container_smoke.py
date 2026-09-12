"""Real multi-user acceptance check; run only in a disposable Linux container.

Requires OPEN_TERMINAL_MULTI_USER=true and permission to provision OS users.
Set OT_NOTEBOOK_DISPOSABLE_TEST=1 only for that disposable container.
The test creates users and files in the container's own /home, never on a host
mount. Import the workspace package over the image's installed package.
"""

import asyncio
import json
import os
import pwd
import subprocess
from pathlib import Path

import httpx
import nbformat

from open_terminal.main import app
from open_terminal.utils import notebooks
from open_terminal.utils.user_isolation import resolve_user


def as_user(username, code, *args):
    return subprocess.run(
        ["sudo", "-n", "-u", username, "--", "python", "-c", code, *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


async def main():
    assert os.environ.get("OT_NOTEBOOK_DISPOSABLE_TEST") == "1", (
        "Use a disposable test container"
    )
    assert os.environ.get("OPEN_TERMINAL_MULTI_USER") == "true"
    users = {}
    for user_id in ("ot-alice-smoke", "ot-bob-smoke"):
        username, home = resolve_user(user_id)
        notebook = nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_code_cell(
                    "import json, os\n"
                    "from pathlib import Path\n"
                    "Path('kernel-output.txt').write_text('created by kernel')\n"
                    "print(json.dumps({'uid': os.getuid(), 'gid': os.getgid(), 'pid': os.getpid(), "
                    "'cwd': os.getcwd(), 'home': os.environ.get('HOME')}))"
                )
            ]
        )
        path = str(Path(home) / "private notebooks" / "test.ipynb")
        as_user(
            username,
            "from pathlib import Path; import sys; "
            "Path(sys.argv[1]).parent.mkdir(mode=0o700); "
            "Path(sys.argv[1]).write_text(sys.argv[2])",
            path,
            nbformat.writes(notebook),
        )
        users[user_id] = (username, home, path)

    alice_id, bob_id = users
    alice, home, path = users[alice_id]
    notebook_directory = str(Path(path).parent)
    bob_path = users[bob_id][2]
    headers = {"X-User-Id": alice_id, "X-Session-Id": "notebook-smoke"}
    if os.environ.get("OPEN_TERMINAL_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["OPEN_TERMINAL_API_KEY"]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers=headers,
        timeout=60,
    ) as client:
        missing_user = await client.post(
            "/notebooks",
            json={"path": path},
            headers={"X-User-Id": ""},
        )
        assert missing_user.status_code == 403, missing_user.text
        response = await client.post("/notebooks", json={"path": path})
        assert response.status_code == 200, response.text
        session = response.json()["id"]
        kernel_runtime = notebooks._sessions[session].runtime_directory
        try:
            for overrides in ({"X-User-Id": bob_id}, {"X-Session-Id": "other"}):
                for method, suffix, payload in (
                    ("GET", "", None),
                    ("DELETE", "", None),
                    ("POST", "/execute", {"cell_index": 0}),
                ):
                    denied = await client.request(
                        method,
                        f"/notebooks/{session}{suffix}",
                        headers=overrides,
                        json=payload,
                    )
                    assert denied.status_code == 404, denied.text
            denied = await client.post("/notebooks", json={"path": bob_path})
            assert denied.status_code == 403, denied.text
            link = str(Path(home) / "other.ipynb")
            as_user(
                alice,
                "import os, sys; os.symlink(sys.argv[1], sys.argv[2])",
                bob_path,
                link,
            )
            denied = await client.post("/notebooks", json={"path": link})
            assert denied.status_code in (400, 403, 404), denied.text
            response = await client.post(
                f"/notebooks/{session}/execute", json={"cell_index": 0}
            )
            assert response.status_code == 200, response.text
            result = response.json()
            assert result["status"] == "ok", result
            output = "".join(item.get("text", "") for item in result["outputs"])
            identity = json.loads(output.strip())
            kernel_pid = identity.pop("pid")
            account = pwd.getpwnam(alice)
            assert identity == {
                "uid": account.pw_uid,
                "gid": account.pw_gid,
                "cwd": notebook_directory,
                "home": home,
            }, identity
            for filename in (path, str(Path(notebook_directory) / "kernel-output.txt")):
                owner = json.loads(
                    as_user(
                        alice,
                        "import json, os, sys; s=os.stat(sys.argv[1]); "
                        "print(json.dumps([s.st_uid, s.st_gid]))",
                        filename,
                    )
                )
                assert owner == [account.pw_uid, account.pw_gid], owner
            saved = nbformat.reads(
                as_user(
                    alice,
                    "from pathlib import Path; import sys; "
                    "print(Path(sys.argv[1]).read_text())",
                    path,
                ),
                as_version=4,
            )
            assert saved.cells[0].execution_count == 1
            denied = await client.post(
                f"/notebooks/{session}/execute",
                json={"cell_index": 0, "source": f"open({bob_path!r}).read()"},
            )
            assert denied.status_code == 200, denied.text
            assert denied.json()["status"] == "error", denied.text
            assert "PermissionError" in denied.text, denied.text
            assert (await client.get(f"/notebooks/{session}")).json()[
                "status"
            ] == "ready"
            print(
                "PASS: owner/context authorization, path/symlink denial, real kernel UID/GID, "
                "working directory, HOME, notebook save, output ownership, OS read isolation"
            )
        finally:
            stopped = await client.delete(f"/notebooks/{session}")
            assert stopped.status_code == 200, stopped.text
        assert (await client.get(f"/notebooks/{session}")).status_code == 404
        try:
            os.kill(kernel_pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise AssertionError("Kernel process survived session deletion")
        assert not Path(kernel_runtime).exists()
        print(
            "PASS: owner can stop kernel; process exited, session and connection directory removed"
        )


if __name__ == "__main__":
    asyncio.run(main())
