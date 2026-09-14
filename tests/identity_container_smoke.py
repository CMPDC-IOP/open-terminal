"""Real identity-isolation acceptance check; run only in a disposable container.

Requires root, OPEN_TERMINAL_MULTI_USER=true and OT_IDENTITY_DISPOSABLE_TEST=1.
The script never touches a host mount: it provisions throwaway accounts inside
the container's own /home and exercises the runtime identity contract.

Covers: concurrent first access, UID/GID uniqueness, container rebuild and
access-order changes, incomplete initialization retry, historical UID avoidance,
cross-user denial, and identity consistency across files, commands, PTY and
notebooks.
"""

from __future__ import annotations

import asyncio
import json
import os
import pwd
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import nbformat

HOME = Path("/home")
MAP_PATH = HOME / ".open-terminal" / "identity-map.json"

ALICE_ID = "aaaa1111-1111-1111-1111-111111111111"
BOB_ID = "bbbb2222-2222-2222-2222-222222222222"
CAROL_ID = "cccc3333-3333-3333-3333-333333333333"
DAVE_ID = "dddd4444-4444-4444-4444-444444444444"


def sh(*cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(list(cmd), check=check, capture_output=True, text=True)


def set_up_fixture() -> None:
    assert os.geteuid() == 0, "run as root inside a disposable container"
    from open_terminal.utils.user_isolation import ensure_os_user
    for user_id in (ALICE_ID, BOB_ID, CAROL_ID):
        ensure_os_user(user_id)
    historical = HOME / "dddd4444"
    historical.mkdir()
    (historical / "secret.txt").write_text("historical data")
    sh("chown", "-R", "1002:1002", str(historical))
    sh("chmod", "2770", str(historical))
    sh("groupadd", "-g", "1002", "otreuse")
    sh("useradd", "-M", "-u", "1002", "-g", "1002", "otreuse")


def cross_user_denial() -> None:
    alice = pwd.getpwnam("aaaa1111")
    carol_home = str(HOME / "cccc3333")
    for flag, operation in (("-r", "read"), ("-w", "write"), ("-x", "traverse")):
        probe = sh(
            "runuser", "-u", "aaaa1111", "--", "test", flag, carol_home, check=False
        )
        assert probe.returncode != 0, (operation, probe)
    print("PASS: cross-user read, write and traverse are denied")


def runtime_historical_conflict() -> None:
    from open_terminal.utils.user_isolation import ensure_os_user

    record = ensure_os_user(DAVE_ID)
    assert record.uid not in (1001, 1002), record
    info = os.stat(HOME / "dddd4444")
    assert (info.st_uid, info.st_gid) == (record.uid, record.gid)
    print("PASS: runtime provisioning avoids a historical UID reused by another account")


def concurrent_first_access() -> None:
    from open_terminal.utils.user_isolation import ensure_os_user

    results: list = []
    errors: list = []

    def worker():
        try:
            results.append(ensure_os_user("eeee5555-5555-5555-5555-555555555555"))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors, errors
    assert len({record.uid for record in results}) == 1
    print("PASS: concurrent first access allocates exactly one identity")


def cross_process_first_access() -> None:
    script = (
        "import json; from dataclasses import asdict; "
        "from open_terminal.utils.user_isolation import ensure_os_user; "
        "print(json.dumps(asdict(ensure_os_user('ffff6666-concurrent-processes'))))"
    )
    children = [subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ) for _ in range(8)]
    identities = []
    for child in children:
        stdout, stderr = child.communicate(timeout=30)
        assert child.returncode == 0, stderr
        identities.append(json.loads(stdout))
    assert len({(r["uid"], r["gid"]) for r in identities}) == 1
    assert all(r["state"] == "ready" for r in identities)
    print("PASS: independent processes converge on one UID/GID and ready account")


def ready_permission_recovery() -> None:
    from open_terminal.utils import user_isolation
    record = user_isolation.ensure_os_user(CAROL_ID)
    home = HOME / record.username
    os.chmod(home, 0o2777)
    user_isolation.ensure_os_user(CAROL_ID)
    assert home.stat().st_mode & 0o7777 == 0o2770
    nested = home / "interrupted" / "data.txt"
    nested.parent.mkdir(exist_ok=True)
    nested.write_text("persisted data")
    os.chown(nested, 4242, 4242)
    user_isolation._VERIFIED_HOMES.clear()
    user_isolation.ensure_os_user(CAROL_ID)
    assert (nested.stat().st_uid, nested.stat().st_gid) == (record.uid, record.gid)
    outside = Path("/tmp/outside-identity-home")
    outside.write_text("must remain root owned")
    link = home / "outside-link"
    link.symlink_to(outside)
    user_isolation._VERIFIED_HOMES.clear()
    user_isolation.ensure_os_user(CAROL_ID)
    assert outside.stat().st_uid == 0
    assert link.lstat().st_uid == record.uid
    print("PASS: unsafe home mode and stale internal owners recover without following links")


def rebuild_and_order_change() -> None:
    from open_terminal.utils.user_isolation import ensure_os_user

    first = ensure_os_user(ALICE_ID)
    second = ensure_os_user(BOB_ID)
    assert first.uid != second.uid and first.gid != second.gid
    for username in ("aaaa1111", "bbbb2222"):
        sh("userdel", username)
    again_second = ensure_os_user(BOB_ID)
    again_first = ensure_os_user(ALICE_ID)
    assert (again_first.uid, again_first.gid) == (first.uid, first.gid)
    assert (again_second.uid, again_second.gid) == (second.uid, second.gid)
    print("PASS: registry survives account recreation and access-order changes")


def incomplete_initialization_retry() -> None:
    from open_terminal.utils.user_isolation import ensure_os_user

    record = ensure_os_user(CAROL_ID)
    target = HOME / "cccc3333" / "nested"
    target.mkdir(exist_ok=True)
    (target / "data.txt").write_text("x", encoding="utf-8")
    sh("chown", "-R", "4242:4242", str(target))
    registry = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    registry["users"][CAROL_ID]["state"] = "provisioning"
    MAP_PATH.write_text(json.dumps(registry), encoding="utf-8")
    recovered = ensure_os_user(CAROL_ID)
    assert recovered.uid == record.uid
    info = os.stat(target)
    assert (info.st_uid, info.st_gid) == (record.uid, record.gid)
    print("PASS: incomplete initialization is detected and repaired on retry")


async def http_identity_consistency() -> None:
    from open_terminal.main import app

    alice = pwd.getpwnam("aaaa1111")
    headers = {
        "Authorization": "Bearer " + os.environ["OPEN_TERMINAL_API_KEY"],
        "X-User-Id": ALICE_ID,
        "X-Session-Id": "identity-smoke",
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers=headers,
        timeout=60,
    ) as client:
        written = await client.post(
            "/files/write", json={"path": "hello.txt", "content": "identity check"}
        )
        assert written.status_code == 200, written.text
        read = await client.get("/files/read", params={"path": "hello.txt"})
        assert read.status_code == 200 and read.json()["content"] == "identity check"
        denied = await client.get(
            "/files/read",
            params={"path": str(HOME / "bbbb2222" / "secret.txt")},
        )
        assert denied.status_code == 403, denied.text

        executed = await client.post(
            "/execute", params={"wait": 15}, json={"command": "id -u"}
        )
        assert executed.status_code == 200, executed.text
        output = "".join(
            entry.get("data", "")
            for entry in executed.json().get("output", [])
        )
        assert str(alice.pw_uid) in output, output

        terminal = await client.post("/api/terminals")
        assert terminal.status_code == 200, terminal.text
        terminal_id = terminal.json()["id"]
        try:
            from open_terminal import main as api

            fd = api._terminal_sessions[terminal_id]["master_fd"]
            os.write(fd, b"id -u" + bytes([10]))
            output = ""
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                try:
                    output += os.read(fd, 65536).decode(errors="replace")
                except BlockingIOError:
                    await asyncio.sleep(0.02)
                if str(alice.pw_uid) in output:
                    break
            assert str(alice.pw_uid) in output, output
        finally:
            await client.delete(f"/api/terminals/{terminal_id}")

        notebook = nbformat.v4.new_notebook(
            cells=[nbformat.v4.new_code_cell("import os; print(os.getuid())")]
        )
        notebook_path = str(HOME / "aaaa1111" / "identity.ipynb")
        Path(notebook_path).write_text(nbformat.writes(notebook), encoding="utf-8")
        sh("chown", "aaaa1111:aaaa1111", notebook_path)
        created = await client.post("/notebooks", json={"path": notebook_path})
        assert created.status_code == 200, created.text
        notebook_id = created.json()["id"]
        try:
            run = await client.post(
                f"/notebooks/{notebook_id}/execute", json={"cell_index": 0}
            )
            assert run.status_code == 200 and run.json()["status"] == "ok", run.text
            text = "".join(
                item.get("text", "")
                for item in run.json()["outputs"]
            )
            assert str(alice.pw_uid) in text, text
        finally:
            await client.delete(f"/notebooks/{notebook_id}")

    print("PASS: files, command, PTY and notebook interfaces share one identity")


async def main() -> None:
    assert os.environ.get("OT_IDENTITY_DISPOSABLE_TEST") == "1"
    assert os.environ.get("OPEN_TERMINAL_MULTI_USER") == "true"
    set_up_fixture()
    cross_user_denial()
    runtime_historical_conflict()
    concurrent_first_access()
    cross_process_first_access()
    ready_permission_recovery()
    rebuild_and_order_change()
    incomplete_initialization_retry()
    await http_identity_consistency()
    print("ALL IDENTITY CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
