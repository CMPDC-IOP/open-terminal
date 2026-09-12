"""Actual privilege-drop smoke test, only in a disposable root Linux container.

The cgroup descriptor here is a regular-file fixture. This verifies the trusted
launcher protocol/identity/capabilities, NOT kernel cgroup resource enforcement.
"""

import json
import os
import pwd
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock

if os.environ.get("OT_EXECUTION_DISPOSABLE_TEST") != "1" or os.geteuid() != 0:
    raise SystemExit(
        "Requires OT_EXECUTION_DISPOSABLE_TEST=1 in a disposable root container"
    )

from open_terminal.execution.manager import Manager
from open_terminal.execution.policy import ExecutionPolicy, Limits

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    (root / "cgroup.procs").touch()
    limits = Limits(100, 32 * 1024 * 1024, 16)
    policy = ExecutionPolicy(root, limits, limits, limits * 2, limits, 2, 1)
    tree = Mock()
    tree.create_task.return_value = root
    tree.is_empty.return_value = True
    runtime = Manager(policy, tree)
    code = """
import json, os
status = dict(line.split(':', 1) for line in open('/proc/self/status') if ':' in line)
print(json.dumps({'uid':os.getuid(), 'gid':os.getgid(), 'groups':os.getgroups(),
'caps':{k:status[k].strip() for k in ('CapInh','CapPrm','CapEff','CapBnd','CapAmb')},
'nnp':status['NoNewPrivs'].strip(), 'cwd':os.getcwd(), 'secret':os.getenv('OPEN_TERMINAL_API_KEY')}))
"""
    os.environ["OPEN_TERMINAL_API_KEY"] = "must-not-inherit"
    launch = runtime.prepare(
        "smoke-owner", "nobody", ["/usr/local/bin/python", "-I", "-c", code], cwd="/"
    )
    try:
        process = subprocess.Popen(
            launch.argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **launch.kwargs
        )
        launch.started(process)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stderr
        data = json.loads(stdout)
        account = pwd.getpwnam("nobody")
        assert (data["uid"], data["gid"]) == (account.pw_uid, account.pw_gid)
        assert not data["groups"]
        assert all(int(value, 16) == 0 for value in data["caps"].values())
        assert data["nnp"] == "1"
        assert data["cwd"] == "/" and data["secret"] is None
    finally:
        launch.abort()
    assert not runtime._tasks
    assert (root / "cgroup.procs").read_text() == "0"
    print(
        "Launcher UID/GID, groups, capabilities, no_new_privs, environment and handshake passed; cgroup resource limits NOT tested."
    )
