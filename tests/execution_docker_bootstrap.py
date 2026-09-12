"""Prepare ONLY a disposable container's private cgroup namespace for acceptance."""

import os
import runpy
import signal
import subprocess
import sys
from pathlib import Path

if os.environ.get("OT_EXECUTION_DISPOSABLE_TEST") != "1" or os.geteuid() != 0:
    raise SystemExit("Requires an explicitly disposable root test container")
if Path("/proc/self/cgroup").read_text().strip() != "0::/":
    raise SystemExit("Requires a private cgroup namespace rooted at this container")
if not Path("/.dockerenv").exists():
    raise SystemExit("Requires Docker")

# Independent wall-clock backstop: tini exits with this child, ending the container.
signal.alarm(180)
root = Path("/sys/fs/cgroup")
quota, period = (root / "cpu.max").read_text().split()
assert quota != "max" and int(quota) / int(period) <= 2
assert int((root / "memory.max").read_text()) <= 2 * 1024**3
assert int((root / "pids.max").read_text()) <= 512
subprocess.run(["mount", "-o", "remount,rw", str(root)], check=True)
harness = root / "harness"
harness.mkdir()
for pid in (root / "cgroup.procs").read_text().split():
    (harness / "cgroup.procs").write_text(pid)
(root / "cgroup.subtree_control").write_text("+cpu +memory +pids")
test_root = root / "ot-test"
test_root.mkdir()
for name, value in {
    "cpu.max": "150000 100000",
    "memory.max": str(1536 * 1024**2),
    "memory.swap.max": "0",
    "memory.oom.group": "0",
    "pids.max": "384",
}.items():
    (test_root / name).write_text(value)
print(
    "BOOTSTRAP: private cgroup subtree ready; outer Docker limits verified", flush=True
)
runpy.run_path(sys.argv[1], run_name="__main__")
