"""Real cgroup v2 acceptance checks for a disposable root container only.

The caller must create and delegate a clean, finite
``/sys/fs/cgroup/ot-test`` subtree before invoking this file.  This runner does
not mount cgroups, alter the host hierarchy, or start Docker.  It exercises the
production ``CgroupTree`` and ``Manager`` against the kernel and writes its
measurements to ``OT_EXECUTION_EVIDENCE_PATH`` (or
``/tmp/open-terminal-kernel-acceptance.json``).
"""

from __future__ import annotations

import json
import os
import selectors
import subprocess
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

if os.environ.get("OT_EXECUTION_DISPOSABLE_TEST") != "1" or os.geteuid() != 0:
    raise SystemExit(
        "Requires OT_EXECUTION_DISPOSABLE_TEST=1 as root in a disposable container"
    )

from open_terminal.execution.cgroups import CgroupTree
from open_terminal.execution.manager import Manager
from open_terminal.execution.policy import ExecutionPolicy, Limits

MiB = 1024 * 1024
ROOT = Path(os.environ.get("OT_EXECUTION_CGROUP_ROOT", "/sys/fs/cgroup/ot-test"))
EVIDENCE_PATH = Path(
    os.environ.get(
        "OT_EXECUTION_EVIDENCE_PATH", "/tmp/open-terminal-kernel-acceptance.json"
    )
)


class RecordingCgroupTree(CgroupTree):
    """Production tree with an audit trail of the real populated readbacks."""

    def __init__(self, policy: ExecutionPolicy):
        super().__init__(policy)
        self.empty_readbacks: dict[str, list[bool]] = {}

    def is_empty(self, path: Path) -> bool:
        empty = super().is_empty(path)
        self.empty_readbacks.setdefault(str(path), []).append(empty)
        return empty


def _limits(limits: Limits) -> dict[str, int]:
    return {
        "cpu_millis": limits.cpu_millis,
        "memory_bytes": limits.memory_bytes,
        "pids": limits.pids,
    }


def _events(path: Path) -> dict[str, int]:
    return {
        key: int(value)
        for key, value in (line.split() for line in path.read_text().splitlines())
    }


def _task_measurement(tree: CgroupTree, path: Path) -> dict[str, Any]:
    """Capture a task leaf's lifecycle controls and inherited budgets."""
    controls = {
        name: (path / name).read_text().strip()
        for name in ("cpu.max", "memory.max", "pids.max", "memory.swap.max")
    }
    _assert(
        controls
        == {
            "cpu.max": "max 100000",
            "memory.max": "max",
            "pids.max": "max",
            "memory.swap.max": "max",
        },
        "task leaf unexpectedly has its own resource budget",
    )
    _assert(
        (path / "memory.oom.group").read_text().strip() == "1",
        "task leaf is not an OOM kill unit",
    )
    return {
        "controls": controls,
        "memory_oom_group": (path / "memory.oom.group").read_text().strip(),
        "usage": tree.task_usage(path),
        "events": _events(path / "cgroup.events"),
    }


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _wait_finished(launch, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = launch.snapshot()
        if snapshot["state"] == "finished":
            return snapshot
        time.sleep(0.025)
    raise AssertionError(f"task did not finish within {timeout}s: {launch.snapshot()}")


def _read_json_line(
    process: subprocess.Popen[str], timeout: float = 3
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        while time.monotonic() < deadline:
            ready = selector.select(deadline - time.monotonic())
            if ready:
                line = process.stdout.readline()
                if line:
                    return json.loads(line)
            if process.poll() is not None:
                stderr = process.stderr.read()
                raise AssertionError(
                    "workload exited before its readiness record: "
                    f"{process.returncode}; {stderr}"
                )
    raise AssertionError("workload did not emit its readiness record")


def _pid_gone(pid: int) -> bool:
    """A short-lived acceptance container makes /proc identity sufficient here."""
    return not Path(f"/proc/{pid}").exists()


def _policy() -> ExecutionPolicy:
    # Totals: 1000 millicores, 960 MiB, and 328 PIDs/threads.
    # This remains below the finite envelope requested for the disposable image.
    service = Limits(100, 128 * MiB, 48)
    helpers = Limits(100, 64 * MiB, 24)
    compute = Limits(800, 768 * MiB, 256)
    return ExecutionPolicy(
        ROOT,
        service,
        helpers,
        compute,
        Limits(500, 288 * MiB, 48),
        max_tasks=2,
        max_user_tasks=2,
        start_timeout=5,
        cleanup_timeout=5,
        max_runtime=6,
        terminate_grace=0.3,
        max_queue=0,
        max_user_queue=0,
        queue_timeout=1,
    )


def _prepared_root() -> dict[str, str]:
    return {
        name: (ROOT / name).read_text().strip()
        for name in (
            "cpu.max",
            "memory.max",
            "memory.swap.max",
            "pids.max",
            "cgroup.controllers",
            "cgroup.type",
        )
    }


def _start(manager: Manager, _owner: str, code: str, *, defer_reaper: bool = False):
    # Manager deliberately keeps a UID-to-owner binding for its full lifetime.
    # Every scenario uses nobody, so retain one owner identity across them while
    # still creating fresh task IDs and fresh cgroups for every workload.
    launch = manager.prepare(
        "kernel-acceptance-owner",
        "nobody",
        [sys.executable, "-I", "-c", code],
        cwd="/",
    )
    if defer_reaper:
        # Keep an OOM task directory present long enough to read memory.events.
        # _advance() below still performs production cleanup through CgroupTree.
        launch._reaper_started = True
    process = subprocess.Popen(
        launch.argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **launch.kwargs,
    )
    launch.started(process)
    return launch, process


CPU_WORKLOAD = r"""
import json
import multiprocessing
import time

def burn(until):
    value = 0
    while time.monotonic() < until:
        value = (value * 1103515245 + 12345) & 0x7fffffff

until = time.monotonic() + 5
workers = [multiprocessing.Process(target=burn, args=(until,)) for _ in range(3)]
for worker in workers:
    worker.start()
print(json.dumps({"workers": len(workers)}), flush=True)
for worker in workers:
    worker.join()
"""

PIDS_AGGREGATE_WORKLOAD = r"""
import json
import threading
import time

stop = threading.Event()
workers = []
error = None
for attempt in range(24):
    try:
        worker = threading.Thread(target=stop.wait)
        worker.start()
        workers.append(worker)
    except RuntimeError as caught:
        error = type(caught).__name__
        break
print(json.dumps({"attempts": 24, "started": len(workers), "error": error}), flush=True)
stop.wait(4)
"""

MEMORY_AGGREGATE_WORKLOAD = r"""
import time
value = bytearray(160 * 1024 * 1024)
for index in range(0, len(value), 4096):
    value[index] = 1
print("allocated", flush=True)
time.sleep(4)
"""

OOM_WORKLOAD = r"""
import time
print("allocating", flush=True)
value = bytearray(320 * 1024 * 1024)
for index in range(0, len(value), 4096):
    value[index] = 1
print("unexpected-allocation-success", flush=True)
time.sleep(2)
"""

DEADLINE_WORKLOAD = r"""
import json
import os
import signal
import time

read_fd, write_fd = os.pipe()
child = os.fork()
if child == 0:
    os.close(read_fd)
    os.setsid()
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    os.write(write_fd, b"1")
    os.close(write_fd)
    time.sleep(8)
    os._exit(0)
os.close(write_fd)
os.read(read_fd, 1)
os.close(read_fd)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
print(json.dumps({"setsid_descendant": child}), flush=True)
time.sleep(8)
"""

ORPHAN_WORKLOAD = r"""
import json
import os
import signal
import time

read_fd, write_fd = os.pipe()
child = os.fork()
if child == 0:
    os.close(read_fd)
    os.setsid()
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    os.write(write_fd, b"1")
    os.close(write_fd)
    time.sleep(6)
    os._exit(0)
os.close(write_fd)
os.read(read_fd, 1)
os.close(read_fd)
print(json.dumps({"orphan": child}), flush=True)
os._exit(0)
"""


def _cpu_aggregate(manager: Manager, tree: CgroupTree) -> dict[str, Any]:
    first, first_process = _start(manager, "cpu-owner", CPU_WORKLOAD)
    second, second_process = _start(manager, "cpu-owner", CPU_WORKLOAD)
    try:
        _read_json_line(first_process)
        _read_json_line(second_process)
        owner = first.path.parent
        _assert(owner == second.path.parent, "same owner did not share a parent cgroup")
        before = _events(owner / "cpu.stat")
        started = time.monotonic()
        time.sleep(2)
        elapsed = time.monotonic() - started
        after = _events(owner / "cpu.stat")
        usage_delta = after["usage_usec"] - before["usage_usec"]
        throttled_delta = after.get("nr_throttled", 0) - before.get("nr_throttled", 0)
        # The 500m owner permits 500,000 usec each second. A 75ms boundary
        # allowance covers sampling partway through a 100ms cpu.max period.
        allowed_usage = manager.policy.user.cpu_millis * 1000 * elapsed + 75_000
        _assert(throttled_delta > 0, "aggregate user CPU cgroup never throttled")
        _assert(
            usage_delta <= allowed_usage,
            f"aggregate CPU exceeded the user budget: {usage_delta}us/{elapsed}s",
        )
        return {
            "owner_cpu_max": (owner / "cpu.max").read_text().strip(),
            "elapsed_seconds": elapsed,
            "usage_usec_delta": usage_delta,
            "usage_usec_allowed": allowed_usage,
            "nr_throttled_delta": throttled_delta,
            "first_task": _task_measurement(tree, first.path),
            "second_task": _task_measurement(tree, second.path),
        }
    finally:
        first.stop(reason="cancelled")
        second.stop(reason="cancelled")


def _user_parent_pids(manager: Manager, tree: CgroupTree) -> dict[str, Any]:
    first, first_process = _start(
        manager, "pids-aggregate-owner", PIDS_AGGREGATE_WORKLOAD
    )
    second = second_process = None
    try:
        first_result = _read_json_line(first_process)
        _assert(
            first_result["started"] == 24,
            "first bounded PID workload did not start all requested threads",
        )
        second, second_process = _start(
            manager, "pids-aggregate-owner", PIDS_AGGREGATE_WORKLOAD
        )
        second_result = _read_json_line(second_process)
        owner = first.path.parent
        _assert(owner == second.path.parent, "PID aggregate tasks lack a shared parent")
        parent_usage = {
            "pids_current": int((owner / "pids.current").read_text()),
            "pids_max": (owner / "pids.max").read_text().strip(),
            "pids_events": _events(owner / "pids.events"),
        }
        _assert(second_result["started"] < 24, "parent pids.max did not deny task two")
        _assert(
            parent_usage["pids_events"].get("max", 0) > 0,
            "parent pids.events did not record aggregate denial",
        )
        _assert(
            parent_usage["pids_current"] <= manager.policy.user.pids,
            "parent exceeded its pids.max",
        )
        return {
            "first_workload": first_result,
            "second_workload": second_result,
            "parent": parent_usage,
            "first_task": _task_measurement(tree, first.path),
            "second_task": _task_measurement(tree, second.path),
        }
    finally:
        first.stop(reason="cancelled")
        if second is not None:
            second.stop(reason="cancelled")


def _user_parent_memory(manager: Manager, tree: CgroupTree) -> dict[str, Any]:
    # Each bounded allocation fits in an unlimited task leaf. The 288 MiB
    # owner budget must be the source of the OOM.
    first, first_process = _start(
        manager, "memory-aggregate-owner", MEMORY_AGGREGATE_WORKLOAD, defer_reaper=True
    )
    second = second_process = None
    try:
        _assert(
            first_process.stdout.readline().strip() == "allocated",
            "first memory task did not allocate",
        )
        second, second_process = _start(
            manager,
            "memory-aggregate-owner",
            MEMORY_AGGREGATE_WORKLOAD,
            defer_reaper=True,
        )
        # The second task is expected to be killed while allocating; do not
        # require an output record after the allocation itself.
        second_process.wait(timeout=6)
        owner = first.path.parent
        parent = {
            "memory_current": int((owner / "memory.current").read_text()),
            "memory_max": (owner / "memory.max").read_text().strip(),
            "memory_events": _events(owner / "memory.events"),
        }
        first_measurement = _task_measurement(tree, first.path)
        second_measurement = _task_measurement(tree, second.path)
        _assert(
            parent["memory_events"].get("oom_kill", 0) > 0,
            "parent memory.events did not record aggregate OOM",
        )
        _assert(
            parent["memory_current"] <= manager.policy.user.memory_bytes,
            "parent exceeded memory.max",
        )
        return {
            "second_returncode": second_process.returncode,
            "parent": parent,
            "first_task": first_measurement,
            "second_task": second_measurement,
        }
    finally:
        for launch in (first, second):
            if launch is None or launch.snapshot()["state"] == "finished":
                continue
            launch.abort()


def _oom(manager: Manager, tree: CgroupTree) -> dict[str, Any]:
    launch, process = _start(manager, "oom-owner", OOM_WORKLOAD, defer_reaper=True)
    try:
        _assert(
            process.stdout.readline().strip() == "allocating",
            "OOM workload did not start",
        )
        process.wait(timeout=6)
        measured = _task_measurement(tree, launch.path)
        owner = launch.path.parent
        owner_memory = {
            "memory_max": (owner / "memory.max").read_text().strip(),
            "memory_swap_max": (owner / "memory.swap.max").read_text().strip(),
            "memory_swap_current": int((owner / "memory.swap.current").read_text()),
            "memory_events": _events(owner / "memory.events"),
        }
        memory_events = measured["usage"]["memory_events"]
        _assert(
            memory_events.get("oom_kill", 0) > 0,
            "kernel did not record task OOM kill",
        )
        _assert(
            owner_memory["memory_events"].get("oom_kill", 0) > 0,
            "user memory.events did not record OOM kill",
        )
        _assert(
            owner_memory["memory_swap_max"] == "0"
            and owner_memory["memory_swap_current"] == 0,
            "zero swap was not effective at the user budget",
        )
        _assert(process.returncode != 0, "over-limit allocation unexpectedly succeeded")
        _assert(launch._advance(), "deferred OOM task did not clean up")
        _assert(launch.end_reason == "oom", f"OOM end reason was {launch.end_reason}")
        return {
            "returncode": process.returncode,
            "user_memory": owner_memory,
            "measurement_before_cleanup": measured,
            "end_reason": launch.end_reason,
        }
    finally:
        if launch.snapshot()["state"] != "finished":
            launch.abort()


def _deadline_descendant(manager: Manager, tree: RecordingCgroupTree) -> dict[str, Any]:
    launch, process = _start(manager, "deadline-owner", DEADLINE_WORKLOAD)
    result = _read_json_line(process)
    path = launch.path
    snapshot = _wait_finished(launch, 5)
    _assert(snapshot["end_reason"] == "timed_out", "deadline did not retain timed_out")
    _assert(
        _pid_gone(result["setsid_descendant"]),
        "setsid descendant survived cgroup.kill",
    )
    observed = tree.empty_readbacks.get(str(path), [])
    _assert(True in observed, "cleanup never observed populated=0 before removal")
    _assert(not path.exists(), "deadline task cgroup was not removed")
    return {
        "setsid_descendant": result["setsid_descendant"],
        "returncode": process.returncode,
        "snapshot": snapshot,
        "populated_zero_observed": True in observed,
    }


def _orphan_cleanup(manager: Manager, tree: RecordingCgroupTree) -> dict[str, Any]:
    launch, process = _start(manager, "orphan-owner", ORPHAN_WORKLOAD)
    result = _read_json_line(process)
    path = launch.path
    snapshot = _wait_finished(launch, 4)
    _assert(
        snapshot["end_reason"] == "completed",
        "natural parent exit was not completed",
    )
    _assert(
        _pid_gone(result["orphan"]),
        "orphan descendant survived parent-exit cleanup",
    )
    observed = tree.empty_readbacks.get(str(path), [])
    _assert(True in observed and not path.exists(), "orphan cgroup was not drained")
    return {
        "orphan": result["orphan"],
        "snapshot": snapshot,
        "populated_zero_observed": True in observed,
    }


def _cancellation_reuses_quota(manager: Manager) -> dict[str, Any]:
    sleeper = 'import time; print("ready", flush=True); time.sleep(5)'
    launch, process = _start(manager, "reuse-owner", sleeper)
    _assert(
        process.stdout.readline().strip() == "ready",
        "cancellation workload did not start",
    )
    path = launch.path
    launch.stop(reason="cancelled")
    launch.stop(reason="cancelled")
    _assert(not manager._tasks, "repeated cancellation retained a task reservation")
    _assert(not path.exists(), "cancelled task cgroup was not removed")
    probe, probe_process = _start(manager, "reuse-owner", 'print("reused", flush=True)')
    try:
        _assert(
            probe_process.stdout.readline().strip() == "reused",
            "quota reuse probe failed",
        )
        snapshot = _wait_finished(probe, 3)
        _assert(snapshot["end_reason"] == "completed", "reuse probe did not complete")
        return {
            "first_task_removed": not path.exists(),
            "second_task_end_reason": snapshot["end_reason"],
            "manager_task_count": len(manager._tasks),
        }
    finally:
        if probe.snapshot()["state"] != "finished":
            probe.abort()


def main() -> None:
    policy = _policy()
    evidence: dict[str, Any] = {
        "root": str(ROOT),
        "policy": {
            name: _limits(getattr(policy, name))
            for name in ("service", "helpers", "compute", "user")
        },
        "checks": {},
    }
    tree = RecordingCgroupTree(policy)
    manager = Manager(policy, tree)
    try:
        evidence["prepared_root"] = _prepared_root()
        tree.initialize()
        evidence["checks"]["cpu_user_aggregate"] = _cpu_aggregate(manager, tree)
        evidence["checks"]["pids_user_aggregate"] = _user_parent_pids(manager, tree)
        evidence["checks"]["memory_user_aggregate"] = _user_parent_memory(manager, tree)
        evidence["checks"]["user_oom"] = _oom(manager, tree)
        deadline_manager = Manager(
            replace(policy, max_runtime=1.2, terminate_grace=0.3), tree
        )
        evidence["checks"]["deadline_setsid_descendant"] = _deadline_descendant(
            deadline_manager, tree
        )
        evidence["checks"]["natural_parent_exit_orphan"] = _orphan_cleanup(
            manager, tree
        )
        evidence["checks"]["cancellation_quota_reuse"] = _cancellation_reuses_quota(
            manager
        )
        evidence["result"] = "passed"
    except BaseException as error:  # preserve real kernel observations on failure
        evidence["result"] = "failed"
        evidence["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        raise
    finally:
        for launch in list(manager._tasks.values()):
            if launch is not None and launch.snapshot()["state"] != "finished":
                try:
                    launch.abort()
                except Exception as cleanup_error:  # noqa: BLE001 — retain original failure and cleanup evidence
                    evidence.setdefault("cleanup_errors", []).append(str(cleanup_error))
        evidence["empty_readbacks"] = tree.empty_readbacks
        EVIDENCE_PATH.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        print(
            "KERNEL_ACCEPTANCE="
            + json.dumps(
                {"evidence": evidence, "evidence_path": str(EVIDENCE_PATH)},
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
