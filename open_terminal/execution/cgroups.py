"""Fail-closed ownership of an administrator-provided cgroup v2 subtree."""

import os
import re
import signal
import stat
import sys
from pathlib import Path

from .policy import ExecutionPolicy, Limits


class CgroupError(RuntimeError):
    pass


_CONTROLLERS = {"cpu", "memory", "pids"}
_HEX = re.compile(r"[0-9a-f]{1,128}\Z")


def _read(path: Path) -> str:
    return path.read_text().strip()


def _write(path: Path, value: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        raw = value.encode()
        if os.write(fd, raw) != len(raw):
            raise CgroupError(f"short write to {path}")
    finally:
        os.close(fd)


def _trusted_path(path: Path) -> None:
    for item in (path, *path.parents):
        info = item.lstat()
        if stat.S_ISLNK(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
            raise CgroupError(
                f"cgroup path must be root-owned and not user-writable: {item}"
            )
        # Delegation can change control-file ownership without making the
        # directory writable. In particular, writable ancestor cgroup.procs
        # would let computation migrate out of its task budget.
        if (item / "cgroup.procs").exists():
            for control in item.iterdir():
                if control.is_dir():
                    continue
                info = control.lstat()
                if (
                    stat.S_ISLNK(info.st_mode)
                    or info.st_uid != 0
                    or info.st_mode & 0o022
                ):
                    raise CgroupError(
                        f"cgroup control file must be root-owned and not user-writable: {control}"
                    )


def _mount_point(root: Path) -> Path:
    candidates = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        before, after = line.split(" - ", 1)
        if after.split()[0] != "cgroup2":
            continue
        encoded = before.split()[4]
        decoded = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), encoded)
        mount = Path(decoded)
        if root.is_relative_to(mount):
            candidates.append(mount)
    if not candidates:
        raise CgroupError("cgroup_root is not on a cgroup v2 mount")
    mount = max(candidates, key=lambda p: len(p.parts))
    if root == mount:
        raise CgroupError(
            "cgroup_root must be a dedicated subtree, not the cgroup mount root"
        )
    return mount


def _events(path: Path) -> dict[str, int]:
    return {
        key: int(value)
        for key, value in (line.split() for line in _read(path).splitlines())
    }


class CgroupTree:
    def __init__(self, policy: ExecutionPolicy):
        self.policy = policy
        self.root = policy.cgroup_root
        self.service_path = self.root / "service"
        self.helpers_path = self.root / "helpers"
        self.compute_path = self.root / "compute"
        self._tasks: set[Path] = set()
        self._initialized = False

    def _check_budgets(self, mount: Path) -> None:
        total = self.policy.total
        current = self.root
        while True:
            for filename, required in (
                ("memory.max", total.memory_bytes),
                ("pids.max", total.pids),
                ("memory.swap.max", 0),
            ):
                path = current / filename
                if not path.exists():
                    if current == self.root:
                        raise CgroupError(f"missing required cgroup interface: {path}")
                    continue  # the real hierarchy root has no resource limit files
                value = _read(path)
                if value == "max":
                    if current == self.root:
                        raise CgroupError(
                            f"the deployment must set a finite {filename} budget"
                        )
                elif int(value) < required:
                    raise CgroupError(f"insufficient ancestor budget: {path}")
            cpu = current / "cpu.max"
            if cpu.exists():
                quota, period = _read(cpu).split()
                if quota == "max":
                    if current == self.root:
                        raise CgroupError(
                            "the deployment must set a finite cpu.max budget"
                        )
                elif int(quota) * 1000 < total.cpu_millis * int(period):
                    raise CgroupError(f"insufficient ancestor CPU budget: {cpu}")
            elif current == self.root:
                raise CgroupError("missing cpu.max")
            oom = current / "memory.oom.group"
            if oom.exists() and _read(oom) != "0":
                raise CgroupError(
                    f"ancestor OOM grouping would include the service: {oom}"
                )
            if current == mount:
                break
            current = current.parent
        if total.cpu_millis > len(os.sched_getaffinity(0)) * 1000:
            raise CgroupError("CPU budgets exceed the API process's available CPU set")

    def _enable(self, path: Path) -> None:
        available = set(_read(path / "cgroup.controllers").split())
        if not _CONTROLLERS <= available:
            raise CgroupError(
                f"cpu, memory and pids controllers are required at {path}"
            )
        _write(path / "cgroup.subtree_control", "+cpu +memory +pids")
        if not _CONTROLLERS <= set(_read(path / "cgroup.subtree_control").split()):
            raise CgroupError(f"controller enablement readback failed at {path}")

    def _limits(self, path: Path, limits: Limits) -> None:
        _trusted_path(path)
        for filename in ("cgroup.kill", "cgroup.events", "memory.swap.max"):
            if not (path / filename).exists():
                raise CgroupError(
                    f"required cgroup interface missing: {path / filename}"
                )
        values = {
            "cpu.max": f"{limits.cpu_millis * 100} 100000",
            "memory.max": str(limits.memory_bytes),
            "pids.max": str(limits.pids),
            "memory.swap.max": "0",
            "memory.oom.group": "0",
        }
        for name, value in values.items():
            _write(path / name, value)
            actual = _read(path / name)
            # The kernel rounds memory limits to a page boundary. Requiring
            # exact values keeps configured hard budgets unambiguous.
            if actual != value:
                raise CgroupError(
                    f"resource limit readback mismatch at {path / name}: {actual}"
                )

    def initialize(self) -> None:
        if self._initialized:
            raise CgroupError("cgroup tree has already been initialized")
        if sys.platform != "linux" or os.geteuid() != 0:
            raise CgroupError(
                "hard execution isolation requires Linux and a root API process"
            )
        if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
            raise CgroupError("managed termination requires pidfd support")
        _trusted_path(self.root)
        mount = _mount_point(self.root)
        self._check_budgets(mount)
        if _read(self.root / "cgroup.type") != "domain":
            raise CgroupError("cgroup_root must be a domain cgroup")
        root_pids = set(_read(self.root / "cgroup.procs").split())
        if root_pids - {str(os.getpid())}:
            raise CgroupError(
                "cgroup_root contains unrelated processes; deployment must move them first"
            )
        for path in self.root.iterdir():
            if path.is_dir() and path.name not in {"service", "helpers", "compute"}:
                raise CgroupError(f"unrecognized cgroup in dedicated subtree: {path}")
        for path in (self.compute_path, self.helpers_path):
            if path.exists():
                _trusted_path(path)
                if not self.is_empty(path):
                    raise CgroupError(
                        f"pre-existing populated cgroup requires operator cleanup: {path}"
                    )
        if self.service_path.exists():
            _trusted_path(self.service_path)
            if set(_read(self.service_path / "cgroup.procs").split()) - {
                str(os.getpid())
            }:
                raise CgroupError("service cgroup contains another process")
        self.service_path.mkdir(exist_ok=True)
        _write(self.service_path / "cgroup.procs", str(os.getpid()))
        if str(os.getpid()) not in _read(self.service_path / "cgroup.procs").split():
            raise CgroupError("API cgroup membership readback failed")
        self._enable(self.root)
        for path, limits in (
            (self.service_path, self.policy.service),
            (self.helpers_path, self.policy.helpers),
            (self.compute_path, self.policy.compute),
        ):
            path.mkdir(exist_ok=True)
            self._limits(path, limits)
        self._enable(self.compute_path)
        # Probe the interfaces on an empty throwaway task without running user code.
        self._initialized = True
        try:
            probe = self.create_task("0" * 64, os.urandom(16).hex())
            self.kill(probe)
            self.remove_task(probe)
        except BaseException:
            self._initialized = False
            raise

    def create_task(self, owner_key: str, task_id: str) -> Path:
        if not self._initialized:
            raise CgroupError("cgroup tree is not initialized")
        if not _HEX.fullmatch(owner_key) or not _HEX.fullmatch(task_id):
            raise CgroupError(
                "owner and task identifiers must be trusted lowercase hexadecimal IDs"
            )
        owner = self.compute_path / f"owner-{owner_key}"
        owner.mkdir(exist_ok=True)
        self._limits(owner, self.policy.user)
        self._enable(owner)
        task = owner / f"task-{task_id}"
        task.mkdir()
        try:
            # Tasks share the owner's limits. Keep a leaf only to terminate a
            # task's descendants together, including when user memory hits OOM.
            _trusted_path(task)
            _write(task / "memory.oom.group", "1")
            if _read(task / "memory.oom.group") != "1":
                raise CgroupError("task OOM grouping readback failed")
        except BaseException:
            task.rmdir()
            raise
        self._tasks.add(task)
        return task

    def terminate(self, path: Path) -> None:
        """Signal current members using pidfds so PID reuse cannot hit other users.

        New forks during this best-effort graceful phase remain bounded by the
        user's resource limits and are removed by cgroup.kill at the grace deadline.
        """
        if path not in self._tasks:
            raise CgroupError("refusing to signal an unowned cgroup")
        for raw_pid in _read(path / "cgroup.procs").split():
            pid = int(raw_pid)
            try:
                descriptor = os.pidfd_open(pid)
            except ProcessLookupError:
                continue
            try:
                if raw_pid in _read(path / "cgroup.procs").split():
                    try:
                        signal.pidfd_send_signal(descriptor, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            finally:
                os.close(descriptor)

    def kill(self, path: Path) -> None:
        if path not in self._tasks:
            raise CgroupError("refusing to kill an unowned cgroup")
        _write(path / "cgroup.kill", "1")

    def is_empty(self, path: Path) -> bool:
        return _events(path / "cgroup.events")["populated"] == 0

    def remove_task(self, path: Path) -> None:
        if path not in self._tasks:
            raise CgroupError("refusing to remove an unowned cgroup")
        if not self.is_empty(path):
            raise CgroupError(f"cannot release a populated task: {path}")
        path.rmdir()
        self._tasks.remove(path)
        try:
            path.parent.rmdir()
        except OSError:
            pass  # Other tasks for this owner still hold its aggregate limits.

    def task_usage(self, path: Path) -> dict:
        if path not in self._tasks:
            raise CgroupError("unowned task cgroup")
        return {
            "memory_bytes": int(_read(path / "memory.current")),
            "pids": int(_read(path / "pids.current")),
            "memory_events": _events(path / "memory.events"),
            "pids_events": _events(path / "pids.events"),
            "cpu": _events(path / "cpu.stat"),
        }
