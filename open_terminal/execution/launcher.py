"""Trusted exec trampoline; invoke with Python isolated mode and a minimal env.

The parent supplies open directory/data descriptors, never user-selected cgroup
paths. A CLOEXEC status pipe reports readiness and startup failures; exec closes
it on success. EOF without readiness means the trusted bootstrap failed.
"""

import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path


def _prctl(libc, option: int, arg: int = 0) -> None:
    if libc.prctl(option, arg, 0, 0, 0) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))


def _drop_privileges(uid: int, gid: int) -> None:
    import resource

    if uid <= 0 or gid <= 0:
        raise ValueError("computation requires a non-root UID and GID")
    libc = ctypes.CDLL(None, use_errno=True)
    # Normalize scheduling while privileged; no user process may request RT CPU.
    os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))
    resource.setrlimit(resource.RLIMIT_RTPRIO, (0, 0))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    _prctl(libc, 38, 1)  # PR_SET_NO_NEW_PRIVS
    _prctl(libc, 47, 4)  # PR_CAP_AMBIENT / PR_CAP_AMBIENT_CLEAR_ALL
    last_cap = int(Path("/proc/sys/kernel/cap_last_cap").read_text())
    for capability in range(last_cap + 1):
        _prctl(libc, 24, capability)  # PR_CAPBSET_DROP
    _prctl(libc, 8, 0)  # PR_SET_KEEPCAPS
    os.setgroups([])
    os.setresgid(gid, gid, gid)
    os.setresuid(uid, uid, uid)

    class CapHeader(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]

    class CapData(ctypes.Structure):
        _fields_ = [
            ("effective", ctypes.c_uint32),
            ("permitted", ctypes.c_uint32),
            ("inheritable", ctypes.c_uint32),
        ]

    header = CapHeader(0x20080522, 0)
    data = (CapData * 2)()
    if libc.capset(ctypes.byref(header), ctypes.byref(data)) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))
    status = dict(
        line.split(":", 1)
        for line in Path("/proc/self/status").read_text().splitlines()
        if ":" in line
    )
    for key in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb"):
        if int(status[key].strip(), 16) != 0:
            raise RuntimeError(f"privilege cleanup failed: {key}")
    if status["NoNewPrivs"].strip() != "1":
        raise RuntimeError("no_new_privs verification failed")
    if (
        os.getresuid() != (uid, uid, uid)
        or os.getresgid() != (gid, gid, gid)
        or os.getgroups()
    ):
        raise RuntimeError("execution identity verification failed")


def _read_environment(fd: int) -> dict[str, str]:
    chunks = []
    length = 0
    while block := os.read(fd, 65536):
        length += len(block)
        if length > 1024 * 1024:
            raise ValueError("execution environment is too large")
        chunks.append(block)
    value = json.loads(b"".join(chunks))
    if not isinstance(value, dict) or any(
        not isinstance(k, str)
        or not k
        or "=" in k
        or "\0" in k
        or not isinstance(v, str)
        or "\0" in v
        for k, v in value.items()
    ):
        raise ValueError("environment must be a JSON object of valid string pairs")
    return value


def _close_descriptors(status_fd: int) -> None:
    for entry in os.listdir("/proc/self/fd"):
        fd = int(entry)
        if fd > 2 and fd != status_fd:
            try:
                os.close(fd)
            except OSError:
                pass  # the directory iterator's own descriptor has closed


def launch(args: argparse.Namespace) -> None:
    os.set_inheritable(args.status_fd, False)
    if sys.platform != "linux" or os.geteuid() != 0:
        raise RuntimeError("the trusted launcher requires Linux and root")
    descriptors = {args.status_fd, args.cgroup_fd, args.environment_fd}
    if len(descriptors) != 3 or min(descriptors) < 3:
        raise ValueError("launcher descriptors must be distinct and above stdio")
    if not args.command:
        raise ValueError("missing execution command")
    procs = os.open(
        "cgroup.procs",
        os.O_WRONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=args.cgroup_fd,
    )
    try:
        if os.write(procs, b"0") != 1:
            raise RuntimeError("failed to enter execution cgroup")
    finally:
        os.close(procs)
    # Read data only after resource accounting begins. Never evaluate environment
    # values in the trusted interpreter or merge the service's environment.
    environment = _read_environment(args.environment_fd)
    if not args.helper or args.uid != 0:
        _drop_privileges(args.uid, args.gid)
    elif args.gid != 0:
        raise ValueError("root helpers require GID 0")
    os.chdir(args.cwd)
    _close_descriptors(args.status_fd)
    ready = f"READY {time.monotonic_ns()} {time.time_ns()}\n".encode()
    if os.write(args.status_fd, ready) != len(ready):
        raise RuntimeError("failed to report launcher readiness")
    os.execvpe(args.command[0], args.command, environment)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cgroup-fd", type=int, required=True)
    parser.add_argument("--status-fd", type=int, required=True)
    parser.add_argument("--environment-fd", type=int, required=True)
    parser.add_argument("--uid", type=int, required=True)
    parser.add_argument("--gid", type=int, required=True)
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--helper", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    try:
        launch(args)
    except BaseException as error:  # noqa: BLE001 - every failure must reach the status pipe
        payload = json.dumps(
            {"error": str(error), "type": type(error).__name__}
        ).encode()[:4096]
        try:
            os.write(args.status_fd, payload)
        finally:
            os._exit(127)


if __name__ == "__main__":
    main()
