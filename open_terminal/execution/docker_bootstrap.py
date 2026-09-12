"""Initialize a dedicated subtree inside Docker's private cgroup namespace."""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from open_terminal import config

from .policy import load_policy


def load_container_policy(path: Path):
    # Bind-mounted source files may be owned by the host deployer. Require a
    # read-only mount, then copy the validated policy into root-owned runtime storage.
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError('Container policy must be an absolute regular file')
    if not os.statvfs(path).f_flag & os.ST_RDONLY:
        raise ValueError('Container policy must be mounted read-only')
    policy = load_policy(path)
    if policy.cgroup_root.parent != Path('/sys/fs/cgroup') or policy.cgroup_root.name == 'bootstrap':
        raise ValueError('Container policy requires a dedicated direct cgroup subtree')
    return policy


def main() -> None:
    # Match the service's system/user/explicit TOML lookup before resolving env.
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--config')
    options, _ = parser.parse_known_args(sys.argv[1:])
    config.init(options.config)
    from open_terminal.env import _resolve_multi_user

    policy = load_container_policy(Path(os.environ['OPEN_TERMINAL_EXECUTION_POLICY_FILE']))
    if (
        os.geteuid() != 0
        or not Path('/.dockerenv').exists()
        or Path('/proc/self/cgroup').read_text().strip() != '0::/'
        or not _resolve_multi_user()
    ):
        raise RuntimeError('Docker cgroup bootstrap requires root, multi-user mode and a private cgroup namespace')

    root = Path('/sys/fs/cgroup')
    subprocess.run(['mount', '-o', 'remount,rw', str(root)], check=True)
    bootstrap = root / 'bootstrap'
    bootstrap.mkdir(exist_ok=True)
    for pid in (root / 'cgroup.procs').read_text().split():
        try:
            (bootstrap / 'cgroup.procs').write_text(pid)
        except ProcessLookupError:
            pass  # A short-lived probe may exit between listing and migration.
    (root / 'cgroup.subtree_control').write_text('+cpu +memory +pids')
    execution = policy.cgroup_root
    execution.mkdir(exist_ok=True)
    for name in ('cpu.max', 'memory.max', 'memory.swap.max', 'pids.max'):
        value = (root / name).read_text()
        if 'max' in value:
            raise RuntimeError(f'Terminal requires a finite container {name}')
        (execution / name).write_text(value)
    (execution / 'memory.oom.group').write_text('0')

    runtime = Path('/run/open-terminal')
    runtime.mkdir(mode=0o700, exist_ok=True)
    # Reuse the manager's root ownership and symlink checks before writing.
    from .manager import _trusted_path

    _trusted_path(runtime)
    target = runtime / 'policy.json'
    data = {
        name: asdict(policy)[name]
        for name in (
            'cgroup_root',
            'service',
            'helpers',
            'compute',
            'user',
            'max_tasks',
            'max_user_tasks',
            'max_runtime',
        )
    }
    data['cgroup_root'] = str(policy.cgroup_root)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(data, stream)
    _trusted_path(target)
    os.environ['OPEN_TERMINAL_EXECUTION_POLICY_FILE'] = str(target)
    (execution / 'cgroup.procs').write_text(str(os.getpid()))
    os.execvp(
        'capsh', ['capsh', '--drop=cap_sys_admin', '--', '-c', 'exec /app/entrypoint.sh "$@"', '--', *sys.argv[1:]]
    )


if __name__ == '__main__':
    main()
