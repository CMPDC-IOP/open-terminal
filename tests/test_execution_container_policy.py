import json
import os
from types import SimpleNamespace

import pytest

from open_terminal.execution.docker_bootstrap import load_container_policy


@pytest.fixture
def source(tmp_path):
    p = tmp_path / 'policy.json'
    data = {'cgroup_root': '/sys/fs/cgroup/fixture', 'max_tasks': 4, 'max_user_tasks': 2, 'max_runtime': 90}
    for name in ('service', 'helpers', 'compute', 'user'):
        data[name] = {'cpu_millis': 500, 'memory_bytes': 1048576, 'pids': 16}
    p.write_text(json.dumps(data))
    return p


def test_accepts_readonly_deployer_owned_policy(source, monkeypatch):
    monkeypatch.setattr(os, 'statvfs', lambda _: SimpleNamespace(f_flag=os.ST_RDONLY))
    result = load_container_policy(source)
    assert result.user.cpu_millis == 500
    assert result.max_runtime == 90


def test_rejects_writable_mount(source, monkeypatch):
    monkeypatch.setattr(os, 'statvfs', lambda _: SimpleNamespace(f_flag=0))
    with pytest.raises(ValueError, match='read-only'):
        load_container_policy(source)


@pytest.mark.parametrize(
    'root', ['/sys/fs/cgroup', '/sys/fs/cgroup/bootstrap', '/tmp/other', '/sys/fs/cgroup/nested/child']
)
def test_rejects_unsafe_cgroup_root(source, monkeypatch, root):
    monkeypatch.setattr(os, 'statvfs', lambda _: SimpleNamespace(f_flag=os.ST_RDONLY))
    data = json.loads(source.read_text())
    data['cgroup_root'] = root
    source.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='dedicated'):
        load_container_policy(source)


def test_rejects_symlink(source):
    link = source.with_name('link.json')
    link.symlink_to(source)
    with pytest.raises(ValueError, match='regular file'):
        load_container_policy(link)


@pytest.mark.parametrize(
    'setting,configured,allowed',
    [('1', False, True), ('yes', False, True), ('TRUE', False, True),
     ('0', True, False), ('false', True, False), ('no', True, False),
     ('', True, False), (None, True, True), (None, False, False)],
)
@pytest.mark.parametrize('explicit', [False, True])
def test_bootstrap_uses_service_multi_user_configuration(
    tmp_path, monkeypatch, setting, configured, allowed, explicit
):
    from open_terminal import config
    from open_terminal.execution import docker_bootstrap as bootstrap

    policy = tmp_path / 'settings.toml'
    policy.write_text(f'multi_user = {str(configured).lower()}\n')
    monkeypatch.setattr(config, '_config', {})
    monkeypatch.setattr(config, '_SYSTEM_CONFIG_PATH', tmp_path / 'missing.toml')
    monkeypatch.setattr(
        config, '_default_user_config_path',
        lambda: tmp_path / 'missing.toml' if explicit else policy,
    )
    monkeypatch.setattr(
        bootstrap.sys, 'argv',
        ['bootstrap', 'run', *([f'--config={policy}'] if explicit else [])],
    )
    monkeypatch.delenv('OPEN_TERMINAL_MULTI_USER', raising=False)
    if setting is not None:
        monkeypatch.setenv('OPEN_TERMINAL_MULTI_USER', setting)
    monkeypatch.setenv('OPEN_TERMINAL_EXECUTION_POLICY_FILE', '/etc/policy.json')
    monkeypatch.setattr(bootstrap, 'load_container_policy', lambda _: None)
    monkeypatch.setattr(bootstrap.os, 'geteuid', lambda: 0)
    original_exists = bootstrap.Path.exists
    original_read = bootstrap.Path.read_text
    monkeypatch.setattr(
        bootstrap.Path, 'exists',
        lambda p: True if str(p) == '/.dockerenv' else original_exists(p),
    )
    monkeypatch.setattr(
        bootstrap.Path, 'read_text',
        lambda p, *a, **kw: '0::/\n' if str(p) == '/proc/self/cgroup'
        else original_read(p, *a, **kw),
    )

    class Validated(Exception):
        pass

    def stop_before_mount(*args, **kwargs):
        raise Validated

    monkeypatch.setattr(bootstrap.subprocess, 'run', stop_before_mount)
    if allowed:
        with pytest.raises(Validated):
            bootstrap.main()
    else:
        with pytest.raises(RuntimeError, match='multi-user mode'):
            bootstrap.main()


def test_bootstrap_skips_process_that_exits_during_migration(monkeypatch):
    from unittest.mock import MagicMock, call

    from open_terminal import config, env
    from open_terminal.execution import docker_bootstrap as bootstrap

    monkeypatch.setattr(config, 'init', lambda _: None)
    monkeypatch.setattr(env, '_resolve_multi_user', lambda: True)
    monkeypatch.setattr(bootstrap.sys, 'argv', ['bootstrap'])
    monkeypatch.setenv('OPEN_TERMINAL_EXECUTION_POLICY_FILE', '/etc/policy.json')
    monkeypatch.setattr(bootstrap, 'load_container_policy', lambda _: None)
    monkeypatch.setattr(bootstrap.os, 'geteuid', lambda: 0)
    monkeypatch.setattr(bootstrap.subprocess, 'run', lambda *a, **kw: None)
    root, group, members, controls = (MagicMock() for _ in range(4))
    members.read_text.return_value = '42\n1\n'
    group.__truediv__.return_value.write_text.side_effect = [ProcessLookupError(), None]

    class Migrated(Exception):
        pass

    controls.write_text.side_effect = Migrated
    root.__truediv__.side_effect = {
        'bootstrap': group, 'cgroup.procs': members, 'cgroup.subtree_control': controls,
    }.__getitem__
    other = MagicMock()
    other.read_text.return_value = '0::/\n'
    monkeypatch.setattr(
        bootstrap, 'Path', lambda path: root if path == '/sys/fs/cgroup' else other,
    )
    with pytest.raises(Migrated):
        bootstrap.main()
    assert group.__truediv__.return_value.write_text.call_args_list == [call('42'), call('1')]
