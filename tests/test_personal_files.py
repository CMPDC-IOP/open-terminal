import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import threading

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient


if sys.platform != 'linux':
    pytest.skip('Personal file management requires Linux', allow_module_level=True)

import pwd

from open_terminal import file_operations, workspace

FILE_OPERATIONS_PATH = Path(file_operations.__file__)


class FakeFilesystem:
    def __init__(self, home: Path, username: str):
        self.home = str(home)
        self.username = username


@pytest.fixture
def client(tmp_path, monkeypatch):
    homes = {}
    for user_id in ('alice-id', 'bob-id'):
        username = user_id.split('-')[0]
        home = tmp_path / 'home' / username
        (home / 'w').mkdir(parents=True)
        homes[user_id] = FakeFilesystem(home, username)

    app = FastAPI()

    def helper_command(filesystem, action, payload):
        return [
            sys.executable,
            '-I',
            str(FILE_OPERATIONS_PATH),
            action,
            '--home',
            filesystem.home,
            '--username',
            filesystem.username,
            '--payload',
            json.dumps(payload),
        ]

    monkeypatch.setattr(workspace, '_operation_command', helper_command)

    def verify_api_key(request: Request):
        if request.headers.get('Authorization') != 'Bearer test-key':
            raise HTTPException(status_code=401, detail='Invalid API key')

    def get_filesystem(request: Request):
        return homes[request.headers['X-User-Id']]

    workspace.install_workspace_routes(app, get_filesystem, verify_api_key)
    with TestClient(app) as test_client:
        yield test_client, homes


def request(client, user_id='alice-id', **kwargs):
    return client.get(
        '/workspace-files',
        headers={'Authorization': 'Bearer test-key', 'X-User-Id': user_id},
        **kwargs,
    )


def content(client, path, user_id='alice-id'):
    return client.get(
        '/workspace-files/content',
        params={'path': path},
        headers={'Authorization': 'Bearer test-key', 'X-User-Id': user_id},
    )


def home_request(client, user_id='alice-id', **kwargs):
    return client.get(
        '/home-files',
        headers={'Authorization': 'Bearer test-key', 'X-User-Id': user_id},
        **kwargs,
    )


def home_content(client, path, user_id='alice-id'):
    return client.get(
        '/home-files/content',
        params={'path': path},
        headers={'Authorization': 'Bearer test-key', 'X-User-Id': user_id},
    )


def home_mutation_headers(user_id='alice-id'):
    return {'Authorization': 'Bearer test-key', 'X-User-Id': user_id}


def home_text(client, path, user_id='alice-id'):
    return client.get('/home-files/text', params={'path': path}, headers=home_mutation_headers(user_id))


def test_requires_api_key_and_upstream_user_id(client):
    test_client, _ = client

    assert test_client.get('/workspace-files').status_code == 401
    assert test_client.get('/workspace-files', headers={'Authorization': 'Bearer test-key'}).status_code == 403
    assert test_client.get('/home-files').status_code == 401
    assert test_client.get('/home-files', headers={'Authorization': 'Bearer test-key'}).status_code == 403


def test_mutation_helper_is_a_fixed_process_for_the_provisioned_user():
    filesystem = FakeFilesystem(Path('/home/alice'), 'alice')

    command = workspace._operation_command(filesystem, 'mkdir', {'path': 'uploads'})

    assert command[:9] == [
        'sudo',
        '-n',
        '-u',
        'alice',
        '--',
        sys.executable,
        '-I',
        '-m',
        'open_terminal.file_operations',
    ]
    assert command[-1] == '{"path":"uploads"}'


def test_refuses_the_single_user_filesystem_without_a_username(client):
    test_client, homes = client
    homes['alice-id'] = type('SingleUserFilesystem', (), {'home': homes['alice-id'].home})()

    assert request(test_client).status_code == 403


@pytest.mark.parametrize('path', ['/etc', '..', 'a/../b', 'a//b', r'a\\b', 'a\x00b'])
def test_rejects_malformed_paths(client, path):
    test_client, _ = client

    assert request(test_client, params={'path': path}).status_code == 400
    assert home_request(test_client, params={'path': path}).status_code == 400


def test_preserves_percent_characters_after_the_framework_decodes_the_query_once(client):
    test_client, homes = client
    (Path(homes['alice-id'].home) / 'w' / '%2e%2e').write_bytes(b'percent is literal')

    response = test_client.get(
        '/workspace-files/content?path=%252e%252e',
        headers={'Authorization': 'Bearer test-key', 'X-User-Id': 'alice-id'},
    )

    assert response.status_code == 200
    assert response.content == b'percent is literal'


def test_missing_workspace_is_empty_at_root_and_never_created(client):
    test_client, homes = client
    missing_home = Path(homes['alice-id'].home).parent / 'missing'
    homes['alice-id'] = FakeFilesystem(missing_home, 'missing')

    response = request(test_client)

    assert response.status_code == 200
    assert response.json() == {'path': '', 'entries': []}
    assert not missing_home.exists()
    assert request(test_client, params={'path': 'nested'}).status_code == 404


def test_home_files_lists_siblings_and_workspace_files(client):
    test_client, homes = client
    home = Path(homes['alice-id'].home)
    (home / 'notes.txt').write_text('personal')
    nested = home / 'w' / 'project' / 'readme.md'
    nested.parent.mkdir()
    nested.write_text('workspace')

    root_response = home_request(test_client)
    nested_response = home_request(test_client, params={'path': 'w/project'})

    assert [entry['name'] for entry in root_response.json()['entries']] == ['notes.txt', 'w']
    assert nested_response.json()['entries'][0]['path'] == 'w/project/readme.md'
    assert home_content(test_client, 'notes.txt').content == b'personal'
    assert home_content(test_client, 'w/project/readme.md').content == b'workspace'


def test_home_files_work_without_a_workspace_directory(client):
    test_client, homes = client
    home = Path(homes['alice-id'].home)
    (home / 'w').rmdir()
    (home / 'notes.txt').write_text('personal')

    response = home_request(test_client)

    assert response.status_code == 200
    assert [entry['name'] for entry in response.json()['entries']] == ['notes.txt']
    assert home_content(test_client, 'notes.txt').content == b'personal'
    assert not (home / 'w').exists()
    assert request(test_client).json() == {'path': '', 'entries': []}


def test_lists_nested_unicode_files_and_streams_raw_bytes(client):
    test_client, homes = client
    workspace_root = Path(homes['alice-id'].home) / 'w'
    nested = workspace_root / '资料' / '2026'
    nested.mkdir(parents=True)
    file_path = nested / '报告.txt'
    payload = b'hello\x00world' * 20_000
    file_path.write_bytes(payload)

    root_response = request(test_client)
    nested_response = request(test_client, params={'path': '资料/2026'})
    content_response = content(test_client, '资料/2026/报告.txt')

    assert root_response.json()['entries'][0]['path'] == '资料'
    entry = nested_response.json()['entries'][0]
    assert entry['name'] == '报告.txt'
    assert entry['path'] == '资料/2026/报告.txt'
    assert entry['type'] == 'file'
    assert entry['size'] == len(payload)
    assert isinstance(entry['modified'], float)
    assert content_response.content == payload
    assert content_response.headers['content-type'].startswith('text/plain')
    assert "filename*=UTF-8''%E6%8A%A5%E5%91%8A.txt" in content_response.headers['content-disposition']


def test_hides_and_refuses_symlinks_and_special_files(client):
    test_client, homes = client
    workspace_root = Path(homes['alice-id'].home) / 'w'
    (workspace_root / 'plain.txt').write_text('safe')
    (workspace_root / 'inside').mkdir()
    os.symlink(workspace_root / 'plain.txt', workspace_root / 'linked-file')
    os.symlink(workspace_root / 'inside', workspace_root / 'linked-directory')
    os.mkfifo(workspace_root / 'pipe')

    response = request(test_client)

    assert [entry['name'] for entry in response.json()['entries']] == ['inside', 'plain.txt']
    assert content(test_client, 'linked-file').status_code == 404
    assert request(test_client, params={'path': 'linked-directory'}).status_code == 404
    assert content(test_client, 'pipe').status_code == 404


def test_refuses_a_symlinked_workspace_root(client):
    test_client, homes = client
    home = Path(homes['alice-id'].home)
    (home / 'w').rmdir()
    target = home.parent / 'outside'
    target.mkdir()
    os.symlink(target, home / 'w')

    assert request(test_client).status_code == 404
    assert content(test_client, 'anything').status_code == 404


def test_refuses_a_symlinked_home_root(client):
    test_client, homes = client
    home = Path(homes['alice-id'].home)
    real_home = home.parent / 'real-alice'
    home.rename(real_home)
    os.symlink(real_home, home)

    assert home_request(test_client).status_code == 404
    assert home_content(test_client, 'w/anything').status_code == 404


def test_authenticated_users_can_only_read_their_own_workspace(client):
    test_client, homes = client
    (Path(homes['alice-id'].home) / 'w' / 'secret.txt').write_text('alice')
    (Path(homes['bob-id'].home) / 'w' / 'secret.txt').write_text('bob')

    assert content(test_client, 'secret.txt', 'alice-id').content == b'alice'
    assert content(test_client, 'secret.txt', 'bob-id').content == b'bob'


def test_authenticated_users_can_only_read_their_own_home(client):
    test_client, homes = client
    (Path(homes['alice-id'].home) / 'secret.txt').write_text('alice')
    (Path(homes['bob-id'].home) / 'secret.txt').write_text('bob')

    assert home_content(test_client, 'secret.txt', 'alice-id').content == b'alice'
    assert home_content(test_client, 'secret.txt', 'bob-id').content == b'bob'


def test_downloads_active_content_as_an_inert_attachment(client):
    test_client, homes = client
    home = Path(homes['alice-id'].home)
    (home / 'w' / 'preview.html').write_text('<script>alert(1)</script>')
    (home / 'preview.html').write_text('<script>alert(1)</script>')

    response = content(test_client, 'preview.html')
    home_response = home_content(test_client, 'preview.html')

    assert response.headers['content-type'].startswith('application/octet-stream')
    assert response.headers['content-disposition'].startswith('attachment;')
    assert response.headers['x-content-type-options'] == 'nosniff'
    assert home_response.headers['content-type'].startswith('application/octet-stream')
    assert home_response.headers['content-disposition'].startswith('attachment;')
    assert home_response.headers['x-content-type-options'] == 'nosniff'


def test_home_files_hide_and_refuse_symlinks(client):
    test_client, homes = client
    home = Path(homes['alice-id'].home)
    (home / 'plain.txt').write_text('safe')
    (home / 'inside').mkdir()
    os.symlink(home / 'plain.txt', home / 'linked-file')
    os.symlink(home / 'inside', home / 'linked-directory')

    response = home_request(test_client)

    assert [entry['name'] for entry in response.json()['entries']] == ['inside', 'plain.txt', 'w']
    assert home_content(test_client, 'linked-file').status_code == 404
    assert home_request(test_client, params={'path': 'linked-directory'}).status_code == 404


def test_mkdir_and_move_mutate_only_the_authenticated_users_home(client):
    test_client, homes = client
    home = Path(homes['alice-id'].home)
    (home / 'source.txt').write_text('alice')

    created = test_client.post('/home-files/mkdir', json={'path': 'uploads'}, headers=home_mutation_headers())
    moved = test_client.post(
        '/home-files/move',
        json={'source': 'source.txt', 'destination': 'uploads/renamed.txt'},
        headers=home_mutation_headers(),
    )

    assert created.status_code == 200
    assert created.json() == {'path': 'uploads'}
    assert moved.status_code == 200
    assert moved.json() == {'path': 'uploads/renamed.txt'}
    assert (home / 'uploads' / 'renamed.txt').read_text() == 'alice'
    assert not (Path(homes['bob-id'].home) / 'uploads').exists()


def test_mutation_conflicts_and_keep_both_uploads(client):
    test_client, homes = client
    home = Path(homes['alice-id'].home)
    (home / 'taken').mkdir()
    (home / 'first.txt').write_text('first')
    (home / 'second.txt').write_text('second')

    mkdir = test_client.post('/home-files/mkdir', json={'path': 'taken'}, headers=home_mutation_headers())
    move = test_client.post(
        '/home-files/move',
        json={'source': 'first.txt', 'destination': 'second.txt'},
        headers=home_mutation_headers(),
    )
    conflict = test_client.post(
        '/home-files/upload',
        data={'path': 'first.txt'},
        files={'file': ('first.txt', b'replacement')},
        headers=home_mutation_headers(),
    )
    kept = test_client.post(
        '/home-files/upload',
        data={'path': 'first.txt', 'conflict': 'keep-both'},
        files={'file': ('first.txt', b'kept')},
        headers=home_mutation_headers(),
    )

    assert (mkdir.status_code, mkdir.json()['detail']) == (409, 'Folder already exists.')
    assert (move.status_code, move.json()['detail']) == (409, 'File already exists.')
    assert (conflict.status_code, conflict.json()['detail']) == (409, 'File already exists.')
    assert kept.status_code == 200
    assert kept.json() == {'path': 'first (1).txt'}
    assert (home / 'first.txt').read_text() == 'first'
    assert (home / 'first (1).txt').read_bytes() == b'kept'


def test_large_conflicting_upload_returns_409_when_the_helper_closes_its_pipe(client):
    test_client, homes = client
    home = Path(homes['alice-id'].home)
    (home / 'taken.bin').write_bytes(b'original')

    response = test_client.post(
        '/home-files/upload',
        data={'path': 'taken.bin'},
        files={'file': ('taken.bin', b'x' * (workspace.CHUNK_SIZE * 4))},
        headers=home_mutation_headers(),
    )

    assert (response.status_code, response.json()['detail']) == (409, 'File already exists.')
    assert (home / 'taken.bin').read_bytes() == b'original'


def test_upload_replaces_only_regular_files_after_staging(client):
    test_client, homes = client
    home = Path(homes['alice-id'].home)
    destination = home / 'existing.txt'
    destination.write_bytes(b'original')

    replaced = test_client.post(
        '/home-files/upload',
        data={'path': 'existing.txt', 'conflict': 'replace'},
        files={'file': ('existing.txt', b'replaced')},
        headers=home_mutation_headers(),
    )

    assert replaced.status_code == 200
    assert replaced.json() == {'path': 'existing.txt'}
    assert destination.read_bytes() == b'replaced'


@pytest.mark.parametrize('mode', (0o600, 0o750))
def test_upload_replacement_preserves_existing_mode(client, mode):
    test_client, homes = client
    home = Path(homes['alice-id'].home)
    destination = home / 'existing.txt'
    destination.write_bytes(b'original')
    destination.chmod(mode)

    replaced = test_client.post(
        '/home-files/upload',
        data={'path': 'existing.txt', 'conflict': 'replace'},
        files={'file': ('existing.txt', b'replaced')},
        headers=home_mutation_headers(),
    )

    assert replaced.status_code == 200
    assert destination.read_bytes() == b'replaced'
    assert stat.S_IMODE(destination.stat().st_mode) == mode


def test_helper_never_partially_replaces_an_interrupted_upload(tmp_path):
    home = tmp_path / 'home' / 'alice'
    home.mkdir(parents=True)
    destination = home / 'existing.txt'
    destination.write_bytes(b'original')

    with pytest.raises(file_operations.FileOperationError, match='interrupted'):
        file_operations.upload(
            home=str(home),
            username='alice',
            path='existing.txt',
            conflict='replace',
            stream=io.BytesIO(b'partial'),
            expected_size=8,
        )

    assert destination.read_bytes() == b'original'
    assert not list(home.glob('.open-terminal-upload-*'))


def test_helper_creates_mutations_as_its_effective_os_user(tmp_path):
    username = pwd.getpwuid(os.geteuid()).pw_name
    home = tmp_path / 'home' / username
    home.mkdir(parents=True)

    file_operations.mkdir(home=str(home), username=username, path='uploads')
    file_operations.upload(
        home=str(home),
        username=username,
        path='uploads/file.txt',
        conflict='error',
        stream=io.BytesIO(b'owned'),
        expected_size=5,
    )

    assert (home / 'uploads').stat().st_uid == os.geteuid()
    assert (home / 'uploads' / 'file.txt').stat().st_uid == os.geteuid()
    assert (home / 'uploads' / 'file.txt').stat().st_gid == os.getegid()
    assert stat.S_IMODE((home / 'uploads' / 'file.txt').stat().st_mode) == 0o660


@pytest.mark.parametrize('path', ['', '/tmp/file', '../file', 'folder/../file', 'folder//file'])
def test_mutations_reject_invalid_paths(client, path):
    test_client, _ = client

    response = test_client.post('/home-files/mkdir', json={'path': path}, headers=home_mutation_headers())

    assert (response.status_code, response.json()['detail']) == (400, 'Invalid file path.')


def test_mutations_refuse_symlink_parents_and_final_entries_without_touching_outside(client):
    test_client, homes = client
    home = Path(homes['alice-id'].home)
    outside = home.parent / 'outside'
    outside.mkdir()
    (outside / 'untouched.txt').write_text('outside')
    os.symlink(outside, home / 'linked-parent')
    os.symlink(outside / 'untouched.txt', home / 'linked-file')
    (home / 'folder').mkdir()

    upload_parent = test_client.post(
        '/home-files/upload',
        data={'path': 'linked-parent/new.txt'},
        files={'file': ('new.txt', b'blocked')},
        headers=home_mutation_headers(),
    )
    replace_link = test_client.post(
        '/home-files/upload',
        data={'path': 'linked-file', 'conflict': 'replace'},
        files={'file': ('linked-file', b'blocked')},
        headers=home_mutation_headers(),
    )
    move_link = test_client.post(
        '/home-files/move',
        json={'source': 'linked-file', 'destination': 'folder/moved.txt'},
        headers=home_mutation_headers(),
    )

    assert upload_parent.status_code == 403
    assert replace_link.status_code == 403
    assert move_link.status_code == 403
    assert (outside / 'untouched.txt').read_text() == 'outside'
    assert not (outside / 'new.txt').exists()


def test_move_refuses_a_directory_into_itself_or_a_descendant(client):
    test_client, homes = client
    home = Path(homes['alice-id'].home)
    (home / 'folder' / 'child').mkdir(parents=True)

    response = test_client.post(
        '/home-files/move',
        json={'source': 'folder', 'destination': 'folder/child/renamed'},
        headers=home_mutation_headers(),
    )

    assert (response.status_code, response.json()['detail']) == (400, 'Cannot move a folder into itself.')
    assert (home / 'folder' / 'child').is_dir()


def test_text_editing_preserves_bom_and_rejects_binary_or_oversized_files(client):
    test_client, homes = client
    home = Path(homes['alice-id'].home)
    (home / 'bom.txt').write_bytes(b'\xef\xbb\xbfhello')
    (home / 'binary.bin').write_bytes(b'hello\x00world')
    (home / 'large.txt').write_bytes(b'x' * (workspace.MAX_TEXT_SIZE + 1))

    text = home_text(test_client, 'bom.txt')
    saved = test_client.post(
        '/home-files/save',
        json={'path': 'bom.txt', 'content': text.json()['content'] + '!', 'version': text.json()['version']},
        headers=home_mutation_headers(),
    )

    assert text.json()['content'] == '\ufeffhello'
    assert saved.status_code == 200
    assert (home / 'bom.txt').read_bytes() == b'\xef\xbb\xbfhello!'
    assert home_text(test_client, 'binary.bin').status_code == 415
    assert home_text(test_client, 'large.txt').status_code == 413
    assert (
        test_client.post(
            '/home-files/save',
            json={
                'path': 'bom.txt',
                'content': 'x' * (workspace.MAX_TEXT_SIZE + 1),
                'version': saved.json()['version'],
            },
            headers=home_mutation_headers(),
        ).status_code
        == 413
    )


def test_save_rejects_stale_versions_and_concurrent_api_writers(tmp_path):
    username = pwd.getpwuid(os.geteuid()).pw_name
    home = tmp_path / 'home' / username
    home.mkdir(parents=True)
    target = home / 'note.txt'
    target.write_text('before')
    version = hashlib.sha256(b'before').hexdigest()

    file_operations.save(
        home=str(home), username=username, path='note.txt', version=version, stream=io.BytesIO(b'first')
    )
    with pytest.raises(file_operations.FileOperationError, match='stale'):
        file_operations.save(
            home=str(home), username=username, path='note.txt', version=version, stream=io.BytesIO(b'stale')
        )

    target.write_text('before')
    outcomes = []
    barrier = threading.Barrier(2)

    def writer(content):
        barrier.wait()
        try:
            outcomes.append(
                file_operations.save(
                    home=str(home), username=username, path='note.txt', version=version, stream=io.BytesIO(content)
                )
            )
        except file_operations.FileOperationError as error:
            outcomes.append(error.code)

    first = threading.Thread(target=writer, args=(b'one',))
    second = threading.Thread(target=writer, args=(b'two',))
    first.start()
    second.start()
    first.join()
    second.join()

    assert len([outcome for outcome in outcomes if isinstance(outcome, dict)]) == 1
    assert outcomes.count('stale') == 1
    assert target.read_bytes() in {b'one', b'two'}


def test_concurrent_trash_root_initialization_waits_for_marker(tmp_path, monkeypatch):
    username = pwd.getpwuid(os.geteuid()).pw_name
    home = tmp_path / 'home' / username
    home.mkdir(parents=True)
    marker_write_started = threading.Event()
    allow_marker_write = threading.Event()
    first_complete = threading.Event()
    second_entered = threading.Event()
    second_complete = threading.Event()
    first_outcomes = []
    second_outcomes = []
    original_write_private_file = file_operations._write_private_file

    def pause_marker_write(*args):
        marker_write_started.set()
        assert allow_marker_write.wait(timeout=5)
        original_write_private_file(*args)

    def open_root(outcomes, complete, entered=None):
        if entered is not None:
            entered.set()
        try:
            outcomes.append(file_operations._open_trash_root(str(home), username, create=True))
        except BaseException as error:
            outcomes.append(error)
        finally:
            complete.set()

    monkeypatch.setattr(file_operations, '_write_private_file', pause_marker_write)
    first = threading.Thread(target=open_root, args=(first_outcomes, first_complete))
    second = threading.Thread(target=open_root, args=(second_outcomes, second_complete, second_entered))
    first.start()
    try:
        assert marker_write_started.wait(timeout=5)
        second.start()
        assert second_entered.wait(timeout=5)
        assert not second_complete.wait(timeout=1)
    finally:
        allow_marker_write.set()
        first.join(timeout=5)
        if second.ident is not None:
            second.join(timeout=5)
        for outcome in first_outcomes + second_outcomes:
            if isinstance(outcome, int):
                os.close(outcome)

    assert not first.is_alive()
    assert not second.is_alive()
    assert first_complete.is_set()
    assert second_complete.is_set()
    assert len(first_outcomes) == 1
    assert len(second_outcomes) == 1
    for outcome in first_outcomes + second_outcomes:
        if isinstance(outcome, BaseException):
            raise outcome


def test_trash_marker_write_failure_removes_new_root_and_allows_retry(tmp_path, monkeypatch):
    username = pwd.getpwuid(os.geteuid()).pw_name
    home = tmp_path / 'home' / username
    home.mkdir(parents=True)
    trash_root = home / '.webui-trash'

    def fail_marker_write(*args):
        raise OSError('marker write failed')

    with monkeypatch.context() as patched:
        patched.setattr(file_operations, '_write_private_file', fail_marker_write)
        with pytest.raises(OSError, match='marker write failed'):
            file_operations._open_trash_root(str(home), username, create=True)

    assert not trash_root.exists()
    trash_fd = file_operations._open_trash_root(str(home), username, create=True)
    assert trash_fd is not None
    os.close(trash_fd)
    assert (trash_root / file_operations._TRASH_MARKER).is_file()


def test_trash_rejects_and_preserves_preexisting_unmarked_root(tmp_path):
    username = pwd.getpwuid(os.geteuid()).pw_name
    home = tmp_path / 'home' / username
    trash_root = home / '.webui-trash'
    trash_root.mkdir(parents=True)

    with pytest.raises(file_operations.FileOperationError, match='forbidden'):
        file_operations._open_trash_root(str(home), username, create=True)

    assert trash_root.is_dir()
    assert not (trash_root / file_operations._TRASH_MARKER).exists()


def test_trash_lists_and_restores_nested_entries_without_provisioning_on_read(client):
    test_client, homes = client
    home = Path(homes['alice-id'].home)
    nested = home / 'documents' / 'draft.txt'
    nested.parent.mkdir()
    nested.write_text('draft')

    assert test_client.get('/home-files/trash', headers=home_mutation_headers()).json() == {'entries': []}
    assert not (home / '.webui-trash').exists()

    deleted = test_client.post(
        '/home-files/trash', json={'path': 'documents/draft.txt'}, headers=home_mutation_headers()
    )
    listed = test_client.get('/home-files/trash', headers=home_mutation_headers())
    restored = test_client.post(
        '/home-files/restore', json={'id': deleted.json()['id']}, headers=home_mutation_headers()
    )

    assert deleted.status_code == 200
    assert deleted.json()['original_path'] == 'documents/draft.txt'
    assert deleted.json()['name'] == 'draft.txt'
    assert deleted.json()['type'] == 'file'
    assert isinstance(deleted.json()['deleted_at'], float)
    assert listed.json() == {'entries': [deleted.json()]}
    assert '.webui-trash' not in {entry['name'] for entry in home_request(test_client).json()['entries']}
    assert restored.json() == {'path': 'documents/draft.txt'}
    assert nested.read_text() == 'draft'


def test_trash_handles_directories_duplicate_names_and_restore_conflicts(client):
    test_client, homes = client
    home = Path(homes['alice-id'].home)
    (home / 'one' / 'same').mkdir(parents=True)
    (home / 'one' / 'same' / 'a.txt').write_text('a')
    (home / 'two').mkdir()
    (home / 'two' / 'same').write_text('b')

    first = test_client.post('/home-files/trash', json={'path': 'one/same'}, headers=home_mutation_headers())
    second = test_client.post('/home-files/trash', json={'path': 'two/same'}, headers=home_mutation_headers())
    (home / 'one' / 'same').mkdir()
    conflict = test_client.post('/home-files/restore', json={'id': first.json()['id']}, headers=home_mutation_headers())
    restored = test_client.post(
        '/home-files/restore', json={'id': second.json()['id']}, headers=home_mutation_headers()
    )

    assert first.json()['id'] != second.json()['id']
    assert conflict.status_code == 409
    assert restored.json() == {'path': 'two/same'}
    assert (home / 'two' / 'same').read_text() == 'b'


def test_restore_requires_the_original_parent_to_still_exist(client):
    test_client, homes = client
    home = Path(homes['alice-id'].home)
    (home / 'gone').mkdir()
    (home / 'gone' / 'file.txt').write_text('recover me')

    deleted = test_client.post('/home-files/trash', json={'path': 'gone/file.txt'}, headers=home_mutation_headers())
    (home / 'gone').rmdir()
    restored = test_client.post(
        '/home-files/restore', json={'id': deleted.json()['id']}, headers=home_mutation_headers()
    )

    assert restored.status_code == 404
    assert test_client.get('/home-files/trash', headers=home_mutation_headers()).json()['entries'] == [deleted.json()]


def test_trash_and_restore_workspace_descendants_but_not_workspace_root(client):
    test_client, homes = client
    home = Path(homes['alice-id'].home)
    note = home / 'w' / 'project' / 'notes.md'
    note.parent.mkdir()
    note.write_text('project note')
    old_directory = home / 'w' / 'old'
    old_directory.mkdir()
    (old_directory / 'orphan.txt').write_text('orphan')

    note_deleted = test_client.post(
        '/home-files/trash', json={'path': 'w/project/notes.md'}, headers=home_mutation_headers()
    )
    old_deleted = test_client.post('/home-files/trash', json={'path': 'w/old'}, headers=home_mutation_headers())
    root_deleted = test_client.post('/home-files/trash', json={'path': 'w'}, headers=home_mutation_headers())
    note_restored = test_client.post(
        '/home-files/restore', json={'id': note_deleted.json()['id']}, headers=home_mutation_headers()
    )
    old_restored = test_client.post(
        '/home-files/restore', json={'id': old_deleted.json()['id']}, headers=home_mutation_headers()
    )

    assert root_deleted.status_code == 403
    assert note_restored.json() == {'path': 'w/project/notes.md'}
    assert old_restored.json() == {'path': 'w/old'}
    assert note.read_text() == 'project note'
    assert (old_directory / 'orphan.txt').read_text() == 'orphan'


def test_trash_refuses_reserved_workspace_and_tampered_metadata(client):
    test_client, homes = client
    home = Path(homes['alice-id'].home)
    (home / 'safe.txt').write_text('safe')

    assert home_request(test_client, params={'path': '.webui-trash'}).status_code == 403
    assert home_content(test_client, '.webui-trash/anything').status_code == 403
    assert test_client.post('/home-files/trash', json={'path': 'w'}, headers=home_mutation_headers()).status_code == 403
    assert (
        test_client.post(
            '/home-files/trash', json={'path': '.webui-trash'}, headers=home_mutation_headers()
        ).status_code
        == 403
    )

    deleted = test_client.post('/home-files/trash', json={'path': 'safe.txt'}, headers=home_mutation_headers()).json()
    wrapper = home / '.webui-trash' / deleted['id']
    (wrapper / 'metadata.json').write_text(json.dumps({**deleted, 'original_path': '../outside', 'name': 'outside'}))
    os.unlink(wrapper / 'data')
    os.symlink(home / 'w', wrapper / 'data')

    assert test_client.get('/home-files/trash', headers=home_mutation_headers()).json() == {'entries': []}
    assert (
        test_client.post('/home-files/restore', json={'id': deleted['id']}, headers=home_mutation_headers()).status_code
        == 403
    )
    assert not (home / 'safe.txt').exists()


def test_file_helpers_return_retryable_overload_without_blocking_browsing(client, monkeypatch):
    from open_terminal.utils import service_processes

    test_client, homes = client
    pool = service_processes.HelperPool(1, 0, 0)
    monkeypatch.setattr(service_processes, '_pool', pool)
    with pool.sync_slot():
        mkdir = test_client.post('/home-files/mkdir', json={'path': 'busy'}, headers=home_mutation_headers())
        upload = test_client.post(
            '/home-files/upload', data={'path': 'busy.txt'},
            files={'file': ('busy.txt', b'content')}, headers=home_mutation_headers(),
        )
        for response in (mkdir, upload):
            assert response.status_code == 503
            assert response.headers['retry-after'] == '1'
        assert home_request(test_client).status_code == 200
    assert not (Path(homes['alice-id'].home) / 'busy.txt').exists()
    assert test_client.post('/home-files/mkdir', json={'path': 'ready'}, headers=home_mutation_headers()).status_code == 200


def test_cancelled_upload_reaps_helper_and_releases_capacity(client, monkeypatch):
    import asyncio

    from open_terminal.utils import service_processes

    _, homes = client
    pool = service_processes.HelperPool(1, 0, 0)
    monkeypatch.setattr(service_processes, '_pool', pool)

    async def exercise():
        reading = asyncio.Event()

        class SlowUpload:
            size = None

            async def read(self, size):
                reading.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(workspace._run_upload_operation(homes['alice-id'], SlowUpload(), 'cancelled.txt', 'error'))
        await asyncio.wait_for(reading.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        assert pool._active == 0
        async with pool.slot():
            pass

    asyncio.run(exercise())
