"""Personal file routes for Open Terminal.

The Open Terminal filesystem object is the source of a user's home directory.
Routes resolve every component from an open directory descriptor without
following symlink path components.
"""

import asyncio
import hashlib
import json
import mimetypes
import os
import stat
import subprocess
import sys
from errno import EACCES, EPERM
from collections.abc import Callable, Iterator
from typing import Annotated, Any, Literal
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask


CHUNK_SIZE = 64 * 1024
MAX_TEXT_SIZE = 2 * 1024 * 1024
_TRASH_DIRECTORY = '.webui-trash'
_CONTROL_CHARACTERS = frozenset(chr(value) for value in range(32)) | {chr(127)}
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
_FILE_FLAGS = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
_ACTIVE_CONTENT_TYPES = frozenset(
    {
        'application/ecmascript',
        'application/javascript',
        'application/xhtml+xml',
        'image/svg+xml',
        'text/html',
        'text/javascript',
    }
)


class WorkspacePathError(Exception):
    """A requested workspace path is missing or unsafe to access."""


class WorkspaceMissingError(WorkspacePathError):
    """A requested workspace path component does not exist."""


class WorkspacePermissionError(WorkspacePathError):
    """The terminal process lacks native filesystem permission."""


class _MkdirRequest(BaseModel):
    path: str


class _MoveRequest(BaseModel):
    source: str
    destination: str


class _SaveRequest(BaseModel):
    path: str
    content: str
    version: str


class _TrashRequest(BaseModel):
    path: str


class _RestoreRequest(BaseModel):
    id: str


def _validate_relative_path(path: str) -> tuple[str, ...]:
    """Return path components without decoding percent signs a second time."""
    if not isinstance(path, str) or '\\' in path or any(char in _CONTROL_CHARACTERS for char in path):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Invalid workspace path')
    if not path:
        return ()
    if path.startswith('/'):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Invalid workspace path')

    parts = tuple(path.split('/'))
    if any(part in {'', '.', '..'} for part in parts):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Invalid workspace path')
    return parts


def _validate_home_path(path: str) -> tuple[str, ...]:
    """Validate a normal home-files path, excluding terminal bookkeeping."""
    parts = _validate_relative_path(path)
    if parts and parts[0] == _TRASH_DIRECTORY:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='Workspace access denied')
    return parts


def _filesystem_home_parts(filesystem: Any) -> tuple[str, ...]:
    """Validate the trusted Open Terminal home before descriptor traversal."""
    username = getattr(filesystem, 'username', None)
    home = getattr(filesystem, 'home', None)
    if not isinstance(username, str) or not username or '/' in username or '\\' in username:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='Workspace is unavailable')
    if any(char in _CONTROL_CHARACTERS for char in username):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='Workspace is unavailable')

    try:
        home_path = os.fspath(home)
    except TypeError as error:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='Workspace is unavailable') from error
    if not isinstance(home_path, str) or not home_path.startswith('/') or '\\' in home_path:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='Workspace is unavailable')

    parts = tuple(part for part in home_path.split('/') if part)
    if not parts or parts[-1] != username or any(part in {'.', '..'} for part in parts):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='Workspace is unavailable')
    return parts


def _workspace_home_parts(filesystem: Any) -> tuple[str, ...]:
    return (*_filesystem_home_parts(filesystem), 'w')


def _open_directory(parent_fd: int, name: str) -> int:
    try:
        descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    except FileNotFoundError as error:
        raise WorkspaceMissingError from error
    except OSError as error:
        if error.errno in {EACCES, EPERM}:
            raise WorkspacePermissionError from error
        raise WorkspacePathError from error
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise WorkspacePathError
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_root(filesystem: Any, root_parts: Callable[[Any], tuple[str, ...]]) -> int | None:
    """Open a trusted filesystem root one component at a time without following links."""
    descriptor = os.open('/', _DIRECTORY_FLAGS)
    try:
        for component in root_parts(filesystem):
            try:
                child = _open_directory(descriptor, component)
            except WorkspaceMissingError:
                os.close(descriptor)
                return None
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_relative_directory(
    filesystem: Any,
    parts: tuple[str, ...],
    root_parts: Callable[[Any], tuple[str, ...]],
) -> int | None:
    descriptor = _open_root(filesystem, root_parts)
    if descriptor is None:
        return None
    try:
        for component in parts:
            child = _open_directory(descriptor, component)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_relative_file(
    filesystem: Any,
    parts: tuple[str, ...],
    root_parts: Callable[[Any], tuple[str, ...]],
) -> int | None:
    if not parts:
        raise WorkspacePathError
    descriptor = _open_root(filesystem, root_parts)
    if descriptor is None:
        return None
    try:
        for component in parts[:-1]:
            child = _open_directory(descriptor, component)
            os.close(descriptor)
            descriptor = child
        try:
            file_descriptor = os.open(parts[-1], _FILE_FLAGS, dir_fd=descriptor)
        except FileNotFoundError as error:
            raise WorkspaceMissingError from error
        except OSError as error:
            if error.errno in {EACCES, EPERM}:
                raise WorkspacePermissionError from error
            raise WorkspacePathError from error
        try:
            if not stat.S_ISREG(os.fstat(file_descriptor).st_mode):
                raise WorkspacePathError
            return file_descriptor
        except BaseException:
            os.close(file_descriptor)
            raise
    finally:
        os.close(descriptor)


def _workspace_entries(descriptor: int, path: str) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    try:
        names = os.listdir(descriptor)
    except OSError as error:
        raise WorkspacePathError from error

    for name in names:
        try:
            file_stat = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except OSError:
            continue
        if stat.S_ISREG(file_stat.st_mode):
            entry_type = 'file'
        elif stat.S_ISDIR(file_stat.st_mode):
            entry_type = 'directory'
        else:
            continue
        entries.append(
            {
                'name': name,
                'path': f'{path}/{name}' if path else name,
                'type': entry_type,
                'size': file_stat.st_size,
                'modified': file_stat.st_mtime,
            }
        )
    return sorted(entries, key=lambda entry: str(entry['name']))


class _OpenFile:
    """An idempotently closed descriptor, shared by stream and response cleanup."""

    def __init__(self, descriptor: int):
        self.descriptor: int | None = descriptor

    def read(self) -> bytes:
        if self.descriptor is None:
            return b''
        return os.read(self.descriptor, CHUNK_SIZE)

    def close(self) -> None:
        if self.descriptor is not None:
            descriptor, self.descriptor = self.descriptor, None
            os.close(descriptor)


def _stream_file(file: _OpenFile) -> Iterator[bytes]:
    try:
        while chunk := file.read():
            yield chunk
    finally:
        file.close()


def _read_text_file(descriptor: int) -> tuple[str, str]:
    """Read a bounded UTF-8 regular file and calculate its edit version."""
    chunks: list[bytes] = []
    total = 0
    try:
        while chunk := os.read(descriptor, CHUNK_SIZE):
            total += len(chunk)
            if total > MAX_TEXT_SIZE:
                raise _mutation_error_response('too_large', 'text')
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    payload = b''.join(chunks)
    if b'\x00' in payload:
        raise _mutation_error_response('binary', 'text')
    try:
        content = payload.decode('utf-8', 'strict')
    except UnicodeDecodeError:
        raise _mutation_error_response('binary', 'text') from None
    return content, hashlib.sha256(payload).hexdigest()


def _require_user_id(request: Request) -> str:
    user_id = request.headers.get('X-User-Id')
    if not user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='X-User-Id is required')
    return user_id


def _workspace_error_response(error: WorkspacePathError) -> HTTPException:
    if isinstance(error, WorkspacePermissionError):
        return HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='Workspace access denied')
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Workspace path not found')


def _mutation_error_response(error: str, action: str) -> HTTPException:
    if error == 'stale':
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail='File has changed.')
    if error == 'conflict':
        detail = 'Folder already exists.' if action == 'mkdir' else 'File already exists.'
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail)
    if error == 'invalid':
        detail = 'Cannot move a folder into itself.' if action == 'move' else 'Invalid file path.'
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)
    if error == 'missing':
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Workspace path not found')
    if error == 'binary':
        return HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail='File is not UTF-8 text.')
    if error == 'too_large':
        return HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail='File is too large.')
    if error in {'forbidden', 'interrupted'}:
        return HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='Workspace access denied')
    return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='Workspace operation failed')


def _validate_mutation_relative_path(path: str) -> tuple[str, ...]:
    try:
        parts = _validate_relative_path(path)
    except HTTPException as error:
        raise HTTPException(status_code=error.status_code, detail='Invalid file path.') from error
    if not parts:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Invalid file path.')
    return parts


def _validate_home_mutation_path(path: str) -> tuple[str, ...]:
    parts = _validate_mutation_relative_path(path)
    if parts[0] == _TRASH_DIRECTORY:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='Workspace access denied')
    return parts


def _operation_command(filesystem: Any, action: str, payload: dict[str, object]) -> list[str]:
    """Build a fixed helper command with a deliberately minimal environment."""
    username = getattr(filesystem, 'username', None)
    home = getattr(filesystem, 'home', None)
    if not isinstance(username, str) or not isinstance(home, str):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='Workspace is unavailable')
    _filesystem_home_parts(filesystem)
    return [
        'sudo',
        '-n',
        '-u',
        username,
        '--',
        sys.executable,
        '-I',
        '-m',
        'open_terminal.file_operations',
        action,
        '--home',
        home,
        '--username',
        username,
        '--payload',
        json.dumps(payload, ensure_ascii=False, separators=(',', ':')),
    ]


_HELPER_ENV = {'PATH': '/usr/sbin:/usr/bin:/sbin:/bin', 'LANG': 'C', 'PYTHONUNBUFFERED': '1'}


def _decode_operation_result(completed: subprocess.CompletedProcess[bytes], action: str) -> dict[str, object]:
    try:
        result = json.loads(completed.stdout.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise _mutation_error_response('io', action) from None
    if not isinstance(result, dict):
        raise _mutation_error_response('io', action)
    if result.get('ok') is True:
        return {key: value for key, value in result.items() if key != 'ok'}
    error = result.get('error')
    if not isinstance(error, str):
        error = 'io'
    raise _mutation_error_response(error, action)


def _run_file_operation(
    filesystem: Any, action: str, payload: dict[str, object], *, input_bytes: bytes | None = None
) -> dict[str, object]:
    try:
        completed = subprocess.run(
            _operation_command(filesystem, action, payload),
            stdin=subprocess.DEVNULL if input_bytes is None else None,
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=_HELPER_ENV,
            cwd='/',
            check=False,
        )
    except OSError:
        raise _mutation_error_response('io', action) from None
    return _decode_operation_result(completed, action)


def _run_save_operation(filesystem: Any, path: str, content: str, version: str) -> dict[str, object]:
    try:
        encoded = content.encode('utf-8', 'strict')
    except UnicodeEncodeError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Invalid text content.') from None
    if len(encoded) > MAX_TEXT_SIZE:
        raise HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail='File is too large.')
    return _run_file_operation(filesystem, 'save', {'path': path, 'version': version}, input_bytes=encoded)


async def _run_upload_operation(
    filesystem: Any,
    upload: UploadFile,
    path: str,
    conflict: Literal['error', 'replace', 'keep-both'],
) -> dict[str, str]:
    size = getattr(upload, 'size', None)
    payload: dict[str, object] = {'path': path, 'conflict': conflict}
    if isinstance(size, int):
        payload['expected_size'] = size
    try:
        process = subprocess.Popen(
            _operation_command(filesystem, 'upload', payload),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=_HELPER_ENV,
            cwd='/',
        )
    except OSError:
        raise _mutation_error_response('io', 'upload') from None

    try:
        assert process.stdin is not None
        try:
            while chunk := await upload.read(CHUNK_SIZE):
                try:
                    await asyncio.to_thread(process.stdin.write, chunk)
                except BrokenPipeError:
                    break
        finally:
            try:
                await asyncio.to_thread(process.stdin.close)
            except BrokenPipeError:
                pass
            process.stdin = None
        stdout, _ = await asyncio.to_thread(process.communicate)
    except BaseException:
        process.kill()
        await asyncio.to_thread(process.wait)
        raise

    return _decode_operation_result(
        subprocess.CompletedProcess(process.args, process.returncode, stdout, b''), 'upload'
    )


def install_workspace_routes(app: Any, get_filesystem: Callable[..., Any], verify_api_key: Callable[..., Any]) -> None:
    """Install authenticated personal-file read and mutation endpoints on *app*."""

    router = APIRouter(dependencies=[Depends(verify_api_key)], include_in_schema=False)

    def workspace_filesystem(
        _: Annotated[str, Depends(_require_user_id)],
        filesystem: Annotated[Any, Depends(get_filesystem)],
    ) -> Any:
        _filesystem_home_parts(filesystem)
        return filesystem

    def install_file_routes(path_prefix: str, root_parts: Callable[[Any], tuple[str, ...]]) -> None:
        validate_path = _validate_home_path if path_prefix == '/home-files' else _validate_relative_path

        @router.get(path_prefix)
        def list_files(
            filesystem: Annotated[Any, Depends(workspace_filesystem)],
            path: Annotated[str, Query()] = '',
        ) -> dict[str, object]:
            parts = validate_path(path)
            try:
                descriptor = _open_relative_directory(filesystem, parts, root_parts)
            except WorkspaceMissingError:
                descriptor = None
            except WorkspacePathError as error:
                raise _workspace_error_response(error) from error
            if descriptor is None:
                if not parts:
                    return {'path': path, 'entries': []}
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Workspace path not found')
            try:
                entries = _workspace_entries(descriptor, path)
                if path_prefix == '/home-files' and not parts:
                    entries = [entry for entry in entries if entry['name'] != _TRASH_DIRECTORY]
                return {'path': path, 'entries': entries}
            except WorkspacePathError as error:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Workspace path not found') from error
            finally:
                os.close(descriptor)

        @router.get(f'{path_prefix}/content')
        def file_content(
            filesystem: Annotated[Any, Depends(workspace_filesystem)],
            path: Annotated[str, Query()] = '',
        ) -> StreamingResponse:
            parts = validate_path(path)
            try:
                descriptor = _open_relative_file(filesystem, parts, root_parts)
            except WorkspaceMissingError:
                descriptor = None
            except WorkspacePathError as error:
                raise _workspace_error_response(error) from error
            if descriptor is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Workspace file not found')

            filename = parts[-1]
            content_type = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
            if content_type in _ACTIVE_CONTENT_TYPES:
                content_type = 'application/octet-stream'
            content_disposition = f'attachment; filename="download"; filename*=UTF-8\'\'{quote(filename, safe="")}'
            file = _OpenFile(descriptor)
            return StreamingResponse(
                _stream_file(file),
                media_type=content_type,
                headers={
                    'Content-Disposition': content_disposition,
                    'X-Content-Type-Options': 'nosniff',
                },
                background=BackgroundTask(file.close),
            )

    @router.get('/home-files/text')
    def file_text(
        filesystem: Annotated[Any, Depends(workspace_filesystem)],
        path: Annotated[str, Query()] = '',
    ) -> dict[str, str]:
        parts = _validate_home_mutation_path(path)
        try:
            descriptor = _open_relative_file(filesystem, parts, _filesystem_home_parts)
        except WorkspaceMissingError:
            descriptor = None
        except WorkspacePathError as error:
            raise _workspace_error_response(error) from error
        if descriptor is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Workspace file not found')
        content, version = _read_text_file(descriptor)
        return {'path': path, 'content': content, 'version': version}

    @router.post('/home-files/mkdir')
    async def make_directory(
        body: _MkdirRequest,
        filesystem: Annotated[Any, Depends(workspace_filesystem)],
    ) -> dict[str, str]:
        _validate_home_mutation_path(body.path)
        return await asyncio.to_thread(_run_file_operation, filesystem, 'mkdir', {'path': body.path})

    @router.post('/home-files/move')
    async def move_file(
        body: _MoveRequest,
        filesystem: Annotated[Any, Depends(workspace_filesystem)],
    ) -> dict[str, str]:
        _validate_home_mutation_path(body.source)
        _validate_home_mutation_path(body.destination)
        return await asyncio.to_thread(
            _run_file_operation,
            filesystem,
            'move',
            {'source': body.source, 'destination': body.destination},
        )

    @router.post('/home-files/upload')
    async def upload_file(
        filesystem: Annotated[Any, Depends(workspace_filesystem)],
        file: Annotated[UploadFile, File()],
        path: Annotated[str, Form()],
        conflict: Annotated[Literal['error', 'replace', 'keep-both'], Form()] = 'error',
    ) -> dict[str, str]:
        _validate_home_mutation_path(path)
        return await _run_upload_operation(filesystem, file, path, conflict)

    @router.post('/home-files/save')
    async def save_text(
        body: _SaveRequest,
        filesystem: Annotated[Any, Depends(workspace_filesystem)],
    ) -> dict[str, object]:
        _validate_home_mutation_path(body.path)
        if not isinstance(body.version, str) or len(body.version) != 64:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Invalid file version.')
        return await asyncio.to_thread(_run_save_operation, filesystem, body.path, body.content, body.version)

    @router.post('/home-files/trash')
    async def trash_file(
        body: _TrashRequest,
        filesystem: Annotated[Any, Depends(workspace_filesystem)],
    ) -> dict[str, object]:
        parts = _validate_home_mutation_path(body.path)
        if parts == ('w',):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='Workspace access denied')
        return await asyncio.to_thread(_run_file_operation, filesystem, 'trash', {'path': body.path})

    @router.get('/home-files/trash')
    async def list_trash(filesystem: Annotated[Any, Depends(workspace_filesystem)]) -> dict[str, object]:
        return await asyncio.to_thread(_run_file_operation, filesystem, 'list-trash', {})

    @router.post('/home-files/restore')
    async def restore_file(
        body: _RestoreRequest,
        filesystem: Annotated[Any, Depends(workspace_filesystem)],
    ) -> dict[str, object]:
        return await asyncio.to_thread(_run_file_operation, filesystem, 'restore', {'id': body.id})

    install_file_routes('/workspace-files', _workspace_home_parts)
    install_file_routes('/home-files', _filesystem_home_parts)

    app.include_router(router)
