"""Safe, per-user file mutations for the Open Terminal personal-files API.

This module is executed by the terminal server with ``sudo -u <username>``.
It deliberately has no dependencies on Open Terminal so the process inherits
neither API credentials nor application configuration.
"""

import argparse
import ctypes
import fcntl
import hashlib
import errno
import json
import os
import secrets
import stat
import sys
import time
import uuid
from typing import Any, BinaryIO


_CONTROL_CHARACTERS = frozenset(chr(value) for value in range(32)) | {chr(127)}
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
_FILE_FLAGS = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
_RENAME_NOREPLACE = 1
_TRASH_DIRECTORY = '.webui-trash'
_MAX_TEXT_SIZE = 2 * 1024 * 1024


class FileOperationError(Exception):
    """A controlled mutation failure exposed to the API caller."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _parts(path: object) -> tuple[str, ...]:
    if not isinstance(path, str) or not path or path.startswith('/') or '\\' in path:
        raise FileOperationError('invalid')
    if any(character in _CONTROL_CHARACTERS for character in path):
        raise FileOperationError('invalid')
    parts = tuple(path.split('/'))
    if any(part in {'', '.', '..'} for part in parts):
        raise FileOperationError('invalid')
    return parts


def _user_parts(path: object) -> tuple[str, ...]:
    parts = _parts(path)
    if parts[0] == _TRASH_DIRECTORY:
        raise FileOperationError('forbidden')
    return parts


def _home_parts(home: object, username: object) -> tuple[str, ...]:
    if not isinstance(home, str) or not isinstance(username, str) or not username:
        raise FileOperationError('forbidden')
    if not home.startswith('/') or '\\' in home or '/' in username:
        raise FileOperationError('forbidden')
    if any(character in _CONTROL_CHARACTERS for character in home + username):
        raise FileOperationError('forbidden')
    parts = tuple(part for part in home.split('/') if part)
    if not parts or parts[-1] != username or any(part in {'.', '..'} for part in parts):
        raise FileOperationError('forbidden')
    return parts


def _open_directory(parent_fd: int, name: str) -> int:
    try:
        descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    except FileNotFoundError as error:
        raise FileOperationError('missing') from error
    except PermissionError as error:
        raise FileOperationError('forbidden') from error
    except OSError as error:
        raise FileOperationError('forbidden') from error
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise FileOperationError('forbidden')
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_home(home: object, username: object) -> int:
    descriptor = os.open('/', _DIRECTORY_FLAGS)
    try:
        for component in _home_parts(home, username):
            child = _open_directory(descriptor, component)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_parent(home: object, username: object, parts: tuple[str, ...]) -> int:
    descriptor = _open_home(home, username)
    try:
        for component in parts[:-1]:
            child = _open_directory(descriptor, component)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _entry_mode(parent_fd: int, name: str) -> int | None:
    entry = _entry_stat(parent_fd, name)
    return entry.st_mode if entry is not None else None


def _entry_stat(parent_fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise FileOperationError('forbidden') from error


def _open_regular(parent_fd: int, name: str) -> int:
    try:
        descriptor = os.open(name, _FILE_FLAGS, dir_fd=parent_fd)
    except FileNotFoundError as error:
        raise FileOperationError('missing') from error
    except OSError as error:
        raise FileOperationError('forbidden') from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise FileOperationError('forbidden')
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read_bounded(stream: BinaryIO, maximum: int = _MAX_TEXT_SIZE) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while chunk := stream.read(64 * 1024):
        total += len(chunk)
        if total > maximum:
            raise FileOperationError('too_large')
        chunks.append(chunk)
    return b''.join(chunks)


def _read_descriptor(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    return _read_bounded(os.fdopen(os.dup(descriptor), 'rb', closefd=True))


def _require_regular_or_directory(mode: int | None) -> None:
    if mode is None:
        raise FileOperationError('missing')
    if not stat.S_ISREG(mode) and not stat.S_ISDIR(mode):
        raise FileOperationError('forbidden')


def _rename_no_replace(source_parent_fd: int, source: str, destination_parent_fd: int, destination: str) -> None:
    """Rename on Linux without the check-then-overwrite race of os.rename."""
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as error:
        raise FileOperationError('unsupported') from error
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        source_parent_fd,
        os.fsencode(source),
        destination_parent_fd,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileOperationError('conflict')
    if error_number == errno.ENOENT:
        raise FileOperationError('missing')
    if error_number in {errno.EACCES, errno.EPERM, errno.ELOOP, errno.ENOTDIR, errno.EISDIR}:
        raise FileOperationError('forbidden')
    raise FileOperationError('io')


def mkdir(*, home: object, username: object, path: object) -> dict[str, str]:
    parts = _user_parts(path)
    parent_fd = _open_parent(home, username, parts)
    try:
        existing_mode = _entry_mode(parent_fd, parts[-1])
        if existing_mode is not None:
            _require_regular_or_directory(existing_mode)
            raise FileOperationError('conflict')
        try:
            os.mkdir(parts[-1], mode=0o770, dir_fd=parent_fd)
        except FileExistsError as error:
            raise FileOperationError('conflict') from error
        except PermissionError as error:
            raise FileOperationError('forbidden') from error
        return {'path': '/'.join(parts)}
    finally:
        os.close(parent_fd)


def move(*, home: object, username: object, source: object, destination: object) -> dict[str, str]:
    source_parts = _user_parts(source)
    destination_parts = _user_parts(destination)
    source_parent_fd = _open_parent(home, username, source_parts)
    try:
        source_mode = _entry_mode(source_parent_fd, source_parts[-1])
        _require_regular_or_directory(source_mode)
        if stat.S_ISDIR(source_mode) and destination_parts[: len(source_parts)] == source_parts:
            raise FileOperationError('invalid')

        destination_parent_fd = _open_parent(home, username, destination_parts)
        try:
            destination_mode = _entry_mode(destination_parent_fd, destination_parts[-1])
            if destination_mode is not None:
                _require_regular_or_directory(destination_mode)
                raise FileOperationError('conflict')
            _rename_no_replace(source_parent_fd, source_parts[-1], destination_parent_fd, destination_parts[-1])
            return {'path': '/'.join(destination_parts)}
        finally:
            os.close(destination_parent_fd)
    finally:
        os.close(source_parent_fd)


def _upload_name(name: str, number: int) -> str:
    stem, extension = os.path.splitext(name)
    return f'{stem} ({number}){extension}'


def _write_upload(parent_fd: int, stream: BinaryIO, expected_size: int | None, *, mode: int = 0o660) -> str:
    temporary_name = f'.open-terminal-upload-{secrets.token_hex(16)}'
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
    except FileExistsError as error:
        raise FileOperationError('io') from error

    written = 0
    try:
        with os.fdopen(descriptor, 'wb', closefd=True) as output:
            while chunk := stream.read(64 * 1024):
                output.write(chunk)
                written += len(chunk)
            output.flush()
            os.fsync(output.fileno())
            if expected_size is not None and written != expected_size:
                raise FileOperationError('interrupted')
            os.fchmod(output.fileno(), mode)
        return temporary_name
    except BaseException:
        try:
            os.unlink(temporary_name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        raise


def upload(
    *,
    home: object,
    username: object,
    path: object,
    conflict: object,
    stream: BinaryIO,
    expected_size: object = None,
) -> dict[str, str]:
    parts = _user_parts(path)
    if conflict not in {'error', 'replace', 'keep-both'}:
        raise FileOperationError('invalid')
    if expected_size is not None and (not isinstance(expected_size, int) or expected_size < 0):
        raise FileOperationError('invalid')

    parent_fd = _open_parent(home, username, parts)
    temporary_name: str | None = None
    try:
        existing_mode = _entry_mode(parent_fd, parts[-1])
        if existing_mode is not None:
            _require_regular_or_directory(existing_mode)
        if existing_mode is not None and conflict == 'error':
            raise FileOperationError('conflict')
        if existing_mode is not None and conflict == 'replace' and not stat.S_ISREG(existing_mode):
            raise FileOperationError('conflict')

        mode = stat.S_IMODE(existing_mode) if conflict == 'replace' and existing_mode is not None else 0o660
        temporary_name = _write_upload(parent_fd, stream, expected_size, mode=mode)
        if conflict == 'replace':
            try:
                os.replace(temporary_name, parts[-1], src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            except PermissionError as error:
                raise FileOperationError('forbidden') from error
            except OSError as error:
                raise FileOperationError('io') from error
            temporary_name = None
            return {'path': '/'.join(parts)}

        candidate = parts[-1]
        number = 1
        while True:
            try:
                _rename_no_replace(parent_fd, temporary_name, parent_fd, candidate)
            except FileOperationError as error:
                if error.code != 'conflict' or conflict != 'keep-both':
                    raise
                candidate = _upload_name(parts[-1], number)
                number += 1
                continue
            temporary_name = None
            return {'path': '/'.join((*parts[:-1], candidate))}
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)


def save(*, home: object, username: object, path: object, version: object, stream: BinaryIO) -> dict[str, str]:
    parts = _user_parts(path)
    if not isinstance(version, str) or len(version) != 64:
        raise FileOperationError('invalid')
    try:
        int(version, 16)
    except ValueError as error:
        raise FileOperationError('invalid') from error
    content = _read_bounded(stream)
    if b'\x00' in content:
        raise FileOperationError('binary')
    try:
        content.decode('utf-8', 'strict')
    except UnicodeDecodeError as error:
        raise FileOperationError('binary') from error

    parent_fd = _open_parent(home, username, parts)
    temporary_name: str | None = None
    source_fd: int | None = None
    try:
        source_fd = _open_regular(parent_fd, parts[-1])
        fcntl.flock(source_fd, fcntl.LOCK_EX)
        source_stat = os.fstat(source_fd)
        if hashlib.sha256(_read_descriptor(source_fd)).hexdigest() != version:
            raise FileOperationError('stale')

        temporary_name = f'.open-terminal-save-{secrets.token_hex(16)}'
        try:
            output_fd = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                stat.S_IMODE(source_stat.st_mode),
                dir_fd=parent_fd,
            )
        except FileExistsError as error:
            raise FileOperationError('io') from error
        with os.fdopen(output_fd, 'wb', closefd=True) as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
            os.fchmod(output.fileno(), stat.S_IMODE(source_stat.st_mode))

        current_stat = _entry_stat(parent_fd, parts[-1])
        if (
            current_stat is None
            or not stat.S_ISREG(current_stat.st_mode)
            or (current_stat.st_dev, current_stat.st_ino) != (source_stat.st_dev, source_stat.st_ino)
            or hashlib.sha256(_read_descriptor(source_fd)).hexdigest() != version
        ):
            raise FileOperationError('stale')
        try:
            os.replace(temporary_name, parts[-1], src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        except PermissionError as error:
            raise FileOperationError('forbidden') from error
        except OSError as error:
            raise FileOperationError('io') from error
        temporary_name = None
        return {'path': '/'.join(parts), 'version': hashlib.sha256(content).hexdigest()}
    finally:
        if source_fd is not None:
            os.close(source_fd)
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)


_TRASH_MARKER = '.open-terminal-trash-v1'
_TRASH_MARKER_VALUE = b'open-terminal-trash-v1\n'


def _open_file_bytes(parent_fd: int, name: str, maximum: int = 16 * 1024) -> bytes:
    descriptor = _open_regular(parent_fd, name)
    try:
        return _read_bounded(os.fdopen(descriptor, 'rb', closefd=False), maximum)
    finally:
        os.close(descriptor)


def _write_private_file(parent_fd: int, name: str, payload: bytes) -> None:
    try:
        descriptor = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
    except FileExistsError as error:
        raise FileOperationError('forbidden') from error
    try:
        with os.fdopen(descriptor, 'wb', closefd=True) as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        try:
            os.unlink(name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        raise


def _open_trash_root(home: object, username: object, *, create: bool) -> int | None:
    home_fd = _open_home(home, username)
    created = False
    try:
        # Lock the stable home descriptor so other helpers cannot observe a
        # partially initialized trash directory, including during rollback.
        fcntl.flock(home_fd, fcntl.LOCK_EX)
        mode = _entry_mode(home_fd, _TRASH_DIRECTORY)
        if mode is None:
            if not create:
                return None
            try:
                os.mkdir(_TRASH_DIRECTORY, mode=0o700, dir_fd=home_fd)
                created = True
            except FileExistsError:
                pass
            mode = _entry_mode(home_fd, _TRASH_DIRECTORY)
        if mode is None or not stat.S_ISDIR(mode):
            raise FileOperationError('forbidden')
        trash_fd = _open_directory(home_fd, _TRASH_DIRECTORY)
        try:
            marker_mode = _entry_mode(trash_fd, _TRASH_MARKER)
            if marker_mode is None and created:
                _write_private_file(trash_fd, _TRASH_MARKER, _TRASH_MARKER_VALUE)
            elif (
                marker_mode is None
                or not stat.S_ISREG(marker_mode)
                or _open_file_bytes(trash_fd, _TRASH_MARKER) != _TRASH_MARKER_VALUE
            ):
                raise FileOperationError('forbidden')
            return trash_fd
        except BaseException:
            os.close(trash_fd)
            raise
    except BaseException:
        if created:
            try:
                os.rmdir(_TRASH_DIRECTORY, dir_fd=home_fd)
            except OSError:
                pass
        raise
    finally:
        os.close(home_fd)


def _trash_id(value: object) -> str:
    if not isinstance(value, str):
        raise FileOperationError('invalid')
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as error:
        raise FileOperationError('invalid') from error
    return parsed.hex


def _metadata(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise FileOperationError('forbidden')
    try:
        record_id = _trash_id(payload.get('id'))
        original_path = payload.get('original_path')
        parts = _user_parts(original_path)
    except FileOperationError as error:
        raise FileOperationError('forbidden') from error
    if parts == ('w',) or payload.get('name') != parts[-1] or payload.get('type') not in {'file', 'directory'}:
        raise FileOperationError('forbidden')
    deleted_at = payload.get('deleted_at')
    if isinstance(deleted_at, bool) or not isinstance(deleted_at, (int, float)):
        raise FileOperationError('forbidden')
    return {
        'id': record_id,
        'original_path': original_path,
        'name': parts[-1],
        'type': payload['type'],
        'deleted_at': deleted_at,
    }


def _read_metadata(wrapper_fd: int) -> dict[str, object]:
    try:
        decoded = _open_file_bytes(wrapper_fd, 'metadata.json').decode('utf-8', 'strict')
        return _metadata(json.loads(decoded))
    except (UnicodeDecodeError, json.JSONDecodeError, FileOperationError) as error:
        if isinstance(error, FileOperationError):
            raise
        raise FileOperationError('forbidden') from error


def trash(*, home: object, username: object, path: object) -> dict[str, object]:
    parts = _user_parts(path)
    if parts == ('w',):
        raise FileOperationError('forbidden')
    source_parent_fd = _open_parent(home, username, parts)
    trash_fd: int | None = None
    wrapper_fd: int | None = None
    wrapper_name: str | None = None
    try:
        source_mode = _entry_mode(source_parent_fd, parts[-1])
        _require_regular_or_directory(source_mode)
        entry_type = 'directory' if stat.S_ISDIR(source_mode) else 'file'
        trash_fd = _open_trash_root(home, username, create=True)
        assert trash_fd is not None
        record_id = uuid.uuid4().hex
        wrapper_name = record_id
        os.mkdir(wrapper_name, mode=0o700, dir_fd=trash_fd)
        wrapper_fd = _open_directory(trash_fd, wrapper_name)
        record: dict[str, object] = {
            'id': record_id,
            'original_path': '/'.join(parts),
            'name': parts[-1],
            'type': entry_type,
            'deleted_at': time.time(),
        }
        _write_private_file(
            wrapper_fd,
            'metadata.json',
            json.dumps(record, separators=(',', ':'), ensure_ascii=False).encode('utf-8'),
        )
        _rename_no_replace(source_parent_fd, parts[-1], wrapper_fd, 'data')
        os.fsync(wrapper_fd)
        return record
    except BaseException:
        if wrapper_fd is not None:
            try:
                if _entry_mode(wrapper_fd, 'data') is None:
                    os.unlink('metadata.json', dir_fd=wrapper_fd)
            except (FileNotFoundError, OSError):
                pass
        if wrapper_name is not None and trash_fd is not None:
            try:
                os.rmdir(wrapper_name, dir_fd=trash_fd)
            except OSError:
                pass
        raise
    finally:
        if wrapper_fd is not None:
            os.close(wrapper_fd)
        if trash_fd is not None:
            os.close(trash_fd)
        os.close(source_parent_fd)


def list_trash(*, home: object, username: object) -> dict[str, object]:
    trash_fd = _open_trash_root(home, username, create=False)
    if trash_fd is None:
        return {'entries': []}
    entries: list[dict[str, object]] = []
    try:
        for name in os.listdir(trash_fd):
            try:
                _trash_id(name)
                wrapper_fd = _open_directory(trash_fd, name)
                try:
                    record = _read_metadata(wrapper_fd)
                    data_mode = _entry_mode(wrapper_fd, 'data')
                    if (
                        record['id'] == name
                        and data_mode is not None
                        and (
                            (record['type'] == 'file' and stat.S_ISREG(data_mode))
                            or (record['type'] == 'directory' and stat.S_ISDIR(data_mode))
                        )
                    ):
                        entries.append(record)
                finally:
                    os.close(wrapper_fd)
            except (FileOperationError, OSError):
                continue
        return {'entries': sorted(entries, key=lambda entry: (float(entry['deleted_at']), str(entry['id'])))}
    finally:
        os.close(trash_fd)


def restore(*, home: object, username: object, record_id: object) -> dict[str, str]:
    record_id = _trash_id(record_id)
    trash_fd = _open_trash_root(home, username, create=False)
    if trash_fd is None:
        raise FileOperationError('missing')
    wrapper_fd: int | None = None
    destination_parent_fd: int | None = None
    try:
        wrapper_fd = _open_directory(trash_fd, record_id)
        record = _read_metadata(wrapper_fd)
        if record['id'] != record_id:
            raise FileOperationError('forbidden')
        data_mode = _entry_mode(wrapper_fd, 'data')
        if (
            data_mode is None
            or (record['type'] == 'file' and not stat.S_ISREG(data_mode))
            or (record['type'] == 'directory' and not stat.S_ISDIR(data_mode))
        ):
            raise FileOperationError('forbidden')
        parts = _user_parts(record['original_path'])
        if parts == ('w',):
            raise FileOperationError('forbidden')
        destination_parent_fd = _open_parent(home, username, parts)
        if _entry_mode(destination_parent_fd, parts[-1]) is not None:
            raise FileOperationError('conflict')
        _rename_no_replace(wrapper_fd, 'data', destination_parent_fd, parts[-1])
        try:
            os.unlink('metadata.json', dir_fd=wrapper_fd)
            os.rmdir(record_id, dir_fd=trash_fd)
        except OSError:
            pass
        return {'path': '/'.join(parts)}
    finally:
        if destination_parent_fd is not None:
            os.close(destination_parent_fd)
        if wrapper_fd is not None:
            os.close(wrapper_fd)
        os.close(trash_fd)


def execute(
    action: str, *, home: object, username: object, payload: dict[str, Any], stream: BinaryIO
) -> dict[str, str]:
    if action == 'mkdir':
        return mkdir(home=home, username=username, path=payload.get('path'))
    if action == 'move':
        return move(home=home, username=username, source=payload.get('source'), destination=payload.get('destination'))
    if action == 'upload':
        return upload(
            home=home,
            username=username,
            path=payload.get('path'),
            conflict=payload.get('conflict', 'error'),
            stream=stream,
            expected_size=payload.get('expected_size'),
        )
    if action == 'save':
        return save(
            home=home,
            username=username,
            path=payload.get('path'),
            version=payload.get('version'),
            stream=stream,
        )
    if action == 'trash':
        return trash(home=home, username=username, path=payload.get('path'))
    if action == 'list-trash':
        return list_trash(home=home, username=username)
    if action == 'restore':
        return restore(home=home, username=username, record_id=payload.get('id'))
    raise FileOperationError('invalid')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('mkdir', 'move', 'upload', 'save', 'trash', 'list-trash', 'restore'))
    parser.add_argument('--home', required=True)
    parser.add_argument('--username', required=True)
    parser.add_argument('--payload', required=True)
    arguments = parser.parse_args()
    try:
        payload = json.loads(arguments.payload)
        if not isinstance(payload, dict):
            raise FileOperationError('invalid')
        result = execute(
            arguments.action,
            home=arguments.home,
            username=arguments.username,
            payload=payload,
            stream=sys.stdin.buffer,
        )
    except FileOperationError as error:
        print(json.dumps({'ok': False, 'error': error.code}), flush=True)
        return 1
    except Exception:
        print(json.dumps({'ok': False, 'error': 'io'}), flush=True)
        return 1
    print(json.dumps({'ok': True, **result}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
