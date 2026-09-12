"""Read and write notebook files as their owning OS user.

The notebook API invokes this helper with ``sudo -u`` in multi-user mode.
Every path below the trusted home is opened one descriptor at a time with
``O_NOFOLLOW`` so a user cannot replace a path component with a symlink after
the API has authorized it.
"""

import argparse
import json
import os

try:
    import pwd
except ImportError:  # pragma: no cover - notebook user isolation is Linux-only
    pwd = None
import secrets
import stat
import sys
from collections.abc import Iterable

_CONTROL_CHARACTERS = frozenset(chr(value) for value in range(32)) | {chr(127)}
_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_FILE_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
)


class NotebookFileError(Exception):
    """The notebook file was missing or unsafe to access."""


def _relative_parts(path: object) -> tuple[str, ...]:
    if not isinstance(path, str) or not path or path.startswith("/") or "\\" in path:
        raise NotebookFileError
    if any(char in _CONTROL_CHARACTERS for char in path):
        raise NotebookFileError
    parts = tuple(path.split("/"))
    if any(part in {"", ".", ".."} for part in parts):
        raise NotebookFileError
    return parts


def _home_descriptor(home: object, username: object) -> int:
    if not isinstance(home, str) or not os.path.isabs(home):
        raise NotebookFileError
    if not isinstance(username, str) or not username:
        raise NotebookFileError
    if pwd is None:
        raise NotebookFileError
    try:
        account = pwd.getpwnam(username)
    except KeyError as error:
        raise NotebookFileError from error
    if os.path.normpath(account.pw_dir) != os.path.normpath(home):
        raise NotebookFileError
    try:
        descriptor = os.open(home, _DIRECTORY_FLAGS)
    except OSError as error:
        raise NotebookFileError from error
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise NotebookFileError
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _parent_descriptor(home: object, username: object, parts: tuple[str, ...]) -> int:
    descriptor = _home_descriptor(home, username)
    try:
        for component in parts[:-1]:
            try:
                child = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
            except OSError as error:
                raise NotebookFileError from error
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def read_notebook(*, home: object, username: object, path: object) -> bytes:
    """Read one regular, non-symlinked notebook below *home*."""
    parts = _relative_parts(path)
    parent = _parent_descriptor(home, username, parts)
    try:
        try:
            descriptor = os.open(parts[-1], _FILE_FLAGS, dir_fd=parent)
        except OSError as error:
            raise NotebookFileError from error
    finally:
        os.close(parent)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise NotebookFileError
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 64 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    except OSError as error:
        raise NotebookFileError from error
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, content: bytes) -> None:
    view = memoryview(content)
    while view:
        written = os.write(descriptor, view)
        view = view[written:]


def write_notebook(
    *, home: object, username: object, path: object, content: bytes
) -> None:
    """Atomically replace one regular notebook below *home* without symlinks."""
    parts = _relative_parts(path)
    if not isinstance(content, bytes):
        raise NotebookFileError
    parent = _parent_descriptor(home, username, parts)
    temporary_name = f".{parts[-1]}.open-terminal-{secrets.token_hex(16)}"
    temporary_descriptor: int | None = None
    try:
        try:
            existing = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError as error:
            raise NotebookFileError from error
        except OSError as error:
            raise NotebookFileError from error
        if not stat.S_ISREG(existing.st_mode):
            raise NotebookFileError
        try:
            temporary_descriptor = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                stat.S_IMODE(existing.st_mode) & 0o666,
                dir_fd=parent,
            )
            os.fchmod(temporary_descriptor, stat.S_IMODE(existing.st_mode) & 0o666)
            _write_all(temporary_descriptor, content)
            os.fsync(temporary_descriptor)
            os.close(temporary_descriptor)
            temporary_descriptor = None
            os.replace(
                temporary_name,
                parts[-1],
                src_dir_fd=parent,
                dst_dir_fd=parent,
            )
            os.fsync(parent)
        except OSError as error:
            raise NotebookFileError from error
    finally:
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)
        try:
            os.unlink(temporary_name, dir_fd=parent)
        except FileNotFoundError:
            pass
        except OSError:
            pass
        os.close(parent)


def _read_stdin() -> bytes:
    chunks: list[bytes] = []
    while chunk := sys.stdin.buffer.read(64 * 1024):
        chunks.append(chunk)
    return b"".join(chunks)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("read", "write"))
    parser.add_argument("--home", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--path", required=True)
    arguments = parser.parse_args(argv)
    try:
        if arguments.action == "read":
            sys.stdout.buffer.write(
                read_notebook(
                    home=arguments.home,
                    username=arguments.username,
                    path=arguments.path,
                )
            )
        else:
            write_notebook(
                home=arguments.home,
                username=arguments.username,
                path=arguments.path,
                content=_read_stdin(),
            )
    except NotebookFileError:
        print(json.dumps({"ok": False, "error": "unavailable"}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
