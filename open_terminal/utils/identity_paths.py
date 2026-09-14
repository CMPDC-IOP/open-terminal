"""Safe ownership repair for one managed Linux home directory.

Every path component is opened by descriptor with ``O_NOFOLLOW``. Recursive
repair keeps the object being changed open, so a user cannot redirect an
ownership change by swapping a pathname for a symlink.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import stat
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path

from open_terminal.utils.identity_store import IdentityStoreError, _trusted_owner


_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
_PATH_FLAGS = os.O_PATH | os.O_CLOEXEC | os.O_NOFOLLOW
_AT_EMPTY_PATH = 0x1000
_CONTROL_CHARACTERS = frozenset(chr(value) for value in range(32)) | {chr(127)}


def _fchownat():
    try:
        function = ctypes.CDLL(None, use_errno=True).fchownat
    except AttributeError:  # pragma: no cover - Linux always provides it
        return None
    function.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_int,
    )
    function.restype = ctypes.c_int
    return function


_FCHOWNAT = _fchownat()


def _unsafe(message: str, exc: BaseException | None = None) -> IdentityStoreError:
    error = IdentityStoreError("unsafe managed home path: " + message)
    if exc is not None:
        error.__cause__ = exc
    return error


def _username(value: object) -> str:
    try:
        value = os.fspath(value)
    except TypeError as exc:
        raise _unsafe("invalid username", exc) from exc
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or os.sep in value
        or (os.altsep is not None and os.altsep in value)
        or "\x00" in value
        or any(char in _CONTROL_CHARACTERS for char in value)
    ):
        raise _unsafe("invalid username")
    return value


def _root_parts(value: object) -> tuple[str, ...]:
    try:
        value = os.fspath(value)
    except TypeError as exc:
        raise _unsafe("invalid home root", exc) from exc
    if not isinstance(value, str) or not os.path.isabs(value) or "\x00" in value:
        raise _unsafe("home root must be an absolute path")
    parts = tuple(part for part in value.split(os.sep) if part)
    if any(
        part in {".", ".."} or any(char in _CONTROL_CHARACTERS for char in part)
        for part in parts
    ):
        raise _unsafe("invalid home root")
    return parts


def _open_home_root(parts: tuple[str, ...]) -> int:
    try:
        descriptor = os.open(os.sep, _DIRECTORY_FLAGS)
    except OSError as exc:  # pragma: no cover - a working Linux host has /
        raise _unsafe("cannot open filesystem root", exc) from exc
    try:
        for part in parts:
            try:
                child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
            except OSError as exc:
                raise _unsafe(f"cannot open home-root component {part!r}", exc) from exc
            os.close(descriptor)
            descriptor = child
        info = os.fstat(descriptor)
        if not _trusted_owner(info.st_uid):
            raise _unsafe("home root is not owned by a trusted service account")
        if stat.S_IMODE(info.st_mode) & 0o022:
            raise _unsafe("home root is group- or world-writable")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _same_inode(first: os.stat_result, second: os.stat_result) -> bool:
    return first.st_dev == second.st_dev and first.st_ino == second.st_ino


class Entry:
    """A live descriptor for one entry yielded by :meth:`Home.walk`."""

    def __init__(self, descriptor: int, path: Path, relative_path: str, info: os.stat_result):
        self._descriptor = descriptor
        self.path = str(path)
        self.relative_path = relative_path
        self.stat = info

    def chown(self, uid: int, gid: int) -> None:
        """Change this inode's ownership without following a symlink."""
        current = os.fstat(self._descriptor)
        if stat.S_ISREG(current.st_mode) and current.st_nlink > 1:
            raise _unsafe(f"hard-linked regular file at {self.relative_path!r}")
        if _FCHOWNAT is None:  # pragma: no cover - Linux always provides it
            raise _unsafe("fchownat is unavailable")
        if _FCHOWNAT(self._descriptor, b"", uid, gid, _AT_EMPTY_PATH) != 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, os.strerror(error_number))
        self.stat = os.fstat(self._descriptor)


class Home:
    """A managed home held open for a safe walk or repair."""

    def __init__(self, descriptor: int, path: Path):
        self._descriptor = descriptor
        self.path = path
        self._closed = False

    def walk(self) -> Iterator[Entry]:
        if self._closed:
            raise ValueError("managed home handle is closed")
        yield from self._walk_directory(os.dup(self._descriptor), self.path, ".")

    def _walk_directory(self, descriptor: int, path: Path, relative_path: str) -> Iterator[Entry]:
        try:
            yield Entry(descriptor, path, relative_path, os.fstat(descriptor))
            with os.scandir(descriptor) as scan:
                names = sorted(item.name for item in scan)
            for name in names:
                child_path = path / name
                child_relative = name if relative_path == "." else f"{relative_path}/{name}"
                child = _open_child(descriptor, name, child_path, child_relative)
                if stat.S_ISDIR(child.stat.st_mode):
                    try:
                        scan_fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=descriptor)
                    except OSError as exc:
                        os.close(child._descriptor)
                        raise _unsafe(f"directory changed while opening {child_relative!r}", exc) from exc
                    if not _same_inode(child.stat, os.fstat(scan_fd)):
                        os.close(child._descriptor)
                        os.close(scan_fd)
                        raise _unsafe(f"directory changed while opening {child_relative!r}")
                    os.close(child._descriptor)
                    yield from self._walk_directory(scan_fd, child_path, child_relative)
                else:
                    try:
                        yield child
                    finally:
                        os.close(child._descriptor)
        finally:
            os.close(descriptor)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            os.close(self._descriptor)


def _open_child(parent_fd: int, name: str, path: Path, relative_path: str) -> Entry:
    try:
        descriptor = os.open(name, _PATH_FLAGS, dir_fd=parent_fd)
    except OSError as exc:
        raise _unsafe(f"cannot open {relative_path!r}", exc) from exc
    try:
        info = os.fstat(descriptor)
        try:
            current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except OSError as exc:
            raise _unsafe(f"entry changed while opening {relative_path!r}", exc) from exc
        if not _same_inode(info, current):
            raise _unsafe(f"entry changed while opening {relative_path!r}")
        return Entry(descriptor, path, relative_path, info)
    except BaseException:
        os.close(descriptor)
        raise


@contextmanager
def open_home(
    home_root: str | os.PathLike[str], username: str, *, create: bool = False
) -> Generator[Home, None, None]:
    """Open a trusted root's direct child without following any symlink."""
    name = _username(username)
    root_fd = _open_home_root(_root_parts(home_root))
    try:
        try:
            descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=root_fd)
        except FileNotFoundError:
            if not create:
                raise _unsafe("managed home does not exist") from None
            os.mkdir(name, 0o700, dir_fd=root_fd)
            descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=root_fd)
    except OSError as exc:
        raise _unsafe("managed home is not a non-symlink directory", exc) from exc
    finally:
        os.close(root_fd)
    home = Home(descriptor, Path(os.fspath(home_root)) / name)
    try:
        yield home
    finally:
        home.close()


def repair_home(
    home_root: str | os.PathLike[str], username: str, uid: int, gid: int
) -> None:
    """Create, repair, and verify one managed home."""
    with open_home(home_root, username, create=True) as home:
        for entry in home.walk():
            if (entry.stat.st_uid, entry.stat.st_gid) != (uid, gid):
                entry.chown(uid, gid)
        os.fchmod(home._descriptor, 0o2770)
    with open_home(home_root, username) as home:
        for entry in home.walk():
            if (entry.stat.st_uid, entry.stat.st_gid) != (uid, gid):
                raise IdentityStoreError(
                    f"ownership repair did not persist for {entry.relative_path!r}"
                )
        if stat.S_IMODE(os.fstat(home._descriptor).st_mode) != 0o2770:
            raise IdentityStoreError("managed home mode repair did not persist")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="safely repair one managed home")
    parser.add_argument("home_root")
    parser.add_argument("username")
    parser.add_argument("uid", type=int)
    parser.add_argument("gid", type=int)
    arguments = parser.parse_args(argv)
    repair_home(arguments.home_root, arguments.username, arguments.uid, arguments.gid)
    return 0


if __name__ == "__main__":  # pragma: no cover - invoked as a privileged helper
    raise SystemExit(main())
