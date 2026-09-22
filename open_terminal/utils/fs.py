"""Filesystem abstraction for multi-user mode.

Provides :class:`UserFS`, a unified interface for file operations.

All I/O uses native Python (``aiofiles`` / ``os``).  In multi-user mode
the server process is added to each provisioned user's group, and home
directories are ``chmod 2770`` (setgid + group rwx), so standard file
operations work without subprocess.

After each write operation a ``sudo chown`` call fixes file ownership
so that files belong to the provisioned user, not the server process.
"""

import asyncio
import os
import shutil
from typing import TextIO

import aiofiles
import aiofiles.os

from open_terminal.utils.service_processes import run_helper


MAX_READ_LINES = 2000
MAX_READ_BYTES = 50 * 1024
READ_CHUNK_CHARS = 8192
_LINE_ENDINGS = tuple("\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029")



class UserFS:
    """Filesystem operations scoped to an optional OS user.

    *username* is used for ownership fixups after writes (``None`` = stdlib).
    *home* is the user's home directory (default working directory).

    When *username* is set, path validation prevents access to other
    users' home directories (``/home/<other_user>/…``).
    """

    def __init__(self, username: str | None = None, home: str | None = None):
        self.username = username
        self.home = home or os.getcwd()

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def resolve_path(self, path: str, cwd: str | None = None) -> str:
        """Resolve *path* to an absolute path relative to the user's home.

        Absolute paths are normalised in place.  Relative paths are joined
        to *cwd* (if provided) or ``self.home`` so that they resolve against
        the session's working directory rather than the server process's
        ``os.getcwd()``.

        In multi-user mode, paths under ``/home/user`` (the server process's
        default home) are automatically rewritten to the provisioned user's
        home directory, since LLMs often hardcode that path.
        """
        if os.path.isabs(path):
            # Swap /home/user (and /home/usr, a common LLM hallucination)
            # → user's actual home when multi-user is active
            if self.username and self.home != "/home/user":
                for prefix in ("/home/user", "/home/usr"):
                    if path == prefix:
                        path = self.home
                        break
                    elif path.startswith(prefix + "/"):
                        path = self.home + path[len(prefix):]
                        break
            return os.path.normpath(path)
        return os.path.normpath(os.path.join(cwd or self.home, path))

    # ------------------------------------------------------------------
    # Path validation
    # ------------------------------------------------------------------

    def is_path_allowed(self, path: str) -> bool:
        """Return *False* if *path* is inside another user's home directory."""
        if not self.username:
            return True
        resolved = os.path.abspath(path)
        if not resolved.startswith("/home/"):
            return True
        parts = resolved.split("/")  # ['', 'home', '<user>', ...]
        if len(parts) >= 3:
            target_user_dir = parts[2]
            own_home_name = os.path.basename(self.home)
            if target_user_dir != own_home_name:
                return False
        return True

    def _check_path(self, path: str) -> None:
        """Reject paths inside another user's home directory."""
        if not self.is_path_allowed(path):
            raise PermissionError(
                f"Access denied: {os.path.abspath(path)} belongs to another user"
            )

    async def _chown(self, path: str) -> None:
        """Fix ownership of *path* to the provisioned user.

        Also sets group-write permission so the server process (which is
        in the provisioned user's group) can overwrite the file on
        subsequent writes.
        """
        if self.username:
            await asyncio.to_thread(
                run_helper,
                ["sudo", "chown", f"{self.username}:{self.username}", path],
                check=True, capture_output=True,
            )
            await asyncio.to_thread(
                run_helper,
                ["sudo", "chmod", "g+w", path],
                check=True, capture_output=True,
            )

    async def _ensure_parents(self, path: str) -> None:
        """Create parent directories for *path* with correct permissions.

        In multi-user mode, uses ``sudo -u`` to create directories as the
        provisioned user (so creation succeeds even inside ``755`` dirs
        made by ``run_command``), then sets ``2770`` on each directory in
        the chain so the server process has group-write access.

        In single-user mode, falls back to plain ``makedirs``.
        """
        if not self.username:
            await aiofiles.os.makedirs(path, exist_ok=True)
            return
        # Create as the provisioned user to bypass 755 restrictions.
        await asyncio.to_thread(
            run_helper,
            ["sudo", "-u", self.username, "mkdir", "-p", path],
            check=True, capture_output=True,
        )
        # Walk upward, setting 2770 so the server process (which is in the
        # user's group) can create files inside these directories.
        target = os.path.normpath(path)
        home = os.path.normpath(self.home)
        while target != home and target.startswith(home + "/"):
            await asyncio.to_thread(
                run_helper,
                ["sudo", "chmod", "2770", target],
                check=True, capture_output=True,
            )
            target = os.path.dirname(target)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def read(self, path: str) -> bytes:
        """Read raw bytes from *path*."""
        self._check_path(path)
        async with aiofiles.open(path, "rb") as f:
            return await f.read()

    async def read_text(self, path: str, encoding: str = "utf-8") -> str:
        """Read text from *path*."""
        self._check_path(path)
        async with aiofiles.open(path, "r", encoding=encoding, errors="strict") as f:
            return await f.read()

    async def read_text_page(
        self,
        path: str,
        *,
        start_line: int = 1,
        end_line: int | None = None,
        start_column: int = 1,
    ) -> dict:
        """Return a bounded page without buffering the entire file or line."""
        self._check_path(path)

        def read_page():
            with open(path, encoding="utf-8", errors="strict") as stream:
                return read_text_page(
                    stream,
                    start_line=start_line,
                    end_line=end_line,
                    start_column=start_column,
                )

        return await asyncio.to_thread(read_page)

    async def exists(self, path: str) -> bool:
        """Check if *path* exists."""
        self._check_path(path)
        return await aiofiles.os.path.exists(path)

    async def isfile(self, path: str) -> bool:
        """Check if *path* is a regular file."""
        self._check_path(path)
        return await aiofiles.os.path.isfile(path)

    async def isdir(self, path: str) -> bool:
        """Check if *path* is a directory."""
        self._check_path(path)
        return await aiofiles.os.path.isdir(path)

    async def stat(self, path: str) -> dict:
        """Return size, mtime, and type for *path*."""
        self._check_path(path)
        s = await aiofiles.os.stat(path)
        return {
            "size": s.st_size,
            "modified": s.st_mtime,
            "type": "directory" if os.path.isdir(path) else "file",
            "writable": self._is_writable_sync(path),
        }

    def _is_writable_sync(self, path: str) -> bool:
        if not self.is_path_allowed(path):
            return False
        try:
            if os.statvfs(path).f_flag & getattr(os, "ST_RDONLY", 0):
                return False
        except (AttributeError, OSError):
            pass
        try:
            return os.access(path, os.W_OK, effective_ids=True)
        except TypeError:
            return os.access(path, os.W_OK)

    async def is_writable(self, path: str) -> bool:
        """Return whether the current process can write to *path*."""
        return await asyncio.to_thread(self._is_writable_sync, path)

    async def listdir(self, path: str) -> list[dict]:
        """List directory contents with type, size, mtime, and writability."""
        self._check_path(path)
        def _list_sync():
            entries = []
            for name in sorted(os.listdir(path)):
                full = os.path.join(path, name)
                if not self.is_path_allowed(full):
                    continue
                try:
                    s = os.stat(full)
                    entries.append({
                        "name": name,
                        "type": "directory" if os.path.isdir(full) else "file",
                        "size": s.st_size,
                        "modified": s.st_mtime,
                        "writable": self._is_writable_sync(full),
                    })
                except OSError:
                    continue
            return entries
        return await asyncio.to_thread(_list_sync)

    async def walk(self, path: str) -> list[tuple[str, list[str], list[str]]]:
        """Walk directory tree. Returns list of (dirpath, dirnames, filenames).

        In multi-user mode, directories belonging to other users are pruned
        so their contents are never yielded.
        """
        self._check_path(path)
        def _walk_filtered():
            result = []
            for dirpath, dirnames, filenames in os.walk(path):
                # Prune directories belonging to other users (in-place
                # modification prevents os.walk from descending into them).
                dirnames[:] = [
                    d for d in dirnames
                    if self.is_path_allowed(os.path.join(dirpath, d))
                ]
                filenames = [
                    f for f in filenames
                    if self.is_path_allowed(os.path.join(dirpath, f))
                ]
                result.append((dirpath, dirnames, filenames))
            return result
        return await asyncio.to_thread(_walk_filtered)

    # ------------------------------------------------------------------
    # Write operations (native Python + chown for correct ownership)
    # ------------------------------------------------------------------

    async def write(self, path: str, content: str, encoding: str = "utf-8") -> None:
        """Write text *content* to *path*, creating parent dirs."""
        self._check_path(path)
        parent = os.path.dirname(path)
        if parent:
            await self._ensure_parents(parent)
        async with aiofiles.open(path, "w", encoding=encoding) as f:
            await f.write(content)
        await self._chown(path)

    async def write_bytes(self, path: str, data: bytes) -> None:
        """Write raw *data* to *path*, creating parent dirs."""
        self._check_path(path)
        parent = os.path.dirname(path)
        if parent:
            await self._ensure_parents(parent)
        async with aiofiles.open(path, "wb") as f:
            await f.write(data)
        await self._chown(path)

    async def mkdir(self, path: str) -> None:
        """Create directory *path* and parents."""
        self._check_path(path)
        await self._ensure_parents(path)

    async def remove(self, path: str) -> None:
        """Remove *path* (file or directory)."""
        self._check_path(path)
        if os.path.isdir(path):
            await asyncio.to_thread(shutil.rmtree, path)
        else:
            await aiofiles.os.remove(path)

    async def move(self, source: str, destination: str) -> None:
        """Move *source* to *destination*."""
        self._check_path(source)
        self._check_path(destination)
        await asyncio.to_thread(shutil.move, source, destination)
        await self._chown(destination)


def read_text_page(
    stream: TextIO,
    *,
    start_line: int = 1,
    end_line: int | None = None,
    start_column: int = 1,
    max_lines: int = MAX_READ_LINES,
    max_bytes: int = MAX_READ_BYTES,
) -> dict:
    """Scan with bounded buffers, retaining at most one page of UTF-8 text.

    Columns count Unicode characters, including the newline. Scanning past the
    page keeps total_lines accurate and validates UTF-8 without loading the file.
    Lines follow str.splitlines() semantics. Streams must normalize CRLF/CR
    to LF, as the ordinary text reader does.
    """
    if max_lines < 1 or max_bytes < 4:
        raise ValueError("Page limits must allow a line and any UTF-8 character")

    last_line = start_line + max_lines - 1
    if end_line is not None:
        last_line = min(end_line, last_line)
    line = column = 1
    total_lines = 0
    pieces: list[str] = []
    output_bytes = 0
    output_end_line = None
    next_position = None
    reason = None
    start_found = start_column == 1

    while buffer := stream.readline(READ_CHUNK_CHARS):
        for chunk in buffer.splitlines(keepends=True):
            total_lines = line
            if line == start_line and column <= start_column < column + len(chunk):
                start_found = True
            if line >= start_line and next_position is None:
                skip = max(0, start_column - column) if line == start_line else 0
                selected = chunk[skip:]
                if selected:
                    if line > last_line:
                        next_position = (line, column + skip)
                        reason = "lines"
                    else:
                        encoded = selected.encode("utf-8")
                        remaining = max_bytes - output_bytes
                        prefix = encoded[:remaining].decode("utf-8", errors="ignore")
                        if prefix:
                            pieces.append(prefix)
                            output_bytes += len(prefix.encode("utf-8"))
                            output_end_line = line
                        if len(encoded) > remaining:
                            next_position = (line, column + skip + len(prefix))
                            reason = "bytes"
            if chunk.endswith(_LINE_ENDINGS):
                line += 1
                column = 1
            else:
                column += len(chunk)

    if not start_found:
        raise ValueError("start_column is beyond the requested line")

    return {
        "total_lines": total_lines,
        "content": "".join(pieces),
        "start_line": start_line,
        "start_column": start_column,
        "end_line": output_end_line,
        "returned_bytes": output_bytes,
        "truncated": next_position is not None,
        "truncation_reason": reason,
        "next_start_line": next_position[0] if next_position else None,
        "next_start_column": next_position[1] if next_position else None,
    }
