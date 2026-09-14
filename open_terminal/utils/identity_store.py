"""Persistent, private registry of upstream users and Linux identities."""
from __future__ import annotations

import argparse
import fcntl
import grp
import json
import logging
import os
import pwd
import stat
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from open_terminal import config

log = logging.getLogger(__name__)
REGISTRY_VERSION = 1
STATES = ("allocated", "provisioning", "ready")


class IdentityStoreError(RuntimeError):
    """The identity registry cannot safely answer a request."""


class IdentityConflictError(IdentityStoreError):
    """A requested identity is already owned by somebody else."""


@dataclass
class IdentityRecord:
    user_id: str
    username: str
    uid: int
    gid: int
    state: str
    created_at: str
    updated_at: str


def _default_map_path() -> Path:
    value = os.environ.get("OPEN_TERMINAL_IDENTITY_MAP", config.get("identity_map", ""))
    return Path(value) if value else Path("/home/.open-terminal/identity-map.json")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _trusted_owner(uid: int) -> bool:
    if uid in (0, os.geteuid()):
        return True
    # The standard image grants its service account passwordless sudo. Its
    # persisted /home mount can remain owned by it when the server runs as root.
    try:
        service = pwd.getpwnam("user")
    except KeyError:
        return False
    return uid == service.pw_uid and service.pw_dir == "/home/user"


def _validate_trusted_parent(path: Path, *, allow_missing=False) -> bool:
    """Verify ancestors without creating directories or following symlinks."""
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            if allow_missing:
                return False
            raise IdentityStoreError("registry parent does not exist: " + str(current))
        except OSError as exc:
            raise IdentityStoreError("cannot inspect registry parent: " + str(current)) from exc
        is_tmp = current == Path("/tmp") and bool(info.st_mode & stat.S_ISVTX)
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISDIR(info.st_mode)
            or (not _trusted_owner(info.st_uid) and not is_tmp)
            or (info.st_mode & 0o022 and not is_tmp)
        ):
            raise IdentityStoreError("registry parent has unsafe ownership or permissions: " + str(current))
    return True


def _private_file(path: Path, label: str):
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise IdentityStoreError("cannot inspect " + label + ": " + str(path)) from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.geteuid()
        or info.st_mode & 0o077
    ):
        raise IdentityStoreError(label + " must be a private regular file: " + str(path))
    return info


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise IdentityStoreError("identity registry contains duplicate JSON key: " + repr(key))
        result[key] = value
    return result


class IdentityStore:
    """Atomic JSON registry with one reentrant provisioning operation gate."""

    def __init__(self, path=None, home_root=None):
        self.path = Path(path) if path is not None else _default_map_path()
        self.home_root = Path(home_root) if home_root is not None else self.path.parent.parent
        self.operation_lock_path = self.path.parent / ".identity-map.operation.lock"
        self._thread_lock = threading.RLock()
        self._operation_depth = 0

    def _ensure_directory(self):
        directory = self.path.parent
        _validate_trusted_parent(directory.parent)
        try:
            os.mkdir(directory, 0o700)
        except FileExistsError:
            pass
        try:
            info = os.lstat(directory)
        except OSError as exc:
            raise IdentityStoreError("cannot inspect registry directory: " + str(directory)) from exc
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o700
        ):
            raise IdentityStoreError("registry directory must be owned by this user and mode 0700: " + str(directory))

    @contextmanager
    def operation(self):
        """Hold a complete provisioning transaction across nested store calls."""
        with self._thread_lock:
            if self._operation_depth:
                self._operation_depth += 1
                try:
                    yield
                finally:
                    self._operation_depth -= 1
                return
            self._ensure_directory()
            flags = os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(self.operation_lock_path, flags, 0o600)
            try:
                info = os.fstat(fd)
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_uid != os.geteuid()
                    or info.st_mode & 0o077
                ):
                    raise IdentityStoreError("registry lock must be a private regular file")
                fcntl.flock(fd, fcntl.LOCK_EX)
                self._operation_depth = 1
                try:
                    yield
                finally:
                    self._operation_depth = 0
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    @staticmethod
    def _empty_document():
        return {"version": REGISTRY_VERSION, "next_uid": 0, "next_gid": 0, "users": {}}

    @staticmethod
    def _validate_user_id(user_id):
        if not isinstance(user_id, str) or not user_id:
            raise IdentityStoreError("user_id must be a non-empty string")

    @staticmethod
    def _validate_username(username):
        if (
            not isinstance(username, str)
            or not username
            or username in (".", "..")
            or any(char in username for char in ("/", "\\", "\x00"))
        ):
            raise IdentityStoreError("username must be a safe non-empty string")

    @staticmethod
    def _validate_identifier(value, name):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 2**31 - 1:
            raise IdentityStoreError(name + " must be an integer in the Linux ID range")

    @classmethod
    def _validate_document(cls, document):
        if not isinstance(document, dict) or document.get("version") != REGISTRY_VERSION:
            raise IdentityStoreError("unsupported or invalid identity registry")
        users = document.setdefault("users", {})
        if not isinstance(users, dict):
            raise IdentityStoreError("identity registry users must be an object")
        usernames, uids, gids = set(), set(), set()
        for user_id, record in users.items():
            cls._validate_user_id(user_id)
            if not isinstance(record, dict):
                raise IdentityStoreError("identity record must be an object")
            cls._validate_username(record.get("username"))
            cls._validate_identifier(record.get("uid"), "uid")
            cls._validate_identifier(record.get("gid"), "gid")
            if record.get("state", "allocated") not in STATES:
                raise IdentityStoreError("invalid identity state: " + repr(record.get("state")))
            username, uid, gid = record["username"], record["uid"], record["gid"]
            if username in usernames or uid in uids or gid in gids:
                raise IdentityConflictError("duplicate username, UID, or GID in identity registry")
            usernames.add(username)
            uids.add(uid)
            gids.add(gid)
        for counter in ("next_uid", "next_gid"):
            value = document.get(counter, 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise IdentityStoreError(counter + " must be a non-negative integer")

    def _load(self):
        if not _validate_trusted_parent(self.path.parent, allow_missing=True):
            return self._empty_document()
        if _private_file(self.path, "identity registry") is None:
            return self._empty_document()
        try:
            with open(self.path, encoding="utf-8") as handle:
                document = json.load(handle, object_pairs_hook=_json_object)
        except json.JSONDecodeError as exc:
            raise IdentityStoreError("identity registry is corrupt: " + str(self.path)) from exc
        self._validate_document(document)
        return document

    def _write(self, document):
        self._ensure_directory()
        self._validate_document(document)
        _private_file(self.path, "identity registry")
        fd, temporary = tempfile.mkstemp(
            dir=self.path.parent, prefix=".identity-map.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(document, indent=2, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, self.path)
            directory_fd = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    @staticmethod
    def _record(user_id, record):
        return IdentityRecord(
            user_id,
            record["username"],
            record["uid"],
            record["gid"],
            record.get("state", "allocated"),
            record.get("created_at", ""),
            record.get("updated_at", ""),
        )

    def get(self, user_id):
        record = self._load()["users"].get(user_id)
        return self._record(user_id, record) if record else None

    def all_records(self):
        return {
            user_id: self._record(user_id, record)
            for user_id, record in self._load()["users"].items()
        }

    def usernames(self):
        return {record["username"]: user_id for user_id, record in self._load()["users"].items()}

    def _occupied(self, document):
        try:
            uids = {account.pw_uid for account in pwd.getpwall()}
            gids = {group.gr_gid for group in grp.getgrall()}
            entries = list(os.scandir(self.home_root))
        except (OSError, AttributeError) as exc:
            raise IdentityStoreError("cannot enumerate occupied identities") from exc
        for entry in entries:
            if entry.name.startswith("."):
                continue
            try:
                info = os.stat(entry.path, follow_symlinks=False)
            except OSError as exc:
                raise IdentityStoreError("cannot inspect home directory: " + entry.path) from exc
            if stat.S_ISDIR(info.st_mode):
                uids.add(info.st_uid)
                gids.add(info.st_gid)
        uids.update(record["uid"] for record in document["users"].values())
        gids.update(record["gid"] for record in document["users"].values())
        return uids, gids

    def _other_homes(self, identifier, username, field):
        try:
            entries = list(os.scandir(self.home_root))
        except OSError as exc:
            raise IdentityStoreError("cannot scan home directories") from exc
        found = set()
        for entry in entries:
            if entry.name.startswith(".") or entry.name == username:
                continue
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise IdentityStoreError("cannot inspect home directory: " + entry.path) from exc
            if stat.S_ISDIR(info.st_mode) and getattr(info, "st_" + field) == identifier:
                found.add(entry.name)
        return found

    @staticmethod
    def _owner(identifier, group=False):
        try:
            if group:
                return grp.getgrgid(identifier).gr_name
            return pwd.getpwuid(identifier).pw_name
        except KeyError:
            return None

    @staticmethod
    def _uid_floor():
        value = os.environ.get("OPEN_TERMINAL_UID_MIN", config.get("uid_min"))
        if value is None:
            return 1000
        try:
            value = int(value)
        except ValueError as exc:
            raise IdentityStoreError("OPEN_TERMINAL_UID_MIN must be an integer") from exc
        if not 1 <= value <= 2**31 - 1:
            raise IdentityStoreError("OPEN_TERMINAL_UID_MIN is out of range")
        return value

    @classmethod
    def _coerce(cls, value, name):
        try:
            value = int(value)
        except (TypeError, ValueError) as exc:
            raise IdentityStoreError(name + " must be an integer") from exc
        cls._validate_identifier(value, name)
        return value

    def _validate_adoption(self, document, username, uid, gid, uids, gids):
        records = list(document["users"].values())
        if uid in {record["uid"] for record in records} or gid in {
            record["gid"] for record in records
        }:
            raise IdentityConflictError("adopted UID or GID is already in the registry")
        if uid in uids and (
            self._owner(uid) not in (None, username)
            or self._other_homes(uid, username, "uid")
        ):
            raise IdentityConflictError("cannot adopt UID " + str(uid))
        if gid in gids and (
            self._owner(gid, group=True) not in (None, username)
            or self._other_homes(gid, username, "gid")
        ):
            raise IdentityConflictError("cannot adopt GID " + str(gid))

    def allocate(self, user_id, username, *, adopt_uid=None, adopt_gid=None):
        self._validate_user_id(user_id)
        self._validate_username(username)
        if (adopt_uid is None) != (adopt_gid is None):
            raise IdentityStoreError("adopt_uid and adopt_gid must be provided together")
        if adopt_uid is not None:
            adopt_uid = self._coerce(adopt_uid, "adopt_uid")
            adopt_gid = self._coerce(adopt_gid, "adopt_gid")
        with self.operation():
            document = self._load()
            existing = document["users"].get(user_id)
            if existing:
                if existing["username"] != username or (
                    adopt_uid is not None
                    and (existing["uid"], existing["gid"]) != (adopt_uid, adopt_gid)
                ):
                    raise IdentityConflictError("user_id is already bound to another identity")
                return self._record(user_id, existing)
            if any(record["username"] == username for record in document["users"].values()):
                raise IdentityConflictError("username is already bound to another user")
            uids, gids = self._occupied(document)
            if adopt_uid is not None:
                uid, gid = adopt_uid, adopt_gid
                self._validate_adoption(document, username, uid, gid, uids, gids)
            else:
                candidate = max(
                    document.get("next_uid", 0),
                    document.get("next_gid", 0),
                    max(uids, default=0),
                    max(gids, default=0),
                    self._uid_floor(),
                )
                while candidate in uids or candidate in gids:
                    candidate += 1
                uid = gid = candidate
                document["next_uid"] = candidate
                document["next_gid"] = candidate
            now = _now()
            record = {
                "username": username,
                "uid": uid,
                "gid": gid,
                "state": "allocated",
                "created_at": now,
                "updated_at": now,
            }
            document["users"][user_id] = record
            self._write(document)
            log.info(
                "identity allocated: user_id=%s username=%s uid=%d gid=%d",
                user_id,
                username,
                uid,
                gid,
            )
            return self._record(user_id, record)

    def mark(self, user_id, state):
        self._validate_user_id(user_id)
        if state not in STATES:
            raise IdentityStoreError("invalid identity state: " + repr(state))
        with self.operation():
            document = self._load()
            record = document["users"].get(user_id)
            if record is None:
                raise IdentityStoreError("unknown user_id: " + repr(user_id))
            record["state"], record["updated_at"] = state, _now()
            self._write(document)
            return self._record(user_id, record)


def initialize_directory(path, owner_uid, owner_gid):
    """Create a private registry directory; never repair an existing target."""
    target = Path(path)
    if not target.is_absolute() or target.name in ("", ".", ".."):
        raise IdentityStoreError("registry directory must be an absolute leaf path")
    try:
        owner_uid, owner_gid = int(owner_uid), int(owner_gid)
    except (TypeError, ValueError) as exc:
        raise IdentityStoreError("registry owner IDs must be integers") from exc
    if owner_uid < 0 or owner_gid < 0:
        raise IdentityStoreError("registry owner IDs must be non-negative")
    _validate_trusted_parent(target.parent)
    try:
        os.mkdir(target, 0o700)
    except FileExistsError:
        info = os.lstat(target)
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISDIR(info.st_mode)
            or (info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode))
            != (owner_uid, owner_gid, 0o700)
        ):
            raise IdentityStoreError("existing registry directory has unsafe ownership or mode")
        return
    fd = os.open(
        target,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fchown(fd, owner_uid, owner_gid)
        os.fchmod(fd, 0o700)
    finally:
        os.close(fd)


def _main(argv=None):
    parser = argparse.ArgumentParser(description="Initialize an identity registry directory")
    parser.add_argument("--initialize", metavar="PATH", required=True)
    parser.add_argument("uid", type=int)
    parser.add_argument("gid", type=int)
    args = parser.parse_args(argv)
    initialize_directory(args.initialize, args.uid, args.gid)


_store = None
_store_lock = threading.Lock()

def get_identity_store():
    global _store
    with _store_lock:
        if _store is None: _store = IdentityStore()
        return _store

def set_identity_store(store):
    global _store
    with _store_lock:
        _store = store


if __name__ == "__main__":
    _main()
