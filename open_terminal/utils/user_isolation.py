"""Per-user OS account provisioning for multi-user mode.

When OPEN_TERMINAL_MULTI_USER=true, each distinct X-User-Id is mapped to a
dedicated Linux user account.  Commands and file operations then run as that
OS user, and chmod 2770 on the home directory provides kernel-enforced
isolation between users.

Identity stability contract:
  * The full upstream user identifier is the identity anchor.
  * A persistent registry under /home binds it to a username, UID and GID.
  * OS accounts are always created with the recorded identifiers, so
    container recreation cannot reshuffle them.
  * Provisioning is a persisted state machine (allocated, provisioning,
    ready).  A user is only returned to callers after the account, home
    ownership and permissions have been prepared and verified.
  * Recursive ownership repair is scoped to /home/<username> only, never to
    the whole /home tree.
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import pwd
import re
import shutil
import subprocess
import stat
import sys

from open_terminal.env import USER_PREFIX
from open_terminal.utils.identity_paths import open_home, repair_home
from open_terminal.utils.identity_store import (
    IdentityConflictError,
    IdentityRecord,
    IdentityStore,
    IdentityStoreError,
    get_identity_store,
)

log = logging.getLogger(__name__)

MAX_USER_ID_LENGTH = 256
# Deep verification is needed once per service process and after recovery.
# The root directory fingerprint also invalidates it on mode/owner changes.
_VERIFIED_HOMES: dict[str, tuple] = {}


class ProvisioningError(RuntimeError):
    """Transient failure while preparing an OS account; safe to retry."""


def _run_privileged(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run a command with appropriate privilege escalation.

    When the process is already running as root (UID 0), the command is
    executed directly.  Otherwise sudo is prepended.
    """
    if os.getuid() == 0:
        return subprocess.run(cmd, check=True, capture_output=True)
    return subprocess.run(["sudo", *cmd], check=True, capture_output=True)


def check_environment() -> None:
    """Validate that the host supports multi-user mode.

    Raises RuntimeError at startup when the platform is not Linux or the
    required privilege escalation tools are not available.
    """
    if platform.system() != "Linux":
        raise RuntimeError(
            "OPEN_TERMINAL_MULTI_USER requires Linux "
            f"(current platform: {platform.system()})"
        )
    if shutil.which("useradd") is None:
        raise RuntimeError(
            "OPEN_TERMINAL_MULTI_USER requires useradd to be installed"
        )
    if os.getuid() != 0 and shutil.which("sudo") is None:
        raise RuntimeError(
            "OPEN_TERMINAL_MULTI_USER requires either running as root "
            "or sudo to be installed. Use the standard image, run with "
            "user: '0:0', or use Terminals for container-per-user isolation."
        )
    store = get_identity_store()
    try:
        if os.geteuid() != 0:
            _run_privileged([
                sys.executable, "-m", "open_terminal.utils.identity_store",
                "--initialize", str(store.path.parent), str(os.geteuid()), str(os.getegid()),
            ])
        store._ensure_directory()
    except OSError as exc:
        raise RuntimeError(
            f"OPEN_TERMINAL_MULTI_USER cannot write the identity registry at {store.path}: {exc}"
        ) from exc


def sanitize_username(user_id: str) -> str:
    """Convert an arbitrary user ID into a valid Linux username.

    Uses the first 8 lowercase alphanumeric characters of the user ID,
    optionally prefixed by OPEN_TERMINAL_USER_PREFIX.  Prepends u only when
    the result starts with a digit (Linux usernames must begin with a letter
    or underscore).  Falls back to a short hash when the ID contains fewer
    than 4 usable characters.
    """
    cleaned = re.sub(r"[^a-z0-9]", "", user_id.lower())
    if len(cleaned) >= 4:
        name = cleaned[:8]
    else:
        name = hashlib.sha256(user_id.encode()).hexdigest()[:8]
    name = f"{USER_PREFIX}{name}"
    if name[0].isdigit():
        name = f"u{name}"
    return name


def _collision_suffix(user_id: str) -> str:
    return hashlib.sha256(user_id.encode()).hexdigest()[:4]


def _select_username(user_id: str, store: IdentityStore) -> tuple[str, pwd.struct_passwd | None]:
    """Pick a unique username for user_id.

    The base name keeps compatibility with directories created by earlier
    versions.  When the base name is already bound to a different upstream
    identity, a deterministic hash suffix disambiguates the new user instead
    of silently sharing an account.
    """
    base = sanitize_username(user_id)
    bound = store.usernames()
    if bound.get(base) == user_id:
        try:
            return base, pwd.getpwnam(base)
        except KeyError:
            return base, None
    if base in bound:
        candidate = base + _collision_suffix(user_id)
        while candidate in bound or _account_exists(candidate):
            candidate = base + hashlib.sha256(candidate.encode()).hexdigest()[:4]
        return candidate, None
    try:
        account = pwd.getpwnam(base)
    except KeyError:
        return base, None
    if (
        account.pw_dir == str(store.home_root / base)
        and account.pw_uid not in (0, os.geteuid())
        and account.pw_name != "user"
    ):
        return base, account
    candidate = base + _collision_suffix(user_id)
    while True:
        try:
            pwd.getpwnam(candidate)
        except KeyError:
            if candidate not in bound:
                return candidate, None
        candidate = base + hashlib.sha256(candidate.encode()).hexdigest()[:4]


def _ensure_group(record: IdentityRecord) -> None:
    import grp

    try:
        group = grp.getgrnam(record.username)
        if group.gr_gid != record.gid:
            raise IdentityConflictError(
                f"group {record.username!r} exists with GID {group.gr_gid}, "
                f"expected {record.gid}"
            )
        return
    except KeyError:
        pass
    try:
        group = grp.getgrgid(record.gid)
        raise IdentityConflictError(
            f"GID {record.gid} exists as group {group.gr_name!r}, "
            f"expected {record.username!r}"
        )
    except KeyError:
        pass
    try:
        _run_privileged(["groupadd", "-g", str(record.gid), record.username])
    except subprocess.CalledProcessError as exc:
        raise ProvisioningError(
            f"groupadd failed for {record.username!r} (gid {record.gid}): "
            f"exit {exc.returncode}"
        ) from exc


def _account_exists(username: str) -> bool:
    try:
        pwd.getpwnam(username)
        return True
    except KeyError:
        return False


def _ensure_account(record: IdentityRecord, *, home_root=None) -> pwd.struct_passwd:
    home_root = home_root if home_root is not None else get_identity_store().home_root
    expected_home = str(home_root / record.username)
    # Pin and validate the managed directory before invoking account tools.
    # useradd must not perform its own path-based skeleton copy or chown.
    if os.geteuid() == 0:
        with open_home(home_root, record.username, create=True):
            pass
    try:
        account = pwd.getpwnam(record.username)
    except KeyError:
        account = None
    if account is not None:
        if account.pw_uid != record.uid or account.pw_gid != record.gid:
            raise IdentityConflictError(
                f"account {record.username!r} exists with UID/GID "
                f"{account.pw_uid}/{account.pw_gid}, registry requires "
                f"{record.uid}/{record.gid}; refusing to proceed"
            )
        if account.pw_dir != expected_home:
            raise IdentityConflictError(
                f"account {record.username!r} has an unexpected home directory"
            )
        _ensure_group(record)
        return account
    try:
        foreign = pwd.getpwuid(record.uid)
        raise IdentityConflictError(
            f"UID {record.uid} is already used by account {foreign.pw_name!r}"
        )
    except KeyError:
        pass
    _ensure_group(record)
    try:
        _run_privileged([
            "useradd", "-M", "-d", expected_home, "-u", str(record.uid),
            "-g", str(record.gid), "-s", "/bin/bash", record.username,
        ])
    except subprocess.CalledProcessError as exc:
        raise ProvisioningError(
            f"useradd failed for {record.username!r}: exit {exc.returncode}"
        ) from exc
    try:
        return pwd.getpwnam(record.username)
    except KeyError as exc:
        raise ProvisioningError(
            f"account {record.username!r} missing after useradd"
        ) from exc


def _prepare_permissions(record: IdentityRecord, home: str) -> None:
    """Repair only entries pinned beneath this identity's managed home."""
    store = get_identity_store()
    if home != str(store.home_root / record.username):
        raise IdentityConflictError("refusing permission repair outside managed home")
    try:
        if os.geteuid() != 0:
            _run_privileged([
                sys.executable, "-m", "open_terminal.utils.identity_paths",
                str(store.home_root), record.username, str(record.uid), str(record.gid),
            ])
            return
        repair_home(store.home_root, record.username, record.uid, record.gid)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ProvisioningError(
            f"permission repair failed for {record.username!r}: {exc}"
        ) from exc


def _prepare_group_memberships(record: IdentityRecord) -> None:
    """Give the service user group access and mirror Docker socket access."""
    if os.getuid() != 0:
        server_user = os.getenv("USER", "user")
        try:
            _run_privileged(["usermod", "-aG", record.username, server_user])
        except subprocess.CalledProcessError as exc:
            raise ProvisioningError(
                f"could not add service user to group {record.username!r}: "
                f"exit {exc.returncode}"
            ) from exc
        _refresh_supplementary_groups(server_user)
    docker_socket = "/var/run/docker.sock"
    if os.path.exists(docker_socket):
        import grp

        sock_gid = os.stat(docker_socket).st_gid
        try:
            sock_group = grp.getgrgid(sock_gid).gr_name
            _run_privileged(["usermod", "-aG", sock_group, record.username])
            log.info(
                "added user=%s to docker socket group=%s",
                record.username,
                sock_group,
            )
        except (KeyError, subprocess.CalledProcessError) as exc:
            log.warning(
                "could not add user=%s to docker socket group: %s",
                record.username,
                exc,
            )


def _refresh_supplementary_groups(server_user: str) -> None:
    import grp

    try:
        pw = pwd.getpwnam(server_user)
        group_ids = sorted(
            {g.gr_gid for g in grp.getgrall() if server_user in g.gr_mem}
            | {pw.pw_gid}
        )
        os.setgroups(group_ids)
    except (KeyError, PermissionError) as exc:
        log.warning(
            "could not refresh supplementary groups for %s (%s); "
            "a restart applies group membership",
            server_user,
            exc,
        )
    except OSError as exc:
        log.warning(
            "could not refresh supplementary groups for %s (%s); "
            "a restart applies group membership",
            server_user,
            exc,
        )


def _verify_ready(record: IdentityRecord, home: str) -> None:
    try:
        account = pwd.getpwnam(record.username)
    except KeyError as exc:
        raise ProvisioningError(
            f"account {record.username!r} disappeared during provisioning"
        ) from exc
    if account.pw_uid != record.uid or account.pw_gid != record.gid:
        raise IdentityConflictError(
            f"account {record.username!r} changed identity during provisioning"
        )
    try:
        info = os.stat(home, follow_symlinks=False)
    except OSError as exc:
        raise ProvisioningError(f"home directory {home} is not accessible") from exc
    if info.st_uid != record.uid or info.st_gid != record.gid:
        raise ProvisioningError(
            f"home directory {home} has owner {info.st_uid}:{info.st_gid}, "
            f"expected {record.uid}:{record.gid}"
        )
    if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o2770:
        raise ProvisioningError(f"home directory {home} lacks mode 2770")


def ensure_os_user(user_id: str) -> IdentityRecord:
    """Provision or recover the OS identity for an upstream user id.

    Returns the registry record only after the account exists with the
    recorded UID/GID and the home directory ownership has been prepared.
    Failures keep the persisted state below ready so the next call retries.
    """
    if not isinstance(user_id, str) or not user_id.strip():
        raise IdentityStoreError("X-User-Id must be a non-empty string")
    if len(user_id) > MAX_USER_ID_LENGTH:
        raise IdentityStoreError("X-User-Id exceeds 256 characters")
    store = get_identity_store()
    with store.operation():
        record = store.get(user_id)
        if record is None:
            username, existing = _select_username(user_id, store)
            record = store.allocate(
                user_id, username,
                adopt_uid=existing.pw_uid if existing else None,
                adopt_gid=existing.pw_gid if existing else None,
            )
        recreated = not _account_exists(record.username)
        if recreated:
            record = store.mark(user_id, "provisioning")
        account = _ensure_account(record, home_root=store.home_root)
        home = account.pw_dir
        try:
            _verify_ready(record, home)
            info = os.stat(home, follow_symlinks=False)
            fingerprint = (record.uid, record.gid, info.st_dev, info.st_ino, info.st_ctime_ns)
            if record.state == "ready" and _VERIFIED_HOMES.get(home) == fingerprint:
                return record
            if record.state == "ready" and os.geteuid() == 0:
                with open_home(store.home_root, record.username) as tree:
                    verified = all(
                        (entry.stat.st_uid, entry.stat.st_gid) == (record.uid, record.gid)
                        for entry in tree.walk()
                    )
                if verified:
                    _prepare_group_memberships(record)
                    _VERIFIED_HOMES[home] = fingerprint
                    return record
        except (OSError, ProvisioningError):
            pass
        store.mark(user_id, "provisioning")
        _prepare_permissions(record, home)
        _prepare_group_memberships(record)
        _verify_ready(record, home)
        info = os.stat(home, follow_symlinks=False)
        result = store.mark(user_id, "ready")
        _VERIFIED_HOMES[home] = (
            record.uid, record.gid, info.st_dev, info.st_ino, info.st_ctime_ns,
        )
        return result


def resolve_user(user_id: str) -> tuple[str, str]:
    """Map an upstream user ID to an OS user, provisioning if needed.

    Returns (username, home_dir).  The persisted registry is the source of
    truth; a successful result is only produced after provisioning reached
    the ready state.
    """
    record = ensure_os_user(user_id)
    account = pwd.getpwnam(record.username)
    return record.username, account.pw_dir
