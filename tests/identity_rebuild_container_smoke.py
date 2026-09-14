"""Two-container check: run seed, then verify with only the same /home volume."""

import json
import os
import pwd
import subprocess
import sys
from pathlib import Path

from open_terminal.utils.user_isolation import ensure_os_user

USERS = ("rebuild-a-full-id", "rebuild-b-full-id")
EXPECTED = Path("/home/.open-terminal/rebuild-fixture.json")


def main():
    assert os.geteuid() == 0
    assert os.environ.get("OT_IDENTITY_DISPOSABLE_TEST") == "1"
    if sys.argv[1:] == ["seed"]:
        expected = {}
        for user_id in USERS:
            record = ensure_os_user(user_id)
            target = Path("/home") / record.username / "persisted.txt"
            target.write_text(user_id)
            os.chown(target, record.uid, record.gid)
            expected[user_id] = [record.username, record.uid, record.gid]
        EXPECTED.write_text(json.dumps(expected))
        print("PASS: seeded identities and data in disposable /home volume")
    elif sys.argv[1:] == ["verify"]:
        expected = json.loads(EXPECTED.read_text())
        for username, _, _ in expected.values():
            try:
                pwd.getpwnam(username)
            except KeyError:
                continue
            raise AssertionError("OS account unexpectedly survived container destruction")
        for user_id in reversed(USERS):
            record = ensure_os_user(user_id)
            assert [record.username, record.uid, record.gid] == expected[user_id]
            target = Path("/home") / record.username / "persisted.txt"
            assert target.read_text() == user_id
            assert (target.stat().st_uid, target.stat().st_gid) == (record.uid, record.gid)
            subprocess.run(["runuser", "-u", record.username, "--", "test", "-w", str(target)], check=True)
        print("PASS: fresh container restores stable identities in reverse access order")
    else:
        raise SystemExit("expected seed or verify")


if __name__ == "__main__":
    main()
