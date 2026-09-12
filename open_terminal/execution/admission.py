"""Bounded, round-robin admission; waiting requests own no processes or cgroups."""

import concurrent.futures
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from fastapi import HTTPException


def busy(message):
    return HTTPException(429, message, headers={"Retry-After": "1"})


@dataclass(eq=False)
class Ticket:
    owner: str
    key: str
    username: str
    argv: list[str]
    cwd: str | None
    env: dict | None
    created: float = field(default_factory=time.monotonic)
    created_at: float = field(default_factory=time.time)
    ready: concurrent.futures.Future = field(default_factory=concurrent.futures.Future)
    pending: bool = False


class FairQueue:
    def __init__(self, manager):
        self.manager = manager
        self.condition = threading.Condition(manager._lock)
        self.owners = deque()
        self.waiting = {}
        self.count = 0
        self.last_owner = None
        self.worker = None
        self.closed = False

    def submit(self, ticket):
        with self.condition:
            if self.closed:
                raise HTTPException(503, "Execution manager is stopping")
            if not self.count and self.manager._has_capacity(ticket.key):
                self._grant(ticket)
                return ticket
            policy = self.manager.policy
            if (
                self.count >= policy.max_queue
                or len(self.waiting.get(ticket.key, ())) >= policy.max_user_queue
            ):
                raise busy("Compute waiting queue is full. Retry later.")
            if ticket.key not in self.waiting:
                self.waiting[ticket.key] = deque()
                # A new owner gets a turn before the most recently served owner.
                if self.owners and self.owners[-1] == self.last_owner:
                    self.owners.insert(len(self.owners) - 1, ticket.key)
                else:
                    self.owners.append(ticket.key)
            self.waiting[ticket.key].append(ticket)
            ticket.pending = True
            self.count += 1
            if self.worker is None:
                self.worker = threading.Thread(
                    target=self._run, name="execution-admission", daemon=True
                )
                self.worker.start()
            self.condition.notify_all()
            return ticket

    def _remove(self, ticket):
        queue = self.waiting[ticket.key]
        queue.remove(ticket)
        ticket.pending = False
        self.count -= 1
        if not queue:
            del self.waiting[ticket.key]
            self.owners.remove(ticket.key)

    def _grant(self, ticket):
        try:
            launch = self.manager.prepare(
                ticket.owner,
                ticket.username,
                ticket.argv,
                cwd=ticket.cwd,
                env=ticket.env,
                _queued=True,
            )
            launch.queued_at = ticket.created_at
            launch.queue_wait_seconds = max(0, time.monotonic() - ticket.created)
        except Exception as error:  # noqa: BLE001 - delivered to the awaiting caller
            ticket.ready.set_exception(error)
        else:
            self.last_owner = ticket.key
            ticket.ready.set_result(launch)

    def cancel(self, ticket):
        """Return a concurrently granted launch to clean outside the queue lock."""
        with self.condition:
            if ticket.pending:
                self._remove(ticket)
                ticket.ready.cancel()
                self.condition.notify_all()
                return None
            if (
                ticket.ready.done()
                and not ticket.ready.cancelled()
                and ticket.ready.exception() is None
            ):
                return ticket.ready.result()
            return None

    def _eligible(self, key):
        if not self.manager._has_capacity(key):
            return False
        if key in self.manager._owners:
            new_user_waiting = any(
                owner not in self.manager._owners for owner in self.waiting
            )
            another_user_fits = (
                self.manager.policy.user * (len(self.manager._owners) + 1)
            ).fits_within(self.manager.policy.compute)
            if new_user_waiting and not another_user_fits:
                # Let existing reservations drain rather than letting a busy
                # owner renew them indefinitely ahead of a new user.
                return False
        return True

    def _dispatch(self):
        now = time.monotonic()
        for queue in list(self.waiting.values()):
            for ticket in list(queue):
                if now - ticket.created >= self.manager.policy.queue_timeout:
                    self._remove(ticket)
                    ticket.ready.set_exception(
                        busy("Compute queue wait timed out. Retry later.")
                    )
        while self.count:
            chosen = None
            for _ in range(len(self.owners)):
                key = self.owners[0]
                self.owners.rotate(-1)
                if self._eligible(key):
                    chosen = self.waiting[key][0]
                    break
            if chosen is None:
                break
            self._remove(chosen)
            if time.monotonic() - chosen.created >= self.manager.policy.queue_timeout:
                chosen.ready.set_exception(
                    busy("Compute queue wait timed out. Retry later.")
                )
            else:
                self._grant(chosen)

    def _run(self):
        with self.condition:
            while not self.closed:
                self._dispatch()
                delay = None
                if self.count:
                    delay = max(
                        0.001,
                        min(q[0].created for q in self.waiting.values())
                        + self.manager.policy.queue_timeout
                        - time.monotonic(),
                    )
                self.condition.wait(delay)

    def close(self):
        with self.condition:
            self.closed = True
            for queue in list(self.waiting.values()):
                for ticket in list(queue):
                    self._remove(ticket)
                    ticket.ready.set_exception(
                        HTTPException(503, "Execution manager is stopping")
                    )
            self.condition.notify_all()
