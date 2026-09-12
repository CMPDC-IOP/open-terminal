"""Bounded, process-local idempotency for resource creation, not output reads."""

import asyncio
import concurrent.futures
import hashlib
import json
import logging
import math
import re
import threading
import time
from dataclasses import dataclass, field

from fastapi import HTTPException

log = logging.getLogger(__name__)
_KEY = re.compile(r"[A-Za-z0-9._:-]{1,128}\Z")


@dataclass
class Failure:
    status: int
    detail: object
    headers: dict | None = None
    retryable: bool = True

    def raise_error(self):
        raise HTTPException(self.status, self.detail, headers=self.headers)


@dataclass
class Outcome:
    value: object = None
    failure: Failure | None = None


@dataclass
class Entry:
    fingerprint: str
    owner: str
    active: object
    exists: object
    ready: concurrent.futures.Future = field(default_factory=concurrent.futures.Future)
    waiters: set = field(default_factory=set)
    task: object = None
    completed: float | None = None


class SharedRequest:
    """Original request attributes plus the aggregate creation-waiter lifetime."""

    def __init__(self, request, registry, entry):
        self._request = request
        self._registry = registry
        self._entry = entry

    def __getattr__(self, name):
        return getattr(self._request, name)

    async def is_disconnected(self):
        with self._registry.lock:
            return not self._entry.waiters


def _consume(future):
    if not future.cancelled():
        future.exception()


async def _settle(future):
    """Cancellation must not interrupt cleanup of the last creation attempt."""
    while not future.done():
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError:
            continue
    return future.result()


class Registry:
    def __init__(
        self, *, max_entries=4096, max_user_entries=256, ttl=3600, max_waiters=64
    ):
        for name, value in (
            ("max_entries", max_entries),
            ("max_user_entries", max_user_entries),
            ("max_waiters", max_waiters),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(ttl) not in (float, int) or not math.isfinite(ttl) or ttl <= 0:
            raise ValueError("ttl must be finite and positive")
        self.max_entries = max_entries
        self.max_user_entries = max_user_entries
        self.max_waiters = max_waiters
        self.ttl = ttl
        self.lock = threading.RLock()
        self.entries = {}
        self.owner_counts = {}

    def _purge(self):
        now = time.monotonic()
        for key, entry in list(self.entries.items()):
            if entry.completed is None or entry.waiters:
                continue
            outcome = entry.ready.result()
            if outcome.failure is not None:
                # The factory has finished unwinding. Unknown errors may mean
                # cleanup failed, so only release confirmed creation failures.
                if not outcome.failure.retryable:
                    continue
            else:
                if now - entry.completed < self.ttl:
                    continue
                if entry.active is not None:
                    try:
                        if entry.active(outcome.value):
                            continue
                    except Exception:
                        log.exception(
                            "Cannot verify idempotent resource lifetime; retaining its key"
                        )
                        continue
            del self.entries[key]
            self.owner_counts[entry.owner] -= 1
            if not self.owner_counts[entry.owner]:
                del self.owner_counts[entry.owner]

    def _driver_done(self, entry, task):
        # A task cancelled before its first coroutine step never reaches the
        # driver's except block; its waiters must still receive a terminal result.
        with self.lock:
            if not entry.ready.done():
                if not task.cancelled():
                    task.exception()
                entry.completed = time.monotonic()
                entry.ready.set_result(
                    Outcome(
                        failure=Failure(
                            409,
                            "Original creation was cancelled. Retry with the same Idempotency-Key.",
                        )
                    )
                )
                entry.task = None

    async def _drive(self, entry, factory, request):
        try:
            value = await factory(SharedRequest(request, self, entry))
            outcome = Outcome(value=value)
        except asyncio.CancelledError:
            outcome = Outcome(
                failure=Failure(
                    409, "Original creation was cancelled. Retry with the same Idempotency-Key."
                )
            )
        except HTTPException as error:
            # Factories propagate HTTP errors only after successful cleanup.
            # Store no traceback/request frames; concurrent waiters share the error.
            outcome = Outcome(
                failure=Failure(error.status_code, error.detail, error.headers)
            )
        except Exception:
            log.exception("Idempotent resource creation failed")
            outcome = Outcome(
                failure=Failure(
                    500, "Resource creation failed; cleanup could not be confirmed.",
                    retryable=False,
                )
            )
        with self.lock:
            entry.completed = time.monotonic()
            entry.ready.set_result(outcome)
            entry.task = None

    async def run(
        self, request, operation, payload, factory, *, active=None, exists=None
    ):
        keys = request.headers.getlist("idempotency-key")
        if not keys:
            return await factory(request)
        if len(keys) != 1 or not _KEY.fullmatch(keys[0]):
            raise HTTPException(
                400,
                "Idempotency-Key must be 1-128 letters, digits, '.', '_', ':' or '-'",
            )
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        if len(encoded) > 1024 * 1024:
            raise HTTPException(413, "Idempotent creation parameters exceed 1 MiB")
        fingerprint = hashlib.sha256(encoded).hexdigest()
        owner = request.headers.get("x-user-id", "")
        scope = (owner, request.headers.get("x-session-id", ""), operation, keys[0])
        waiter = object()
        with self.lock:
            self._purge()
            entry = self.entries.get(scope)
            if entry is not None and entry.fingerprint != fingerprint:
                raise HTTPException(
                    409,
                    "Idempotency-Key was already used with different creation parameters",
                )
            if entry is None:
                if (
                    len(self.entries) >= self.max_entries
                    or self.owner_counts.get(owner, 0) >= self.max_user_entries
                ):
                    raise HTTPException(
                        429,
                        "Idempotency records are full. Retry later.",
                        headers={"Retry-After": "1"},
                    )
                entry = Entry(fingerprint, owner, active, exists)
                entry.waiters.add(waiter)
                self.entries[scope] = entry
                self.owner_counts[owner] = self.owner_counts.get(owner, 0) + 1
                entry.task = asyncio.create_task(self._drive(entry, factory, request))
                entry.task.add_done_callback(
                    lambda task: self._driver_done(entry, task)
                )
            else:
                if len(entry.waiters) >= self.max_waiters:
                    raise HTTPException(
                        429,
                        "Too many concurrent retries for this Idempotency-Key",
                        headers={"Retry-After": "1"},
                    )
                entry.waiters.add(waiter)
        ready = asyncio.wrap_future(entry.ready)
        monitor = None

        async def watch_disconnect():
            while True:
                if await request.is_disconnected():
                    raise HTTPException(
                        499, "Request disconnected during resource creation"
                    )
                await asyncio.sleep(0.1)

        try:
            if not entry.ready.done():
                monitor = asyncio.create_task(watch_disconnect())
                done, _ = await asyncio.wait(
                    (ready, monitor), return_when=asyncio.FIRST_COMPLETED
                )
                if monitor in done:
                    monitor.result()
            else:
                await asyncio.shield(ready)
            outcome = ready.result()
            if outcome.failure is not None:
                outcome.failure.raise_error()
            if entry.exists is not None and not entry.exists(outcome.value):
                raise HTTPException(
                    410,
                    "The resource created with this Idempotency-Key no longer exists",
                )
            return outcome.value
        finally:
            if monitor is not None:
                monitor.cancel()
                monitor.add_done_callback(_consume)
            with self.lock:
                entry.waiters.discard(waiter)
                task = (
                    entry.task if not entry.waiters and not entry.ready.done() else None
                )
                if task is not None and not task.cancelling():
                    task.get_loop().call_soon_threadsafe(
                        lambda: task.cancel() if not task.cancelling() else None
                    )
            if task is not None:
                await _settle(ready)
            with self.lock:
                self._purge()


_default = None
_default_lock = threading.Lock()


def get_registry():
    global _default
    with _default_lock:
        if _default is None:
            _default = Registry()
        return _default


async def run_creation(
    request, operation, payload, factory, *, active=None, exists=None
):
    if not request.headers.getlist("idempotency-key"):
        return await factory(request)
    return await get_registry().run(
        request, operation, payload, factory, active=active, exists=exists
    )
