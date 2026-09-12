import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from starlette.datastructures import Headers

from open_terminal.utils import idempotency as module


def test_default_registry_has_no_deployment_configuration_reader(monkeypatch):
    monkeypatch.setenv("OPEN_TERMINAL_IDEMPOTENCY_TTL", "0")
    monkeypatch.setattr(module, "_default", None)
    created = []

    def create_registry(**kwargs):
        created.append(kwargs)
        return object()

    monkeypatch.setattr(module, "Registry", create_registry)

    module.get_registry()
    assert created == [{}]


def request(key="key", owner="alice", context="chat"):
    headers = {"x-user-id": owner, "x-session-id": context}
    if key is not None:
        headers["idempotency-key"] = key
    return SimpleNamespace(
        headers=Headers(headers), is_disconnected=AsyncMock(return_value=False)
    )


def test_concurrent_retries_share_one_creation():
    async def exercise():
        registry = module.Registry()
        entered, finish = asyncio.Event(), asyncio.Event()
        calls = []

        async def create(shared):
            calls.append(shared.headers["x-user-id"])
            entered.set()
            await finish.wait()
            return {"id": "one"}

        first = asyncio.create_task(
            registry.run(request(), "command", {"cmd": "x"}, create)
        )
        await entered.wait()
        second = asyncio.create_task(
            registry.run(request(), "command", {"cmd": "x"}, create)
        )
        await asyncio.sleep(0)
        finish.set()
        a, b = await asyncio.gather(first, second)
        assert a == b == {"id": "one"}
        assert calls == ["alice"]
        assert await registry.run(request(), "command", {"cmd": "x"}, create) == a
        assert len(calls) == 1

    asyncio.run(exercise())


def test_scope_conflicts_and_payload_order():
    async def exercise():
        registry = module.Registry()
        create = AsyncMock(return_value="one")
        await registry.run(request(), "command", {"env": {"B": "2", "A": "1"}}, create)
        await registry.run(request(), "command", {"env": {"A": "1", "B": "2"}}, create)
        assert create.await_count == 1
        with pytest.raises(HTTPException) as error:
            await registry.run(request(), "command", {"cmd": "changed"}, create)
        assert error.value.status_code == 409
        for req, operation in [
            (request(owner="bob"), "command"),
            (request(context="other"), "command"),
            (request(), "notebook"),
        ]:
            await registry.run(req, operation, {}, create)
        assert create.await_count == 4

    asyncio.run(exercise())


def test_one_cancelled_waiter_does_not_cancel_shared_creation():
    async def exercise():
        registry = module.Registry()
        entered, finish = asyncio.Event(), asyncio.Event()
        alive = []

        async def create(shared):
            entered.set()
            await finish.wait()
            alive.append(not await shared.is_disconnected())
            return "one"

        first = asyncio.create_task(registry.run(request(), "command", {}, create))
        await entered.wait()
        second = asyncio.create_task(registry.run(request(), "command", {}, create))
        await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        finish.set()
        assert await second == "one"
        assert alive == [True]

    asyncio.run(exercise())


def test_last_cancellation_waits_for_cleanup_before_same_key_can_retry():
    async def exercise():
        registry = module.Registry()
        entered, cleaned = asyncio.Event(), asyncio.Event()

        async def create(shared):
            entered.set()
            try:
                await asyncio.Future()
            finally:
                await asyncio.sleep(0.01)
                cleaned.set()

        first = asyncio.create_task(registry.run(request(), "command", {}, create))
        await entered.wait()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert cleaned.is_set()
        assert not registry.entries
        assert not registry.owner_counts
        replacement = AsyncMock(return_value="replacement")
        assert await registry.run(request(), "command", {}, replacement) == "replacement"
        replacement.assert_awaited_once()

    asyncio.run(exercise())


def test_disconnected_waiter_is_removed_and_creation_cleaned():
    async def exercise():
        registry = module.Registry()
        cleaned = asyncio.Event()
        req = request()
        req.is_disconnected.return_value = True

        async def create(shared):
            try:
                await asyncio.Future()
            finally:
                cleaned.set()

        with pytest.raises(HTTPException) as error:
            await registry.run(req, "command", {}, create)
        assert error.value.status_code == 499
        assert cleaned.is_set()

    asyncio.run(exercise())


def test_bounds_do_not_evict_existing_retry_records():
    async def exercise():
        registry = module.Registry(max_entries=2, max_user_entries=1)
        create = AsyncMock(return_value="one")
        await registry.run(request(), "command", {}, create)
        with pytest.raises(HTTPException) as error:
            await registry.run(request(key="another"), "command", {}, create)
        assert error.value.status_code == 429
        await registry.run(request(owner="bob"), "command", {}, create)
        with pytest.raises(HTTPException):
            await registry.run(request(owner="charlie"), "command", {}, create)
        assert await registry.run(request(), "command", {}, create) == "one"
        assert create.await_count == 2

    asyncio.run(exercise())


def test_ttl_preserves_active_resource_and_expires_finished_record(monkeypatch):
    async def exercise():
        clock = SimpleNamespace(now=100)
        monkeypatch.setattr(module.time, "monotonic", lambda: clock.now)
        registry = module.Registry(ttl=2)
        live = [True]
        create = AsyncMock(return_value="one")
        await registry.run(request(), "command", {}, create, active=lambda _: live[0])
        clock.now = 103
        await registry.run(request(), "command", {}, create)
        assert create.await_count == 1
        live[0] = False
        await registry.run(request(), "command", {}, create)
        assert create.await_count == 2

    asyncio.run(exercise())


def test_deleted_resource_is_gone_not_recreated():
    async def exercise():
        registry = module.Registry()
        present = [True]
        create = AsyncMock(return_value="one")
        await registry.run(
            request(), "terminal", {}, create, exists=lambda _: present[0]
        )
        present[0] = False
        with pytest.raises(HTTPException) as error:
            await registry.run(request(), "terminal", {}, create)
        assert error.value.status_code == 410
        assert create.await_count == 1

    asyncio.run(exercise())


def test_failed_creation_releases_capacity_and_same_key_can_retry():
    async def exercise():
        registry = module.Registry()
        create = AsyncMock(
            side_effect=HTTPException(429, "queue full", headers={"Retry-After": "1"})
        )
        for _ in range(2):
            with pytest.raises(HTTPException) as error:
                await registry.run(request(), "command", {}, create)
            assert error.value.status_code == 429
        assert create.await_count == 2
        assert not registry.entries
        assert not registry.owner_counts
        success = AsyncMock(return_value="fresh")
        assert await registry.run(request(), "command", {}, success) == "fresh"
        success.reset_mock()
        for _ in range(2):
            await registry.run(request(key=None), "command", {}, success)
        assert success.await_count == 2

    asyncio.run(exercise())


@pytest.mark.parametrize("key", ["", " ", "x" * 129, "é", "a/b"])
def test_invalid_keys_do_not_start_creation(key):
    async def exercise():
        create = AsyncMock()
        with pytest.raises(HTTPException) as error:
            await module.Registry().run(request(key=key), "command", {}, create)
        assert error.value.status_code == 400
        create.assert_not_awaited()

    asyncio.run(exercise())


def test_pending_entry_never_expires_and_waiters_are_bounded(monkeypatch):
    async def exercise():
        clock = SimpleNamespace(now=100)
        monkeypatch.setattr(module.time, "monotonic", lambda: clock.now)
        registry = module.Registry(ttl=1, max_waiters=1)
        entered, finish = asyncio.Event(), asyncio.Event()

        async def create(shared):
            entered.set()
            await finish.wait()
            return "one"

        first = asyncio.create_task(registry.run(request(), "command", {}, create))
        await entered.wait()
        clock.now = 200
        with pytest.raises(HTTPException) as error:
            await registry.run(request(), "command", {}, AsyncMock())
        assert error.value.status_code == 429
        assert len(registry.entries) == 1
        finish.set()
        assert await first == "one"

    asyncio.run(exercise())


def test_duplicate_key_headers_are_rejected():
    async def exercise():
        req = request()
        req.headers = Headers(
            raw=[(b"idempotency-key", b"a"), (b"idempotency-key", b"b")]
        )
        factory = AsyncMock()
        with pytest.raises(HTTPException) as error:
            await module.Registry().run(req, "command", {}, factory)
        assert error.value.status_code == 400
        factory.assert_not_awaited()

    asyncio.run(exercise())


def test_waiter_gets_terminal_result_if_driver_cancelled_before_start():
    async def exercise():
        registry = module.Registry()
        entry = module.Entry("fingerprint", "alice", None, None)
        task = asyncio.create_task(asyncio.sleep(60))
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        registry._driver_done(entry, task)
        assert entry.ready.result().failure.status == 409

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_entries": 0},
        {"max_user_entries": -1},
        {"max_waiters": True},
        {"ttl": float("inf")},
        {"ttl": float("nan")},
        {"ttl": 0},
        {"ttl": True},
    ],
)
def test_invalid_record_limits(kwargs):
    with pytest.raises(ValueError):
        module.Registry(**kwargs)


def test_retries_during_cleanup_share_failure_without_interrupting_cleanup():
    async def exercise():
        registry = module.Registry()
        entered, cleaning, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def create(shared):
            entered.set()
            try:
                await asyncio.Future()
            finally:
                cleaning.set()
                await finish.wait()

        first = asyncio.create_task(registry.run(request(), "command", {}, create))
        await entered.wait()
        first.cancel()
        await cleaning.wait()
        replacement = AsyncMock(return_value="replacement")
        retry = asyncio.create_task(registry.run(request(), "command", {}, replacement))
        await asyncio.sleep(0)
        retry.cancel()
        await asyncio.sleep(0)
        assert not first.done()
        assert not retry.done()
        replacement.assert_not_awaited()
        # Another retry remains attached to the original cleanup and receives
        # its failure. Only a subsequent request can start a new creation.
        waiting = asyncio.create_task(registry.run(request(), "command", {}, replacement))
        await asyncio.sleep(0)
        finish.set()
        results = await asyncio.gather(first, retry, waiting, return_exceptions=True)
        assert all(isinstance(result, asyncio.CancelledError) for result in results[:2])
        assert isinstance(results[2], HTTPException)
        assert results[2].status_code == 409
        replacement.assert_not_awaited()
        assert await registry.run(request(), "command", {}, replacement) == "replacement"
        replacement.assert_awaited_once()

    asyncio.run(exercise())


def test_unconfirmed_cleanup_retains_key_even_after_ttl(monkeypatch):
    async def exercise():
        clock = SimpleNamespace(now=100)
        monkeypatch.setattr(module.time, "monotonic", lambda: clock.now)
        registry = module.Registry(ttl=1)
        failed = AsyncMock(side_effect=RuntimeError("cleanup failed"))
        with pytest.raises(HTTPException) as error:
            await registry.run(request(), "command", {}, failed)
        assert error.value.status_code == 500
        clock.now = 200
        replacement = AsyncMock()
        with pytest.raises(HTTPException, match="cleanup could not be confirmed"):
            await registry.run(request(), "command", {}, replacement)
        replacement.assert_not_awaited()
        assert len(registry.entries) == 1

    asyncio.run(exercise())
