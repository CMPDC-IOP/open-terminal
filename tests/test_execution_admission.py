"""Fair admission contracts with controlled identities and tiny fake budgets."""

import asyncio
import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from open_terminal.execution import manager as module
from open_terminal.execution.policy import ExecutionPolicy, Limits


class FakeLaunch:
    def __init__(self, manager, identity, path, *args):
        self.manager = manager
        self.identity = identity
        self.closed = False
        self.started_at = None
        self._deadline_monotonic = None

    def abort(self):
        if not self.closed:
            self.closed = True
            self.manager._release(*self.identity)

    def request_stop(self):
        self.abort()


@pytest.fixture
def manager(tmp_path, monkeypatch):
    unit = Limits(100, 4096, 4)
    policy = ExecutionPolicy(
        tmp_path,
        unit,
        unit,
        unit,
        unit,
        1,
        1,
        max_queue=8,
        max_user_queue=4,
        queue_timeout=1,
    )
    monkeypatch.setattr(
        "pwd.getpwnam",
        lambda name: SimpleNamespace(
            pw_uid=10000 + int(name),
            pw_gid=10000 + int(name),
            pw_name=name,
            pw_dir="/home/" + name,
        ),
    )
    monkeypatch.setattr(module, "Launch", FakeLaunch)
    tree = Mock()
    tree.create_task.side_effect = lambda owner, task: tmp_path / owner / task
    result = module.Manager(policy, tree)
    yield result
    result.shutdown()


def submit(manager, owner):
    return manager.submit(owner, owner, ["true"])


def test_round_robin_prevents_same_owner_flood(manager):
    initial = submit(manager, "1").ready.result(timeout=1)
    a = submit(manager, "1")
    aa = submit(manager, "1")
    b = submit(manager, "2")
    bb = submit(manager, "2")
    assert manager.tree.create_task.call_count == 1
    assert manager._queue.count == 4
    initial.abort()
    second = b.ready.result(timeout=1)
    assert not a.ready.done()
    assert second.started_at is None and second._deadline_monotonic is None
    second.abort()
    third = a.ready.result(timeout=1)
    assert not bb.ready.done()
    third.abort()
    fourth = bb.ready.result(timeout=1)
    fourth.abort()
    aa.ready.result(timeout=1).abort()
    assert not manager._tasks
    assert manager._queue.count == 0


def test_new_user_budget_is_not_starved_by_renewals(manager):
    manager.policy = replace(manager.policy, max_tasks=2, max_user_tasks=2)
    one = submit(manager, "1").ready.result(timeout=1)
    two = submit(manager, "1").ready.result(timeout=1)
    renewal = submit(manager, "1")
    new_user = submit(manager, "2")
    one.abort()
    with manager._lock:
        manager._queue._dispatch()
        assert not renewal.ready.done()
        assert not new_user.ready.done()
    two.abort()
    new_user.ready.result(timeout=1).abort()
    renewal.ready.result(timeout=1).abort()


def test_queue_bounds_and_disabled_queue(manager):
    initial = submit(manager, "1").ready.result(timeout=1)
    manager.policy = replace(manager.policy, max_queue=2, max_user_queue=1)
    a = submit(manager, "1")
    with pytest.raises(HTTPException) as error:
        submit(manager, "1")
    assert error.value.status_code == 429
    b = submit(manager, "2")
    with pytest.raises(HTTPException):
        submit(manager, "3")
    manager.cancel_ticket(a)
    manager.cancel_ticket(b)
    manager.policy = replace(manager.policy, max_queue=0)
    with pytest.raises(HTTPException):
        submit(manager, "2")
    initial.abort()
    submit(manager, "2").ready.result(timeout=1).abort()


def test_queue_timeout_never_creates_task(manager):
    initial = submit(manager, "1").ready.result(timeout=1)
    manager.policy = replace(manager.policy, queue_timeout=0.02)
    waiting = submit(manager, "2")
    with pytest.raises(HTTPException) as error:
        waiting.ready.result(timeout=1)
    assert error.value.status_code == 429
    assert manager.tree.create_task.call_count == 1
    assert manager._queue.count == 0
    initial.abort()


def test_start_preparation_failure_does_not_block_next_user(manager):
    initial = submit(manager, "1").ready.result(timeout=1)
    broken = submit(manager, "2")
    healthy = submit(manager, "3")
    manager.tree.create_task.side_effect = [OSError("test failure"), "/fake"]
    initial.abort()
    with pytest.raises(OSError):
        broken.ready.result(timeout=1)
    healthy.ready.result(timeout=1).abort()
    assert not manager._tasks


def test_cancel_pending_and_grant_race_release_once(manager):
    initial = submit(manager, "1").ready.result(timeout=1)
    waiting = submit(manager, "2")
    manager.cancel_ticket(waiting)
    assert waiting.ready.cancelled()
    granted = submit(manager, "3")
    initial.abort()
    launch = granted.ready.result(timeout=1)
    manager.cancel_ticket(granted)
    manager.cancel_ticket(granted)
    assert launch.closed and not manager._tasks


def test_shutdown_rejects_queued_work(manager):
    submit(manager, "1").ready.result(timeout=1)
    waiting = submit(manager, "2")
    manager.shutdown()
    with pytest.raises(HTTPException) as error:
        waiting.ready.result(timeout=1)
    assert error.value.status_code == 503
    assert not manager._tasks
    assert manager._queue.count == 0


def test_direct_prepare_cannot_jump_waiters(manager):
    initial = submit(manager, "1").ready.result(timeout=1)
    waiting = submit(manager, "2")
    with manager._lock:
        initial.abort()
        with pytest.raises(HTTPException) as error:
            manager.prepare("3", "3", ["true"])
        assert error.value.status_code == 429
    waiting.ready.result(timeout=1).abort()


def test_queue_wait_is_async_and_cancellation_removes_ticket(manager, monkeypatch):
    monkeypatch.setattr(module, "_runtime", lambda: manager)
    initial = submit(manager, "1").ready.result(timeout=1)
    enqueued = threading.Event()
    original = manager.submit

    def tracked(*args, **kwargs):
        result = original(*args, **kwargs)
        enqueued.set()
        return result

    monkeypatch.setattr(manager, "submit", tracked)

    async def exercise():
        waiting = asyncio.create_task(module.prepare_async("2", "2", ["true"]))
        await asyncio.wait_for(asyncio.to_thread(enqueued.wait), 1)
        assert (
            await asyncio.wait_for(module.async_call(lambda: "control available"), 1)
            == "control available"
        )
        assert (
            await asyncio.wait_for(asyncio.to_thread(lambda: "files available"), 1)
            == "files available"
        )
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        assert manager._queue.count == 0
        assert len(manager._tasks) == 1

    asyncio.run(exercise())
    initial.abort()


def test_retained_command_payload_is_bounded(manager):
    with pytest.raises(HTTPException) as error:
        manager.submit("1", "1", ["x" * (1024 * 1024 + 1)])
    assert error.value.status_code == 413
    manager.tree.create_task.assert_not_called()


def test_disconnected_http_request_is_removed_from_queue(manager, monkeypatch):
    monkeypatch.setattr(module, "_runtime", lambda: manager)
    initial = submit(manager, "1").ready.result(timeout=1)

    class DisconnectedRequest:
        async def is_disconnected(self):
            return True

    async def exercise():
        with (
            module.queue_request(DisconnectedRequest()),
            pytest.raises(HTTPException) as error,
        ):
            await module.prepare_async("2", "2", ["true"])
        assert error.value.status_code == 499
        assert manager._queue.count == 0
        assert len(manager._tasks) == 1

    asyncio.run(exercise())
    initial.abort()


@pytest.mark.parametrize("field", ["max_queue", "max_user_queue"])
@pytest.mark.parametrize("value", [-1, 1.5, True, "1"])
def test_queue_count_configuration_is_strict(manager, field, value):
    with pytest.raises(ValueError):
        replace(manager.policy, **{field: value})


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_queue_wait_configuration_is_finite_positive(manager, value):
    with pytest.raises(ValueError):
        replace(manager.policy, queue_timeout=value)


def test_cancellation_during_submit_cleans_already_granted_launch(manager, monkeypatch):
    monkeypatch.setattr(module, "_runtime", lambda: manager)
    granted, release = threading.Event(), threading.Event()
    original = manager.submit

    def delayed(*args, **kwargs):
        ticket = original(*args, **kwargs)
        granted.set()
        assert release.wait(2)
        return ticket

    monkeypatch.setattr(manager, "submit", delayed)

    async def exercise():
        request = asyncio.create_task(module.prepare_async("1", "1", ["true"]))
        try:
            await asyncio.wait_for(asyncio.to_thread(granted.wait), 1)
            request.cancel()
            await asyncio.sleep(0)
            assert manager._tasks
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert not manager._tasks

    asyncio.run(exercise())
