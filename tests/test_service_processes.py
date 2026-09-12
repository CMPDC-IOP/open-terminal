import asyncio
import os
import sys
import threading
from types import SimpleNamespace

import pytest

from open_terminal.utils import service_processes as helpers


@pytest.mark.parametrize(
    "limit,queue,timeout",
    [(0, 1, 1), (1, -1, 1), (1, 1, -1), (1, 1, float("nan")), (1, 1, float("inf"))],
)
def test_rejects_invalid_pool_settings(limit, queue, timeout):
    with pytest.raises(ValueError):
        helpers.HelperPool(limit, queue, timeout)


def test_async_and_thread_helpers_share_a_bounded_fifo_queue():
    async def exercise():
        pool = helpers.HelperPool(1, 1, 1)
        acquired = threading.Event()

        def worker():
            with pool.sync_slot():
                acquired.set()

        async with pool.slot():
            task = asyncio.create_task(asyncio.to_thread(worker))
            async with asyncio.timeout(2):
                while not pool._waiting:
                    await asyncio.sleep(0.001)
            with pytest.raises(helpers.HelpersBusy) as error:
                async with pool.slot():
                    pytest.fail("Queue was full")
            assert error.value.status_code == 503
            assert error.value.headers == {"Retry-After": "1"}
            assert not acquired.is_set()
        await task
        assert acquired.is_set()
        async with pool.slot():
            pass

    asyncio.run(exercise())


def test_timeout_and_cancellation_remove_waiters_and_do_not_leak_granted_slots():
    async def exercise():
        pool = helpers.HelperPool(1, 1, 0.01)
        async with pool.slot():
            for _ in range(2):
                with pytest.raises(helpers.HelpersBusy):
                    async with pool.slot():
                        pytest.fail("Active slot was not released")
                assert not pool._waiting

        async def waiter():
            async with pool.slot():
                await asyncio.sleep(0)

        for cancel_before_release in (True, False):
            for _ in range(20):
                async with pool.slot():
                    waiting = asyncio.create_task(waiter())
                    await asyncio.sleep(0)
                    if cancel_before_release:
                        waiting.cancel()
                        with pytest.raises(asyncio.CancelledError):
                            await waiting
                if not cancel_before_release:
                    waiting.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await waiting
                async with pool.slot():
                    pass
        assert pool._active == 0
        assert not pool._waiting

    asyncio.run(exercise())


def test_cancelled_thread_keeps_its_slot_until_process_completion(monkeypatch):
    pool = helpers.HelperPool(1, 0, 0)
    monkeypatch.setattr(helpers, "_pool", pool)
    started = threading.Event()
    release = threading.Event()

    def blocking_run(*args, **kwargs):
        started.set()
        assert release.wait(3)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(helpers.subprocess, "run", blocking_run)

    async def exercise():
        task = asyncio.create_task(
            asyncio.to_thread(helpers.run_helper, ["fixed-helper"])
        )
        try:
            async with asyncio.timeout(2):
                while not started.is_set():
                    await asyncio.sleep(0.001)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            with pytest.raises(helpers.HelpersBusy):
                async with pool.slot():
                    pytest.fail("Cancelled request released a running helper")
        finally:
            release.set()
        async with asyncio.timeout(2):
            while pool._active:
                await asyncio.sleep(0.001)
        async with pool.slot():
            pass

    asyncio.run(exercise())


def test_cancelled_real_helper_is_reaped_before_slot_is_reused(monkeypatch):
    monkeypatch.setattr(helpers, "_pool", helpers.HelperPool(1, 0, 0))

    async def exercise():
        started = asyncio.Event()
        process_id = None

        async def operation():
            nonlocal process_id
            async with helpers.open_helper(
                sys.executable,
                "-c",
                "import os,time; print(os.getpid(), flush=True); time.sleep(60)",
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
            ) as process:
                process_id = int(await process.stdout.readline())
                started.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(operation())
        await asyncio.wait_for(started.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        if os.name == "posix":
            with pytest.raises(ProcessLookupError):
                os.kill(process_id, 0)
        async with helpers._pool.slot():
            pass

    asyncio.run(exercise())


def test_launch_failure_and_nonzero_exit_release_slots(monkeypatch):
    monkeypatch.setattr(helpers, "_pool", helpers.HelperPool(1, 0, 0))

    async def exercise():
        with pytest.raises(FileNotFoundError):
            async with helpers.open_helper("/nonexistent/open-terminal-helper"):
                pytest.fail("Launch should fail")
        async with helpers.open_helper(
            sys.executable, "-c", "raise SystemExit(7)"
        ) as process:
            await process.communicate()
            assert process.returncode == 7
        async with helpers._pool.slot():
            pass

    asyncio.run(exercise())


def test_comparison_uses_helper_pool_and_preserves_results(tmp_path, monkeypatch):
    from open_terminal.utils.file_compare import CompareRequest, run_comparison
    from open_terminal.utils.fs import UserFS

    (tmp_path / "before.txt").write_text("before\n")
    (tmp_path / "after.txt").write_text("after\n")
    pool = helpers.HelperPool(1, 0, 0)
    monkeypatch.setattr(helpers, "_pool", pool)

    class ConnectedRequest:
        async def receive(self):
            await asyncio.Event().wait()

    async def exercise():
        payload = CompareRequest(original="before.txt", revised="after.txt")
        filesystem = UserFS(home=str(tmp_path))
        async with pool.slot():
            with pytest.raises(helpers.HelpersBusy):
                await run_comparison(ConnectedRequest(), payload, filesystem)
        result = await run_comparison(ConnectedRequest(), payload, filesystem)
        assert result["additions"] == 1
        assert result["deletions"] == 1
        assert pool._active == 0

    asyncio.run(exercise())
