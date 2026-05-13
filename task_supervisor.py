"""
task_supervisor.py — restart-on-failure wrapper for long-lived async tasks.

Phase 3 Fix #2: every long-lived background coroutine in main.py is currently
spawned as a fire-and-forget asyncio.create_task. If it returns or raises
(normally a `BaseException` like CancelledError or a fatal asyncio bug),
the task object is garbage-collected with no notification, and the bot
continues running on whatever state was left in memory.

This module wraps such coroutines in a supervisor that:

  * catches exceptions (not BaseException — CancelledError still propagates),
  * logs them with the supervisor's name,
  * optionally fires a throttled alert,
  * restarts the coroutine with exponential backoff,
  * cleanly exits when its own task is cancelled.

Usage:

    from task_supervisor import supervise

    task = asyncio.create_task(
        supervise("stream_candles", lambda: stream_candles(symbols))
    )

The factory pattern (a no-arg callable returning a coroutine) is required
because a coroutine instance can only be awaited once; we need to create a
fresh one on each restart.
"""

import asyncio
import time
from typing import Awaitable, Callable, Optional

from logger import log


async def supervise(
    name: str,
    coro_factory: Callable[[], Awaitable[None]],
    base_backoff: float = 1.0,
    max_backoff: float = 60.0,
    alert_fn: Optional[Callable[[str], Awaitable[None]]] = None,
) -> None:
    """Run `coro_factory()` forever. Restart on return or exception with
    exponential backoff. Propagates CancelledError so the parent can clean up.
    """
    backoff = base_backoff
    restarts = 0
    while True:
        start = time.time()
        try:
            log(f"🛡️ supervisor[{name}]: starting (restart #{restarts})")
            await coro_factory()
            # If we reach here, the coroutine returned normally — restart.
            log(f"⚠️ supervisor[{name}]: coroutine returned normally; restarting")
        except asyncio.CancelledError:
            log(f"🛑 supervisor[{name}]: cancelled — exiting")
            raise
        except Exception as e:
            log(
                f"❌ supervisor[{name}]: crashed after {time.time() - start:.1f}s — "
                f"{type(e).__name__}: {e}",
                level="ERROR",
            )
            if alert_fn is not None:
                try:
                    await alert_fn(
                        f"🚨 supervisor[{name}] crashed: {type(e).__name__}: {e}"
                    )
                except Exception as alert_err:
                    log(
                        f"⚠️ supervisor[{name}]: alert send failed: {alert_err}",
                        level="WARN",
                    )

        # If the wrapped coro ran for a long time before failing, reset
        # backoff — it was probably a transient issue, not a hot-loop crash.
        if time.time() - start > 60:
            backoff = base_backoff

        log(f"🔁 supervisor[{name}]: restarting in {backoff:.1f}s")
        try:
            await asyncio.sleep(backoff)
        except asyncio.CancelledError:
            log(f"🛑 supervisor[{name}]: cancelled during backoff — exiting")
            raise

        backoff = min(backoff * 2.0, max_backoff)
        restarts += 1


__all__ = ["supervise"]
