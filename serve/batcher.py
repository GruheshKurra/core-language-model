from __future__ import annotations

import asyncio
from typing import Callable


class MicroBatcher:
    def __init__(self, fn: Callable[[list], list], max_batch: int = 8, max_wait: float = 0.005):
        self.fn = fn
        self.max_batch = max_batch
        self.max_wait = max_wait
        self.queue: asyncio.Queue = asyncio.Queue()
        self._worker = None

    def _ensure_worker(self) -> None:
        if self._worker is None:
            self._worker = asyncio.ensure_future(self._run())

    async def submit(self, item):
        self._ensure_worker()
        fut = asyncio.get_running_loop().create_future()
        await self.queue.put((item, fut))
        return await fut

    async def _collect(self, batch: list) -> None:
        while len(batch) < self.max_batch:
            batch.append(await self.queue.get())

    async def _run(self) -> None:
        while True:
            first = await self.queue.get()
            batch = [first]
            try:
                await asyncio.wait_for(self._collect(batch), timeout=self.max_wait)
            except asyncio.TimeoutError:
                pass
            items = [item for item, _ in batch]
            try:
                results = self.fn(items)
                for (_, fut), result in zip(batch, results):
                    if not fut.done():
                        fut.set_result(result)
            except Exception as exc:
                for _, fut in batch:
                    if not fut.done():
                        fut.set_exception(exc)
