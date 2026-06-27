import asyncio

from serve.batcher import MicroBatcher


def test_results_correct_and_ordered():
    def fn(items):
        return [x * 2 for x in items]

    async def main():
        mb = MicroBatcher(fn, max_batch=8, max_wait=0.1)
        return await asyncio.gather(*[mb.submit(i) for i in range(8)])

    results = asyncio.run(main())
    assert results == [i * 2 for i in range(8)]


def test_groups_into_batches():
    sizes = []

    def fn(items):
        sizes.append(len(items))
        return items

    async def main():
        mb = MicroBatcher(fn, max_batch=16, max_wait=0.1)
        await asyncio.gather(*[mb.submit(i) for i in range(10)])

    asyncio.run(main())
    assert sum(sizes) == 10
    assert max(sizes) > 1


def test_exception_propagates():
    def fn(items):
        raise ValueError("boom")

    async def main():
        mb = MicroBatcher(fn, max_batch=2, max_wait=0.05)
        return await mb.submit(1)

    try:
        asyncio.run(main())
        assert False
    except ValueError as exc:
        assert "boom" in str(exc)
