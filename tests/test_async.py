import asyncio

import pytest

from semcache import SemanticCache


def run(coro):
    return asyncio.run(coro)


def test_aget_or_call_caches_paraphrase():
    cache = SemanticCache()
    calls = []

    async def llm(prompt):
        calls.append(prompt)
        await asyncio.sleep(0)
        return "answer"

    async def main():
        a = await cache.aget_or_call("Explain vector databases", llm)
        b = await cache.aget_or_call("explain vector databases?", llm)
        return a, b

    assert run(main()) == ("answer", "answer")
    assert len(calls) == 1
    assert cache.stats.hits == 1


def test_concurrent_identical_prompts_are_coalesced():
    cache = SemanticCache()
    calls = 0

    async def slow_llm(prompt):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)
        return f"done:{prompt}"

    async def main():
        return await asyncio.gather(*[cache.aget_or_call("summarise the report", slow_llm) for _ in range(5)])

    results = run(main())
    assert results == ["done:summarise the report"] * 5
    assert calls == 1
    assert len(cache) == 1


def test_failure_propagates_and_is_not_cached():
    cache = SemanticCache()
    attempts = 0

    async def flaky(prompt):
        nonlocal attempts
        attempts += 1
        await asyncio.sleep(0.01)
        if attempts == 1:
            raise RuntimeError("rate limited")
        return "ok"

    async def main():
        results = await asyncio.gather(
            cache.aget_or_call("classify this ticket", flaky),
            cache.aget_or_call("classify this ticket", flaky),
            return_exceptions=True,
        )
        assert all(isinstance(r, RuntimeError) for r in results)
        assert len(cache) == 0
        return await cache.aget_or_call("classify this ticket", flaky)

    assert run(main()) == "ok"
    assert attempts == 2


def test_acached_decorator_partitions_by_kwargs():
    cache = SemanticCache()
    seen = []

    @cache.acached()
    async def complete(prompt, model="small"):
        seen.append(model)
        return f"{model}:{prompt}"

    async def main():
        await complete("what is dropout", model="small")
        await complete("what is dropout", model="small")
        await complete("what is dropout", model="large")

    run(main())
    assert seen == ["small", "large"]
