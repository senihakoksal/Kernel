"""The cache pre-warm must send the same prefix the real calls send.

This is a silent failure mode: if prewarm_cache() and Agent.act() build the
system or feed block even slightly differently, the pre-warm writes one cache
entry and the real calls miss it and write another. Nothing errors. The run
completes. You just pay for every write and find out from the bill.

So the test compares the two requests byte-for-byte up to the last cache
breakpoint, which is exactly the span the API matches on.

    uv run pytest tests/test_prewarm.py -v
"""

import asyncio
import json

import pytest

from agents import Agent, feed_block, prewarm_cache, system_blocks
from schema import Record


class _Usage:
    input_tokens = output_tokens = 0
    cache_creation_input_tokens = cache_read_input_tokens = 0


class _Blk:
    type = "text"
    text = '{"content": "x", "reasoning": "y", "score": 0.5}'


class _Response:
    content = [_Blk()]
    usage = _Usage()


class RecordingMessages:
    """Records every request instead of sending it."""

    def __init__(self, sink):
        self._sink = sink

    async def create(self, **kwargs):
        self._sink.append(kwargs)
        return _Response()


class RecordingClient:
    def __init__(self):
        self.requests = []
        self.messages = RecordingMessages(self.requests)


def concept(rnd: int, agent: str) -> Record:
    return Record(round=rnd, agent=agent, role="artist", model="stub", kind="concept",
                  concept_id=f"r{rnd}-{agent}", title="Work", content="a concept " * 30,
                  reasoning="r")


@pytest.fixture
def feed() -> list[Record]:
    return [concept(0, "artist_a"), concept(0, "artist_b"), concept(1, "artist_a")]


def cached_prefix(request: dict) -> str:
    """The span the API matches on: system blocks plus content up to and
    including the last block carrying a cache breakpoint."""
    content = request["messages"][0]["content"]
    last_break = max(i for i, b in enumerate(content) if b.get("cache_control"))
    return json.dumps({"system": request["system"],
                       "content": content[:last_break + 1]},
                      sort_keys=True, ensure_ascii=False)


def test_prewarm_prefix_matches_a_critic_call(feed):
    client = RecordingClient()
    critic = Agent("critic_x", "critic", "Reads for craft.", client)

    asyncio.run(prewarm_cache(client, feed, 1))
    asyncio.run(critic.act(1, feed, feed[0]))

    warm, real = client.requests
    assert cached_prefix(warm) == cached_prefix(real), \
        "pre-warm and the real call would write different cache entries"


def test_prewarm_prefix_matches_an_artist_call(feed):
    client = RecordingClient()
    artist = Agent("artist_a", "artist", "Blunt geometry.", client)
    asyncio.run(prewarm_cache(client, feed, 1))
    asyncio.run(artist.act(1, feed))
    warm, real = client.requests
    assert cached_prefix(warm) == cached_prefix(real)


def test_prewarm_generates_nothing(feed):
    """max_tokens=0 — the point is the cache write, not a response."""
    client = RecordingClient()
    asyncio.run(prewarm_cache(client, feed, 0))
    assert client.requests[0]["max_tokens"] == 0


def test_prewarm_sends_only_the_cached_blocks(feed):
    """No per-agent block: nothing after the breakpoint belongs in the warm-up."""
    client = RecordingClient()
    asyncio.run(prewarm_cache(client, feed, 0))
    content = client.requests[0]["messages"][0]["content"]
    assert len(content) == 1
    assert content[0]["cache_control"] == {"type": "ephemeral"}


def test_prewarm_never_raises(feed):
    """A failed optimisation must not kill a run that costs an hour."""

    class Exploding:
        class messages:
            @staticmethod
            async def create(**kwargs):
                raise RuntimeError("api is down")

    assert asyncio.run(prewarm_cache(Exploding(), feed, 0)) is False


def test_prewarm_is_counted_in_the_tally(feed):
    """It costs tokens, so hiding it would understate the run."""
    from agents import UsageTally
    tally = UsageTally()
    client = RecordingClient()
    asyncio.run(prewarm_cache(client, feed, 3, tally))
    assert tally.calls[3] == 1


def test_both_breakpoints_are_present(feed):
    """The layout depends on two: the system prompt and the feed."""
    assert system_blocks()[0]["cache_control"] == {"type": "ephemeral"}
    assert feed_block(feed)["cache_control"] == {"type": "ephemeral"}


def test_feed_block_changes_with_the_feed(feed):
    """Guard against a builder that ignores its argument — the prefix must
    track the feed, or every round would hit round 0's cache entry."""
    assert feed_block(feed)["text"] != feed_block(feed[:1])["text"]
    assert feed_block([])["text"] != feed_block(feed)["text"]
