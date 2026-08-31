"""The round-0 null condition, enforced as a test.

control.py's claim is that it holds the stimulus fixed and removes exactly one
channel — peer critiques. In round 0 there are no peer critiques to remove yet,
so the two conditions must be indistinguishable: every critic in the control has
to receive the *byte-identical* prompt its treatment counterpart received. If
that ever stops being true, the control is no longer a null baseline and the
treatment-minus-control gap is measuring a prompt difference instead of the
peer-critique channel.

The artworks are identical by construction — control.py replays them. What is
checked here is the thing that can silently drift: the exact prompt string each
critic receives, system prompt + feed + disposition + task instruction.

How it runs both paths without touching the network: every agent gets a stub
client whose messages.create() records the request and returns canned JSON, so
run.run() and control.run_control() execute for real — the real feed assembly,
the real ordering, the real .jsonl round-trip — while making zero API calls. The
comparison is over the full request (system blocks, message-block boundaries,
cache_control, model, max_tokens), not concatenated text, because prompt layout
is part of the prompt: commit 940a61a moved the disposition out of the system
prompt and after the feed, which concatenation alone would not notice.

    uv run pytest tests/test_round0_null.py -v
"""

import asyncio
import contextvars
import difflib
import json

import pytest

import agents
import control
import run
from prompt_check import canonical, render
from schema import Record

# Two rounds, not one: round 0 pins the null, round 1 proves the manipulation
# actually engages. A control that leaked peer critiques would pass a round-0-
# only test and still be broken.
ROUNDS = 2

# A fixture roster, not the live agents.yaml — the invariant under test is a
# property of the code paths, and the test should not start failing because
# someone added an artist.
ROSTER = """\
agents:
  - name: stub_artist_one
    role: artist
    disposition: >
      Works in blunt geometry and flat colour.
  - name: stub_artist_two
    role: artist
    disposition: >
      Works in slow, accumulating text pieces.
  - name: stub_critic_alpha
    role: critic
    disposition: >
      Reads for craft and sincerity; distrusts spectacle.
  - name: stub_critic_beta
    role: critic
    disposition: >
      Reads for context and lineage; distrusts novelty claims.
  - name: stub_critic_gamma
    role: critic
    disposition: >
      Reads for material logic; distrusts the artist's statement.
"""

N_ARTISTS = 2
N_CRITICS = 3

# Set inside Agent.act, read inside the stub client. A ContextVar (not a plain
# global) because agents run concurrently under asyncio.gather: each task gets
# its own copy of the context, so parallel calls cannot clobber each other's
# identity the way a shared global would.
_CURRENT: contextvars.ContextVar = contextvars.ContextVar("kernel_call")


class _TextBlock:
    """Stands in for an SDK text block: agents.py reads .type and .text."""

    type = "text"

    def __init__(self, text: str):
        self.text = text


class _Response:
    def __init__(self, text: str):
        self.content = [_TextBlock(text)]


class _StubMessages:
    """Records the request, returns canned JSON. Never touches the network."""

    def __init__(self, calls: list, phase: dict):
        self._calls = calls
        self._phase = phase

    async def create(self, **payload):
        name, role, round_idx, concept_id = _CURRENT.get()
        self._calls.append({"phase": self._phase["name"], "agent": name, "role": role,
                            "round": round_idx, "concept_id": concept_id,
                            "payload": payload})
        # Deterministic in (agent, round, target): the control regenerates each
        # critic's own critiques, so identical text there keeps round 1's only
        # difference the removal of peers — exactly the variable under test.
        if role == "artist":
            body = {"title": f"Piece by {name} r{round_idx}",
                    # Curly quotes and an em dash on purpose: format_feed wraps
                    # titles in “ ”, and the control reads its concepts back
                    # through JSON. A round-trip that mangles either would show
                    # up as a round-0 prompt mismatch, which is the point.
                    "content": f"“{name}” offers a round-{round_idx} concept — flat, plain, stubbed.",
                    "reasoning": "stub reasoning"}
        else:
            body = {"content": f"{name} on {concept_id} in round {round_idx}: stubbed critique.",
                    "reasoning": "stub reasoning", "score": 0.5}
        return _Response(json.dumps(body, ensure_ascii=False))


class _StubClient:
    def __init__(self, calls: list, phase: dict):
        self.messages = _StubMessages(calls, phase)


def _recording_act(real_act):
    """Wrap Agent.act so the stub client knows whose call it is fielding."""

    async def act(self, round_idx, feed, target=None):
        _CURRENT.set((self.name, self.role, round_idx,
                      target.concept_id if target is not None else None))
        return await real_act(self, round_idx, feed, target)

    return act


@pytest.fixture(scope="module")
def both_conditions(tmp_path_factory):
    """Run a treatment and its control end-to-end in stub mode.

    Module-scoped: both runs are driven once and every test reads the same
    captured calls, so the tests cannot disagree about what happened.
    """
    tmp = tmp_path_factory.mktemp("kernel")
    roster = tmp / "agents.yaml"
    roster.write_text(ROSTER)

    calls: list = []
    phase = {"name": "treatment"}

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(run, "AGENTS_FILE", roster)
        mp.setattr(run, "LOG_DIR", tmp / "logs")
        mp.setattr(control, "LOG_DIR", tmp / "logs")
        # Hermetic: no .env read, no API key needed, no client that could dial out.
        mp.setattr(run, "load_dotenv", lambda *a, **k: None)
        mp.setattr(control, "load_dotenv", lambda *a, **k: None)
        mp.setattr(run, "AsyncAnthropic", lambda **kw: _StubClient(calls, phase))
        mp.setattr(control, "AsyncAnthropic", lambda **kw: _StubClient(calls, phase))
        mp.setattr(agents.Agent, "act", _recording_act(agents.Agent.act))

        treatment_log = asyncio.run(run.run(rounds=ROUNDS))
        phase["name"] = "control"
        control_log = asyncio.run(control.run_control(treatment_log))

    return {"calls": calls, "treatment_log": treatment_log, "control_log": control_log}


def critic_prompts(calls: list, phase: str, round_idx: int) -> dict:
    """{(critic, concept_id): canonical request} for one phase and round."""
    return {(c["agent"], c["concept_id"]): canonical(c["payload"])
            for c in calls
            if c["phase"] == phase and c["role"] == "critic" and c["round"] == round_idx}


def payload_for(calls: list, phase: str, round_idx: int, key: tuple) -> dict:
    for c in calls:
        if (c["phase"] == phase and c["role"] == "critic"
                and c["round"] == round_idx and (c["agent"], c["concept_id"]) == key):
            return c["payload"]
    raise KeyError(key)


def concepts_in(log_path, round_idx: int) -> list:
    """Replayed artworks as logged, in file order — order is feed order."""
    return [r for r in
            (Record(**json.loads(line)) for line in log_path.read_text().splitlines() if line.strip())
            if r.kind == "concept" and r.round == round_idx]


def test_stub_mode_made_no_real_calls(both_conditions):
    """Guard the guard: if the stub were bypassed the rest proves nothing."""
    calls = both_conditions["calls"]
    assert calls, "no calls captured — the stub client was never installed"
    expected_treatment = ROUNDS * (N_ARTISTS + N_CRITICS * N_ARTISTS)
    assert sum(1 for c in calls if c["phase"] == "treatment") == expected_treatment
    # The control replays artworks instead of regenerating them: critics only.
    assert sum(1 for c in calls if c["phase"] == "control") == ROUNDS * N_CRITICS * N_ARTISTS
    assert not any(c["role"] == "artist" for c in calls if c["phase"] == "control")


def test_round0_replayed_artworks_are_identical(both_conditions):
    """Precondition for the prompt check: same stimulus, same order."""
    treat = concepts_in(both_conditions["treatment_log"], 0)
    ctrl = concepts_in(both_conditions["control_log"], 0)
    assert [(c.concept_id, c.title, c.content) for c in treat] == \
           [(c.concept_id, c.title, c.content) for c in ctrl]


def test_round0_critic_prompts_are_byte_identical(both_conditions):
    """The null: in round 0 the control must be indistinguishable from treatment."""
    calls = both_conditions["calls"]
    treat = critic_prompts(calls, "treatment", 0)
    ctrl = critic_prompts(calls, "control", 0)

    assert len(treat) == N_CRITICS * N_ARTISTS, "unexpected treatment critic call count"
    assert set(treat) == set(ctrl), (
        f"different (critic, artwork) pairs critiqued in round 0: "
        f"treatment-only {sorted(set(treat) - set(ctrl))}, "
        f"control-only {sorted(set(ctrl) - set(treat))}"
    )

    mismatched = sorted(key for key in treat if treat[key] != ctrl[key])
    if mismatched:
        key = mismatched[0]
        diff = "\n".join(difflib.unified_diff(
            render(payload_for(calls, "treatment", 0, key)).splitlines(),
            render(payload_for(calls, "control", 0, key)).splitlines(),
            fromfile="treatment", tofile="control", lineterm=""))
        pytest.fail(
            f"round-0 prompts differ for {len(mismatched)}/{len(treat)} pairs "
            f"{mismatched}\n\nFirst difference ({key[0]} on {key[1]}):\n{diff}"
        )


def test_round0_prompts_share_one_system_prompt(both_conditions):
    """Every round-0 critic call, both phases, carries the same system prompt.

    A per-agent system prompt would break the shared cache prefix agents.py is
    built around, and would do it without changing any single pair's equality.
    """
    systems = {json.dumps(c["payload"].get("system"), sort_keys=True)
               for c in both_conditions["calls"]
               if c["role"] == "critic" and c["round"] == 0}
    assert len(systems) == 1, "round-0 critics saw more than one system prompt"


def test_later_rounds_drop_peer_critiques(both_conditions):
    """The manipulation must actually engage once there are peers to remove.

    The mirror image of the round-0 null: if round 1 also matched, the control
    would be leaking peer critiques and the experiment would have no contrast.
    """
    calls = both_conditions["calls"]
    treat = critic_prompts(calls, "treatment", 1)
    ctrl = critic_prompts(calls, "control", 1)
    assert set(treat) == set(ctrl)
    assert all(treat[key] != ctrl[key] for key in treat), (
        "round-1 control prompts match the treatment — peer critiques are not "
        "being withheld, so the control is not isolating anything"
    )

    # And the difference is specifically peers: a control critic's feed may name
    # itself, never another critic.
    for key in ctrl:
        critic = key[0]
        text = render(payload_for(calls, "control", 1, key))
        seen = {line.split()[3] for line in text.splitlines()
                if line.startswith("[round ") and " CRITIC " in line}
        assert seen <= {critic}, f"{critic}'s isolated feed contains peers: {seen - {critic}}"
