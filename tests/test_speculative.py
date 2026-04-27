"""Sprint 004 L2 unit tests.

GGUF metadata validation, sampler determinism, and the
LLAMA_SPEC_FORCE_REJECT_AT debug-env honor test (xfail until the env
hook lands in fork's `common/speculative.cpp` — Phase 2 deferred item).

The metadata test parses with `gguf-py` if importable; otherwise it is
skipped. The sampler/force-reject tests require a live OpenAI-compat
server reachable at `BASE_URL` (default http://localhost:8080); they
are skipped if the server doesn't respond to /health within 2 seconds.
"""
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8080")

# Pinned defaults match scripts/validate_dflash.py.
PROMPT = "Write a quicksort algorithm in Python. Write code only."
TOKENS = 64
SEED = 42


def _server_up(timeout_s: float = 2.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{BASE_URL}/health", timeout=1) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.2)
    return False


@pytest.fixture(scope="module")
def server():
    if not _server_up():
        pytest.skip(f"no llama-server at {BASE_URL}/health")
    return BASE_URL


def _completion(base_url: str, prompt: str, *, tokens: int = TOKENS,
                seed: int = SEED, temp: float = 0.0, top_k: int = 1) -> dict:
    payload = {
        "model": "rotorquant",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": tokens,
        "temperature": temp,
        "top_k": top_k,
        "seed": seed,
        "stream": False,
        "logprobs": True,
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode())


def _tokens(resp: dict) -> list:
    try:
        choice = resp["choices"][0]
        lp = choice.get("logprobs") or {}
        items = lp.get("content") or []
        if items and "token_id" in items[0]:
            return [int(it["token_id"]) for it in items]
        if items:
            return [it.get("token", "") for it in items]
        return [choice["message"]["content"]]
    except (KeyError, IndexError):
        return []


# ── GGUF metadata validation ────────────────────────────────────────────────

class TestGGUFMetadata:
    """Validate DFlash draft GGUF metadata against PR #22105's canonical schema.

    The community drafts (`spiritbuun/Qwen3.6-27B-DFlash-GGUF`,
    `lym00/Qwen3.6-35B-A3B-DFlash-GGUF-Test`) currently fail this validation
    on tensor-name keys (see BENCHMARK-REPORT.md §10). This test codifies the
    schema contract; it skips when no candidate draft is on disk.
    """

    @pytest.fixture
    def gguf(self):
        try:
            import gguf  # type: ignore
        except ImportError:
            pytest.skip("gguf-py not installed")
        return gguf

    @pytest.fixture
    def draft_path(self):
        candidates = [
            Path("/models/dflash-draft-3.6-q4_k_m.gguf"),
            Path("/models/Qwen3.6-35B-A3B-DFlash-q8_0.gguf"),
            Path(os.environ.get("DFLASH_DRAFT_GGUF", "")),
        ]
        for c in candidates:
            if c and c.exists():
                return c
        pytest.skip("no DFlash draft GGUF on disk")

    def test_arch_string_canonical(self, gguf, draft_path):
        reader = gguf.GGUFReader(str(draft_path), "r")
        arch = None
        for kv in reader.fields.values():
            if kv.name == "general.architecture":
                arch = bytes(kv.parts[kv.data[0]]).decode()
        assert arch == "dflash", f"expected arch=dflash, got {arch!r}"

    def test_required_tensors_present(self, gguf, draft_path):
        reader = gguf.GGUFReader(str(draft_path), "r")
        names = {t.name for t in reader.tensors}
        # PR #22105 emits at least an `fc.weight` for the DFlash projection
        # head plus a token embedding under `token_embd.weight`. The community
        # drafts currently fail on `fc.weight`; that's the documented blocker.
        assert "token_embd.weight" in names, names
        assert "fc.weight" in names, (
            "missing fc.weight — community draft format mismatch "
            "(see SPRINT-004.md Phase 3)"
        )


# ── Sampler determinism ─────────────────────────────────────────────────────

class TestSamplerDeterminism:
    def test_two_requests_at_temp0_seed42_match(self, server):
        a = _completion(server, PROMPT)
        b = _completion(server, PROMPT)
        ta, tb = _tokens(a), _tokens(b)
        assert ta and tb, "empty token streams"
        assert ta == tb, f"non-deterministic: first divergence at "\
            f"{next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)}"


# ── LLAMA_SPEC_FORCE_REJECT_AT honor test ───────────────────────────────────

@pytest.mark.xfail(
    reason="LLAMA_SPEC_FORCE_REJECT_AT env hook deferred from Phase 2 — "
    "needs landing in fork's common/speculative.cpp post-cherry-pick."
)
class TestForceReject:
    def test_force_reject_preserves_output(self, server, monkeypatch):
        baseline = _tokens(_completion(server, PROMPT))
        monkeypatch.setenv("LLAMA_SPEC_FORCE_REJECT_AT", "8")
        forced = _tokens(_completion(server, PROMPT))
        assert baseline == forced, (
            "checkpoint+replay should be transparent — "
            "forced rejection must produce identical tokens"
        )
