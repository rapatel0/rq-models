"""Sprint 004 L2 end-to-end Docker integration test.

Brings up the `qwen36-27b-dflash` compose profile, polls /health, submits
a single greedy completion, asserts the response matches a target-only
completion on the same prompt, and tears down.

Skipped unless `DFLASH_E2E=1` and `docker` is on PATH. The full-blown
docker run requires a working DFlash draft GGUF (community drafts fail
to load — see BENCHMARK-REPORT.md §10).
"""
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request

import pytest

PORT = int(os.environ.get("PORT", "8088"))
BASE_URL = f"http://localhost:{PORT}"
PROMPT = "Write a quicksort algorithm in Python. Write code only."
TOKENS = 64
SEED = 42

DFLASH_PROFILE = "qwen36-27b-dflash"
TARGET_PROFILE = "qwen36-27b"

pytestmark = pytest.mark.docker


def _docker_or_skip():
    if not shutil.which("docker"):
        pytest.skip("docker not on PATH")
    if os.environ.get("DFLASH_E2E") != "1":
        pytest.skip("set DFLASH_E2E=1 to run end-to-end docker test "
                    "(requires working draft GGUF)")


def _wait_health(timeout_s: int = 240) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{BASE_URL}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            time.sleep(2)
    return False


def _completion(prompt: str) -> dict:
    payload = {
        "model": "rotorquant",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": TOKENS,
        "temperature": 0.0,
        "top_k": 1,
        "seed": SEED,
        "stream": False,
        "logprobs": True,
    }
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode())


def _content(resp: dict) -> str:
    return resp["choices"][0]["message"]["content"]


def _compose(args: list, env: dict | None = None):
    cmd = ["docker", "compose", *args]
    full_env = {**os.environ, **(env or {})}
    return subprocess.run(cmd, env=full_env, capture_output=True, text=True)


def _run_profile(profile: str) -> str:
    """Bring profile up, return target-only completion content. Caller must
    tear down via `_stop_profile`."""
    up = _compose(["--profile", profile, "up", "-d"], env={"PORT": str(PORT)})
    if up.returncode != 0:
        pytest.fail(f"docker compose up failed: {up.stderr[-2000:]}")
    if not _wait_health():
        _stop_profile(profile)
        pytest.fail(f"profile {profile} did not become healthy in time")
    try:
        resp = _completion(PROMPT)
        return _content(resp)
    finally:
        _stop_profile(profile)


def _stop_profile(profile: str):
    _compose(["--profile", profile, "down"], env={"PORT": str(PORT)})


def test_dflash_matches_target_only():
    _docker_or_skip()
    target_text = _run_profile(TARGET_PROFILE)
    dflash_text = _run_profile(DFLASH_PROFILE)
    assert target_text == dflash_text, (
        f"DFlash output diverged from target-only:\n"
        f"  target: {target_text[:200]!r}\n"
        f"  dflash: {dflash_text[:200]!r}"
    )
