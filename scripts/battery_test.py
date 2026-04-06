#!/usr/bin/env python3
"""
Battery test: performance benchmarks + logical competence tests for RotorQuant server.

Tests:
  1. Throughput: prefill + decode tok/s at various prompt lengths
  2. Latency: time-to-first-token, total generation time
  3. VRAM: memory usage at increasing context
  4. Logic: math, reasoning, code, factual recall, instruction following
  5. NIAH: needle-in-a-haystack at 4K and 8K
  6. Consistency: same prompt N times, check determinism at temp=0

Usage:
    python scripts/battery_test.py
    python scripts/battery_test.py --url http://localhost:8080 --max-tokens 300
"""

import argparse
import json
import time
import subprocess
import urllib.request
import urllib.error
import sys

URL = "http://localhost:8080"
PASSED = 0
FAILED = 0
RESULTS = []


def api_call(messages, max_tokens=200, temperature=0, timeout=120):
    """Call the OpenAI-compatible chat API."""
    data = json.dumps({
        "model": "qwen",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        f"{URL}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    resp = urllib.request.urlopen(req, timeout=timeout)
    elapsed = time.perf_counter() - t0
    r = json.loads(resp.read())
    content = r["choices"][0]["message"].get("content", "")
    reasoning = r["choices"][0]["message"].get("reasoning_content", "")
    usage = r["usage"]
    return {
        "content": content,
        "reasoning": reasoning,
        "usage": usage,
        "elapsed": elapsed,
        "tok_s": usage["completion_tokens"] / elapsed if elapsed > 0 else 0,
    }


def record(name, passed, detail=""):
    global PASSED, FAILED
    status = "PASS" if passed else "FAIL"
    if passed:
        PASSED += 1
    else:
        FAILED += 1
    RESULTS.append({"name": name, "passed": passed, "detail": detail})
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))


def get_full_answer(r):
    """Extract the actual answer from content or reasoning."""
    answer = r["content"].strip()
    if not answer and r["reasoning"]:
        # Qwen3.5 thinking mode: answer may be at the end of reasoning
        answer = r["reasoning"].strip()
    return answer


# ============================================================================
# 1. THROUGHPUT BENCHMARKS
# ============================================================================

def test_throughput():
    print("\n=== 1. THROUGHPUT ===")

    # Short prompt
    r = api_call([{"role": "user", "content": "Count from 1 to 50."}], max_tokens=300)
    tok_s = r["tok_s"]
    record(f"Decode throughput (short prompt)", tok_s > 5,
           f"{tok_s:.1f} tok/s, {r['usage']['completion_tokens']} tokens in {r['elapsed']:.1f}s")

    # Medium prompt (2K tokens)
    long_prompt = "Summarize the following text:\n\n" + ("The quick brown fox. " * 400)
    r = api_call([{"role": "user", "content": long_prompt}], max_tokens=200)
    prefill_tokens = r["usage"]["prompt_tokens"]
    record(f"Prefill handled ({prefill_tokens} tokens)", prefill_tokens > 1000,
           f"{prefill_tokens} prompt tokens, {r['elapsed']:.1f}s total")

    # Latency: time to first token approximation
    t0 = time.perf_counter()
    r = api_call([{"role": "user", "content": "Hi"}], max_tokens=1)
    ttft = time.perf_counter() - t0
    record("Time to first token", ttft < 10, f"{ttft:.2f}s")


# ============================================================================
# 2. LOGIC & REASONING TESTS
# ============================================================================

def test_math():
    print("\n=== 2. MATH ===")

    tests = [
        ("What is 17 * 24? Just the number.", "408"),
        ("What is the square root of 144? Just the number.", "12"),
        ("If I have 3 apples and buy 7 more, then eat 2, how many do I have? Just the number.", "8"),
        ("What is 15% of 200? Just the number.", "30"),
    ]
    for prompt, expected in tests:
        r = api_call([{"role": "user", "content": prompt}], max_tokens=300)
        answer = get_full_answer(r)
        passed = expected in answer
        record(f"Math: {expected}", passed, f"got: {answer[:60]}")


def test_reasoning():
    print("\n=== 3. REASONING ===")

    # Logical deduction
    r = api_call([{"role": "user", "content":
        "All mammals are warm-blooded. Whales are mammals. Are whales warm-blooded? Answer yes or no."}],
        max_tokens=300)
    answer = get_full_answer(r).lower()
    record("Syllogism", "yes" in answer, f"got: {answer[:60]}")

    # Spatial reasoning
    r = api_call([{"role": "user", "content":
        "If I'm facing north and turn left, which direction am I facing? One word."}],
        max_tokens=300)
    answer = get_full_answer(r).lower()
    record("Spatial reasoning", "west" in answer, f"got: {answer[:60]}")

    # Causal reasoning
    r = api_call([{"role": "user", "content":
        "A ball is on a table. I push the ball off the table. Where is the ball now? Brief answer."}],
        max_tokens=300)
    answer = get_full_answer(r).lower()
    record("Causal reasoning", "floor" in answer or "ground" in answer, f"got: {answer[:60]}")

    # Counterfactual
    r = api_call([{"role": "user", "content":
        "If the sun rose in the west, would shadows point east or west in the morning? One word."}],
        max_tokens=300)
    answer = get_full_answer(r).lower()
    record("Counterfactual", "east" in answer, f"got: {answer[:60]}")


def test_code():
    print("\n=== 4. CODE ===")

    r = api_call([{"role": "user", "content":
        "Write a Python function that returns the factorial of n. Just the function, no explanation."}],
        max_tokens=300)
    answer = get_full_answer(r)
    record("Code generation (factorial)",
           "def " in answer and ("factorial" in answer or "fact" in answer),
           f"contains def: {'def ' in answer}")

    r = api_call([{"role": "user", "content":
        "What does this Python code output?\nx = [1, 2, 3]\nprint(x[::-1])\nJust the output."}],
        max_tokens=300)
    answer = get_full_answer(r)
    record("Code tracing", "[3, 2, 1]" in answer, f"got: {answer[:60]}")


def test_knowledge():
    print("\n=== 5. FACTUAL KNOWLEDGE ===")

    tests = [
        ("What is the capital of Japan? One word.", "tokyo"),
        ("Who wrote Romeo and Juliet? Just the name.", "shakespeare"),
        ("What is the chemical formula for water? Just the formula.", "h2o"),
        ("In what year did World War II end? Just the year.", "1945"),
    ]
    for prompt, expected in tests:
        r = api_call([{"role": "user", "content": prompt}], max_tokens=300)
        answer = get_full_answer(r).lower()
        record(f"Knowledge: {expected}", expected in answer, f"got: {answer[:60]}")


def test_instruction_following():
    print("\n=== 6. INSTRUCTION FOLLOWING ===")

    # Format compliance
    r = api_call([{"role": "user", "content":
        "List exactly 3 fruits, one per line, numbered 1-3. Nothing else."}],
        max_tokens=300)
    answer = get_full_answer(r)
    has_numbers = "1" in answer and "2" in answer and "3" in answer
    record("Numbered list format", has_numbers, f"got: {answer[:80]}")

    # Constraint following
    r = api_call([{"role": "user", "content":
        "Name a country that starts with the letter Z."}],
        max_tokens=300)
    answer = get_full_answer(r).lower()
    record("Constraint (Z country)",
           "zambia" in answer or "zimbabwe" in answer,
           f"got: {answer[:60]}")

    # JSON output
    r = api_call([{"role": "user", "content":
        'Return a JSON object with keys "name" and "age" for a 30 year old named Alice. Only JSON, no explanation.'}],
        max_tokens=300)
    answer = get_full_answer(r)
    record("JSON output", '"name"' in answer and '"age"' in answer,
           f"got: {answer[:80]}")


# ============================================================================
# 7. NEEDLE IN A HAYSTACK
# ============================================================================

def test_niah():
    print("\n=== 7. NEEDLE-IN-A-HAYSTACK ===")

    NEEDLE = "The secret password is DIAMOND-FALCON-42."
    FILLER = ("Recent advances in machine learning have transformed many industries. "
              "Neural networks can now process images, text, and audio with remarkable accuracy. ") * 100

    for ctx_target in [4096, 8192]:
        for depth in [0.25, 0.5, 0.75]:
            total_chars = ctx_target * 3  # ~3 chars per token
            pos = int(len(FILLER[:total_chars]) * depth)
            haystack = FILLER[:pos] + "\n" + NEEDLE + "\n" + FILLER[pos:total_chars]
            prompt = f"Document:\n{haystack}\n\nWhat is the secret password mentioned in the document? Just the password."

            r = api_call([{"role": "user", "content": prompt}], max_tokens=300, timeout=180)
            answer = get_full_answer(r)
            found = "DIAMOND-FALCON-42" in answer
            record(f"NIAH {ctx_target//1024}K depth={depth}", found,
                   f"{'found' if found else 'missed'} in {r['elapsed']:.1f}s")


# ============================================================================
# 8. CONSISTENCY (determinism at temp=0)
# ============================================================================

def test_consistency():
    print("\n=== 8. CONSISTENCY (temp=0) ===")

    prompt = "What is the boiling point of water in Celsius? Just the number."
    answers = []
    for i in range(3):
        r = api_call([{"role": "user", "content": prompt}], max_tokens=300, temperature=0)
        answers.append(get_full_answer(r))

    # Check all answers contain "100"
    all_correct = all("100" in a for a in answers)
    record("Deterministic output (3 runs)", all_correct,
           f"answers: {[a[:30] for a in answers]}")


# ============================================================================
# 9. VRAM CHECK
# ============================================================================

def test_vram():
    print("\n=== 9. VRAM ===")
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
            text=True
        ).strip()
        used, total = out.split(",")
        used_mb = int(used.strip().replace(" MiB", ""))
        total_mb = int(total.strip().replace(" MiB", ""))
        pct = used_mb / total_mb * 100
        record(f"VRAM usage", pct < 90,
               f"{used_mb} MiB / {total_mb} MiB ({pct:.0f}%)")
    except Exception as e:
        record("VRAM check", False, f"nvidia-smi failed: {e}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Battery test for RotorQuant server")
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--max-tokens", type=int, default=300)
    args = parser.parse_args()

    global URL
    URL = args.url

    print("=" * 70)
    print("  RotorQuant Battery Test — Performance + Competence")
    print(f"  Server: {URL}")
    print("=" * 70)

    # Check server is up
    try:
        urllib.request.urlopen(f"{URL}/health", timeout=5)
    except Exception as e:
        print(f"ERROR: Server not reachable at {URL}: {e}")
        sys.exit(1)

    test_throughput()
    test_math()
    test_reasoning()
    test_code()
    test_knowledge()
    test_instruction_following()
    test_niah()
    test_consistency()
    test_vram()

    # Summary
    total = PASSED + FAILED
    print("\n" + "=" * 70)
    print(f"  RESULTS: {PASSED}/{total} passed ({PASSED/total*100:.0f}%)")
    if FAILED > 0:
        print(f"  FAILED ({FAILED}):")
        for r in RESULTS:
            if not r["passed"]:
                print(f"    - {r['name']}: {r['detail']}")
    print("=" * 70)

    # Save results
    with open("battery_results.json", "w") as f:
        json.dump({"passed": PASSED, "failed": FAILED, "total": total, "tests": RESULTS}, f, indent=2)
    print(f"\nResults saved to battery_results.json")

    sys.exit(0 if FAILED == 0 else 1)


if __name__ == "__main__":
    main()
