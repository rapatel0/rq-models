#!/usr/bin/env python3
"""
Comprehensive RotorQuant KV cache benchmark for llama.cpp server.

Tests multiple KV cache quantization configs (f16, iso3, planar3, turbo3, etc.)
on Qwen3.5-27B via the llama.cpp server's OpenAI-compatible API, measuring:
  - VRAM usage at various context lengths
  - Generation throughput (prefill + decode tok/s)
  - Perplexity (via llama-perplexity binary on wikitext-2)
  - Needle-In-A-Haystack recall at multiple context lengths
  - KV cache compression ratio (theoretical + measured)

Usage:
    python scripts/benchmark_rotorquant.py
    python scripts/benchmark_rotorquant.py --port 8090 --gpu-layers 99
    python scripts/benchmark_rotorquant.py --configs iso3/iso3 f16/f16 --skip-perplexity
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "/home/ravi/repos/turbo/models/Qwen3.5-27B-Q4_K_M.gguf"
DEFAULT_SERVER = "/home/ravi/repos/turbo/llama-cpp-rq/build/bin/llama-server"
DEFAULT_PERPLEXITY = "/home/ravi/repos/turbo/llama-cpp-rq/build/bin/llama-perplexity"
DEFAULT_PORT = 8080
DEFAULT_GPU_LAYERS = 99
DEFAULT_CTX_SIZE = 32768

WIKITEXT_URL = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet"
WIKITEXT_TXT_URL = "https://raw.githubusercontent.com/piskvorky/gensim/refs/heads/develop/gensim/test/test_data/wikitext-2-raw/wiki.test.raw"

# KV cache configurations: (name, cache_type_k, cache_type_v)
ALL_CONFIGS = [
    ("f16/f16",       "f16",     "f16"),
    ("iso3/iso3",     "iso3",    "iso3"),
    ("planar3/planar3", "planar3", "planar3"),
    ("turbo3/turbo3", "turbo3",  "turbo3"),
    ("iso3/f16",      "iso3",    "f16"),
]

# Bits per element for each cache type (theoretical)
# f16=16, turbo3/iso3/planar3: 14 bytes / 128 elems = 0.875 B/elem = 7 bits
# turbo4/iso4/planar4: 68 bytes / 128 elems = 4.25 bits
BITS_PER_ELEM = {
    "f16": 16.0,
    "f32": 32.0,
    "q8_0": 8.5,
    "q4_0": 4.5,
    "turbo3": 14.0 * 8 / 128,   # 0.875 bpe
    "iso3":  14.0 * 8 / 128,    # 0.875 bpe
    "planar3": 14.0 * 8 / 128,  # 0.875 bpe
    "turbo4": 68.0 * 8 / 128,   # 4.25 bpe
    "iso4":  68.0 * 8 / 128,    # 4.25 bpe
    "planar4": 68.0 * 8 / 128,  # 4.25 bpe
    "turbo2": 10.0 * 8 / 128,   # 0.625 bpe
}

CONTEXT_LENGTHS_VRAM = [1024, 4096, 8192, 16384]
CONTEXT_LENGTHS_NIAH = [4096, 8192, 16384]

NEEDLE_FACT = "The classified research codename is ZEPHYR-NINE-SEVEN-ALPHA."
NEEDLE_QUESTION = "What is the classified research codename mentioned in the document?"
NEEDLE_ANSWER_KEY = "ZEPHYR-NINE-SEVEN-ALPHA"

HAYSTACK_FILLER = (
    "Recent advances in computational linguistics have demonstrated that transformer-based "
    "architectures can achieve remarkable performance on a wide range of natural language "
    "processing tasks including summarization, translation, and question answering. "
    "Researchers continue to explore methods for improving efficiency and reducing the "
    "computational costs of these large-scale models while maintaining accuracy. "
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("benchmark")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def kill_existing_servers():
    """Kill any running llama-server processes."""
    try:
        subprocess.run(
            ["pkill", "-f", "llama-server"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        time.sleep(1)
    except Exception:
        pass


def get_gpu_memory_mib() -> float:
    """Return total GPU memory used in MiB via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            timeout=10,
            text=True,
        )
        # Sum across all GPUs (one line per GPU)
        total = sum(float(x.strip()) for x in out.strip().splitlines() if x.strip())
        return total
    except Exception as e:
        log.warning("nvidia-smi failed: %s", e)
        return -1.0


def wait_for_server(port: int, timeout: int = 120) -> bool:
    """Poll server /health until it responds OK or timeout."""
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                if data.get("status") == "ok":
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def api_chat_completion(port: int, messages: list, max_tokens: int = 128,
                        temperature: float = 0.0, timeout: int = 60) -> dict:
    """Send a chat completion request and return the full JSON response."""
    url = f"http://localhost:{port}/v1/chat/completions"
    payload = json.dumps({
        "model": "qwen",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def api_completions(port: int, prompt: str, max_tokens: int = 1,
                    temperature: float = 0.0, timeout: int = 60) -> dict:
    """Send a text completion request (used for prefill benchmarking)."""
    url = f"http://localhost:{port}/v1/completions"
    payload = json.dumps({
        "model": "qwen",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def build_long_prompt(target_tokens: int, needle: str = "", needle_depth: float = 0.5) -> str:
    """Build a long filler prompt, optionally embedding a needle at a given depth (0-1)."""
    # Rough estimate: ~1.3 tokens per word, ~15 words per filler sentence
    words_per_copy = len(HAYSTACK_FILLER.split())
    tokens_per_copy = int(words_per_copy * 1.3)
    n_copies = max(1, (target_tokens // tokens_per_copy) + 1)
    filler = HAYSTACK_FILLER * n_copies

    if needle:
        # Insert needle at approximate depth
        words = filler.split()
        insert_pos = int(len(words) * needle_depth)
        words.insert(insert_pos, f"\n{needle}\n")
        filler = " ".join(words)

    # Truncate to roughly target_tokens by word count
    words = filler.split()
    target_words = int(target_tokens / 1.3)
    if len(words) > target_words:
        words = words[:target_words]
    return " ".join(words)


def start_server(args, ctk: str, ctv: str, ctx_size: int = None) -> subprocess.Popen:
    """Start llama-server with given cache types. Returns Popen handle."""
    kill_existing_servers()
    time.sleep(1)

    cmd = [
        args.server_binary,
        "--model", args.model,
        "--port", str(args.port),
        "--n-gpu-layers", str(args.gpu_layers),
        "--ctx-size", str(ctx_size or args.ctx_size),
        "--cache-type-k", ctk,
        "--cache-type-v", ctv,
        "--flash-attn",
        "--log-disable",
    ]
    log.info("Starting server: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


def stop_server(proc: subprocess.Popen):
    """Gracefully stop a server process."""
    if proc is None:
        return
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass
    kill_existing_servers()


# ---------------------------------------------------------------------------
# Benchmark: VRAM Usage
# ---------------------------------------------------------------------------

def benchmark_vram(args, ctk: str, ctv: str) -> dict:
    """Measure VRAM at idle and after filling context to various lengths."""
    results = {}
    proc = None
    try:
        proc = start_server(args, ctk, ctv)
        if not wait_for_server(args.port, timeout=120):
            log.error("Server failed to start for VRAM benchmark (%s/%s)", ctk, ctv)
            return {"error": "server_start_failed"}

        time.sleep(3)
        idle_vram = get_gpu_memory_mib()
        results["idle_vram_mib"] = idle_vram
        results["context_vram"] = {}

        for ctx_len in CONTEXT_LENGTHS_VRAM:
            if ctx_len > args.ctx_size:
                continue
            try:
                log.info("  VRAM test: filling %d tokens (%s/%s)", ctx_len, ctk, ctv)
                prompt = build_long_prompt(ctx_len)
                api_completions(args.port, prompt, max_tokens=1, timeout=120)
                time.sleep(2)
                used_vram = get_gpu_memory_mib()
                delta = used_vram - idle_vram
                results["context_vram"][str(ctx_len)] = {
                    "total_vram_mib": used_vram,
                    "delta_vram_mib": delta,
                }
                log.info("    ctx=%d -> delta=%.1f MiB (total=%.1f MiB)", ctx_len, delta, used_vram)
            except Exception as e:
                log.warning("    ctx=%d VRAM test failed: %s", ctx_len, e)
                results["context_vram"][str(ctx_len)] = {"error": str(e)}

    except Exception as e:
        log.error("VRAM benchmark error: %s", e)
        results["error"] = str(e)
    finally:
        stop_server(proc)

    return results


# ---------------------------------------------------------------------------
# Benchmark: Throughput
# ---------------------------------------------------------------------------

def benchmark_throughput(args, ctk: str, ctv: str) -> dict:
    """Measure prefill and decode throughput via the completions API."""
    results = {}
    proc = None
    try:
        proc = start_server(args, ctk, ctv)
        if not wait_for_server(args.port, timeout=120):
            log.error("Server failed to start for throughput benchmark (%s/%s)", ctk, ctv)
            return {"error": "server_start_failed"}

        time.sleep(2)

        # Warmup
        log.info("  Throughput warmup...")
        try:
            api_chat_completion(args.port, [{"role": "user", "content": "Hello"}], max_tokens=16, timeout=60)
        except Exception as e:
            log.warning("  Warmup failed: %s", e)

        # --- Prefill benchmark (long prompt, 1 output token) ---
        for prompt_len in [512, 2048, 4096]:
            if prompt_len > args.ctx_size - 64:
                continue
            try:
                log.info("  Prefill test: %d tokens (%s/%s)", prompt_len, ctk, ctv)
                prompt_text = build_long_prompt(prompt_len)

                t0 = time.perf_counter()
                resp = api_completions(args.port, prompt_text, max_tokens=1, timeout=120)
                t1 = time.perf_counter()

                usage = resp.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", prompt_len)
                wall_time = t1 - t0
                prefill_tps = prompt_tokens / wall_time if wall_time > 0 else 0

                results[f"prefill_{prompt_len}"] = {
                    "prompt_tokens": prompt_tokens,
                    "wall_time_s": round(wall_time, 3),
                    "tokens_per_sec": round(prefill_tps, 1),
                }
                log.info("    prefill %d tok in %.2fs = %.1f tok/s", prompt_tokens, wall_time, prefill_tps)
            except Exception as e:
                log.warning("    Prefill %d failed: %s", prompt_len, e)
                results[f"prefill_{prompt_len}"] = {"error": str(e)}

        # --- Decode benchmark (short prompt, many output tokens) ---
        for gen_tokens in [64, 256]:
            try:
                log.info("  Decode test: generate %d tokens (%s/%s)", gen_tokens, ctk, ctv)
                t0 = time.perf_counter()
                resp = api_chat_completion(
                    args.port,
                    [{"role": "user", "content": "Write a detailed essay about the history of computing."}],
                    max_tokens=gen_tokens,
                    timeout=120,
                )
                t1 = time.perf_counter()

                usage = resp.get("usage", {})
                completion_tokens = usage.get("completion_tokens", gen_tokens)
                wall_time = t1 - t0
                decode_tps = completion_tokens / wall_time if wall_time > 0 else 0

                results[f"decode_{gen_tokens}"] = {
                    "completion_tokens": completion_tokens,
                    "wall_time_s": round(wall_time, 3),
                    "tokens_per_sec": round(decode_tps, 1),
                }
                log.info("    decode %d tok in %.2fs = %.1f tok/s", completion_tokens, wall_time, decode_tps)
            except Exception as e:
                log.warning("    Decode %d failed: %s", gen_tokens, e)
                results[f"decode_{gen_tokens}"] = {"error": str(e)}

    except Exception as e:
        log.error("Throughput benchmark error: %s", e)
        results["error"] = str(e)
    finally:
        stop_server(proc)

    return results


# ---------------------------------------------------------------------------
# Benchmark: Perplexity
# ---------------------------------------------------------------------------

def ensure_wikitext(data_dir: str) -> str:
    """Download wikitext-2 test set if not present. Returns path to text file."""
    os.makedirs(data_dir, exist_ok=True)
    txt_path = os.path.join(data_dir, "wikitext-2-test.txt")
    if os.path.exists(txt_path) and os.path.getsize(txt_path) > 1000:
        return txt_path

    log.info("Downloading wikitext-2 test set...")
    try:
        urllib.request.urlretrieve(WIKITEXT_TXT_URL, txt_path)
        if os.path.getsize(txt_path) > 1000:
            log.info("Downloaded wikitext-2 to %s (%d bytes)", txt_path, os.path.getsize(txt_path))
            return txt_path
    except Exception as e:
        log.warning("Direct wikitext download failed: %s. Creating synthetic test data.", e)

    # Fallback: create a synthetic test file from repeated natural text
    log.info("Creating synthetic test corpus for perplexity measurement...")
    corpus = (
        "The history of natural language processing generally started in the 1950s, "
        "although work can be found from earlier periods. In 1950, Alan Turing published "
        "an article titled Computing Machinery and Intelligence which proposed what is now "
        "called the Turing test as a criterion of intelligence. The Georgetown experiment "
        "in 1954 involved fully automatic translation of more than sixty Russian sentences "
        "into English. The authors claimed that within three or five years, machine "
        "translation would be a solved problem. However, real progress was much slower, "
        "and after the ALPAC report in 1966, which found that ten years of research had "
        "failed to fulfill the expectations, funding for machine translation was "
        "dramatically reduced. Little further research in machine translation was "
        "conducted until the late 1980s when the first statistical machine translation "
        "systems were developed.\n\n"
    ) * 500
    with open(txt_path, "w") as f:
        f.write(corpus)
    return txt_path


def benchmark_perplexity(args, ctk: str, ctv: str) -> dict:
    """Run llama-perplexity on wikitext-2 and parse the result."""
    data_dir = os.path.join(os.path.dirname(args.model), ".benchmark_data")
    txt_path = ensure_wikitext(data_dir)

    cmd = [
        args.perplexity_binary,
        "--model", args.model,
        "--file", txt_path,
        "--n-gpu-layers", str(args.gpu_layers),
        "--ctx-size", "2048",
        "--cache-type-k", ctk,
        "--cache-type-v", ctv,
        "--flash-attn",
    ]
    log.info("  Running perplexity: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min max
        )
        output = result.stdout + "\n" + result.stderr

        # Parse perplexity from output (typical format: "Final estimate: PPL = 6.1234")
        ppl = None
        for line in output.splitlines():
            line_lower = line.lower()
            # Match various llama-perplexity output formats
            if "final estimate" in line_lower and "ppl" in line_lower:
                parts = line.split("=")
                if len(parts) >= 2:
                    try:
                        ppl = float(parts[-1].strip().split()[0].rstrip(","))
                    except ValueError:
                        pass
            elif "perplexity" in line_lower and "=" in line:
                parts = line.split("=")
                if len(parts) >= 2:
                    try:
                        val = float(parts[-1].strip().split()[0].rstrip(","))
                        if ppl is None or val > 0:
                            ppl = val
                    except ValueError:
                        pass

        if ppl is not None:
            log.info("    Perplexity: %.4f (%s/%s)", ppl, ctk, ctv)
            return {"perplexity": round(ppl, 4), "raw_output_tail": output[-500:]}
        else:
            log.warning("    Could not parse perplexity from output")
            return {"error": "parse_failed", "raw_output_tail": output[-1000:]}

    except subprocess.TimeoutExpired:
        log.warning("    Perplexity timed out (%s/%s)", ctk, ctv)
        return {"error": "timeout"}
    except FileNotFoundError:
        log.error("    llama-perplexity binary not found at %s", args.perplexity_binary)
        return {"error": "binary_not_found"}
    except Exception as e:
        log.error("    Perplexity error: %s", e)
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Benchmark: Needle-In-A-Haystack
# ---------------------------------------------------------------------------

def benchmark_niah(args, ctk: str, ctv: str) -> dict:
    """Test long-context recall at multiple context lengths and depths."""
    results = {}
    proc = None
    try:
        proc = start_server(args, ctk, ctv)
        if not wait_for_server(args.port, timeout=120):
            log.error("Server failed to start for NIAH benchmark (%s/%s)", ctk, ctv)
            return {"error": "server_start_failed"}

        time.sleep(2)

        # Warmup
        try:
            api_chat_completion(args.port, [{"role": "user", "content": "Hello"}], max_tokens=8, timeout=30)
        except Exception:
            pass

        depths = [0.1, 0.25, 0.5, 0.75, 0.9]

        for ctx_len in CONTEXT_LENGTHS_NIAH:
            if ctx_len > args.ctx_size:
                continue
            ctx_results = []
            log.info("  NIAH test: ctx=%d (%s/%s)", ctx_len, ctk, ctv)

            for depth in depths:
                try:
                    # Build document with needle
                    doc = build_long_prompt(
                        ctx_len - 200,  # leave room for system/question tokens
                        needle=NEEDLE_FACT,
                        needle_depth=depth,
                    )
                    messages = [
                        {"role": "system", "content": "Answer the question based only on the provided document. Be concise."},
                        {"role": "user", "content": f"Document:\n{doc}\n\nQuestion: {NEEDLE_QUESTION}"},
                    ]

                    resp = api_chat_completion(
                        args.port, messages, max_tokens=64, timeout=120,
                    )
                    answer = resp["choices"][0]["message"]["content"]
                    hit = NEEDLE_ANSWER_KEY.lower() in answer.lower()
                    ctx_results.append({
                        "depth": depth,
                        "hit": hit,
                        "answer_snippet": answer[:200],
                    })
                    log.info("    depth=%.2f hit=%s answer=%s", depth, hit, answer[:80])

                except Exception as e:
                    log.warning("    depth=%.2f failed: %s", depth, e)
                    ctx_results.append({"depth": depth, "hit": False, "error": str(e)})

            hits = sum(1 for r in ctx_results if r.get("hit"))
            recall = hits / len(ctx_results) if ctx_results else 0.0
            results[str(ctx_len)] = {
                "recall": round(recall, 2),
                "hits": hits,
                "total": len(ctx_results),
                "details": ctx_results,
            }
            log.info("    ctx=%d recall=%.0f%% (%d/%d)", ctx_len, recall * 100, hits, len(ctx_results))

    except Exception as e:
        log.error("NIAH benchmark error: %s", e)
        results["error"] = str(e)
    finally:
        stop_server(proc)

    return results


# ---------------------------------------------------------------------------
# Compression Ratio
# ---------------------------------------------------------------------------

def compute_compression_ratio(ctk: str, ctv: str) -> dict:
    """Compute theoretical compression ratio for a KV cache config."""
    bpe_k = BITS_PER_ELEM.get(ctk, 16.0)
    bpe_v = BITS_PER_ELEM.get(ctv, 16.0)
    bpe_f16 = 16.0
    avg_bpe = (bpe_k + bpe_v) / 2.0
    baseline_bpe = bpe_f16  # f16 for both K and V
    ratio = baseline_bpe / avg_bpe if avg_bpe > 0 else 1.0

    # Qwen3.5-27B: 64 layers, 8 KV heads, head_dim=128
    num_layers = 64
    num_kv_heads = 8
    head_dim = 128
    bytes_per_token_f16 = num_layers * num_kv_heads * head_dim * 2 * 2  # K+V, 2 bytes each
    bytes_per_token_actual = num_layers * num_kv_heads * head_dim * (bpe_k + bpe_v) / 8

    return {
        "bits_per_elem_k": round(bpe_k, 2),
        "bits_per_elem_v": round(bpe_v, 2),
        "avg_bits_per_elem": round(avg_bpe, 2),
        "compression_ratio": round(ratio, 2),
        "bytes_per_token_f16": bytes_per_token_f16,
        "bytes_per_token_actual": round(bytes_per_token_actual, 1),
        "mib_per_1k_tokens_f16": round(bytes_per_token_f16 * 1024 / (1024 * 1024), 2),
        "mib_per_1k_tokens_actual": round(bytes_per_token_actual * 1024 / (1024 * 1024), 2),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(all_results: dict):
    """Print a clean comparison table to stdout."""
    configs = list(all_results.keys())
    if not configs:
        log.warning("No results to report.")
        return

    sep = "=" * 100
    print(f"\n{sep}")
    print("  ROTORQUANT KV CACHE BENCHMARK RESULTS")
    print(f"  Model: {all_results[configs[0]].get('model', 'N/A')}")
    print(f"  Date:  {all_results[configs[0]].get('timestamp', 'N/A')}")
    print(sep)

    # --- Compression Ratio Table ---
    print("\n  [1] COMPRESSION RATIO (theoretical)")
    print("  " + "-" * 80)
    print(f"  {'Config':<18} {'K bpe':>8} {'V bpe':>8} {'Avg bpe':>8} {'Ratio':>8} {'MiB/1k tok':>12}")
    print("  " + "-" * 80)
    for cfg in configs:
        cr = all_results[cfg].get("compression_ratio", {})
        if "error" in cr:
            print(f"  {cfg:<18} {'ERROR':>8}")
            continue
        print(f"  {cfg:<18} {cr.get('bits_per_elem_k', 0):>8.2f} {cr.get('bits_per_elem_v', 0):>8.2f} "
              f"{cr.get('avg_bits_per_elem', 0):>8.2f} {cr.get('compression_ratio', 0):>7.2f}x "
              f"{cr.get('mib_per_1k_tokens_actual', 0):>11.2f}")

    # --- VRAM Table ---
    print("\n  [2] VRAM USAGE (MiB delta from idle)")
    print("  " + "-" * 80)
    header = f"  {'Config':<18} {'Idle':>8}"
    for cl in CONTEXT_LENGTHS_VRAM:
        header += f" {'ctx=' + str(cl):>12}"
    print(header)
    print("  " + "-" * 80)
    for cfg in configs:
        vram = all_results[cfg].get("vram", {})
        if "error" in vram:
            print(f"  {cfg:<18} {'ERROR':>8}")
            continue
        row = f"  {cfg:<18} {vram.get('idle_vram_mib', -1):>8.0f}"
        cv = vram.get("context_vram", {})
        for cl in CONTEXT_LENGTHS_VRAM:
            entry = cv.get(str(cl), {})
            delta = entry.get("delta_vram_mib", None)
            if delta is not None:
                row += f" {delta:>11.0f}M"
            else:
                row += f" {'N/A':>12}"
        print(row)

    # --- Throughput Table ---
    print("\n  [3] THROUGHPUT (tokens/sec)")
    print("  " + "-" * 80)
    print(f"  {'Config':<18} {'Prefill 512':>12} {'Prefill 2k':>12} {'Prefill 4k':>12} {'Decode 64':>12} {'Decode 256':>12}")
    print("  " + "-" * 80)
    for cfg in configs:
        tp = all_results[cfg].get("throughput", {})
        if "error" in tp:
            print(f"  {cfg:<18} {'ERROR':>12}")
            continue
        vals = []
        for key in ["prefill_512", "prefill_2048", "prefill_4096", "decode_64", "decode_256"]:
            entry = tp.get(key, {})
            tps = entry.get("tokens_per_sec")
            vals.append(f"{tps:>11.1f}" if tps is not None else f"{'N/A':>11}")
        print(f"  {cfg:<18} " + " ".join(vals))

    # --- Perplexity Table ---
    print("\n  [4] PERPLEXITY (wikitext-2, lower is better)")
    print("  " + "-" * 80)
    print(f"  {'Config':<18} {'Perplexity':>12} {'Delta vs f16':>14}")
    print("  " + "-" * 80)
    baseline_ppl = None
    for cfg in configs:
        ppl_data = all_results[cfg].get("perplexity", {})
        ppl = ppl_data.get("perplexity")
        if cfg == "f16/f16" and ppl is not None:
            baseline_ppl = ppl

    for cfg in configs:
        ppl_data = all_results[cfg].get("perplexity", {})
        ppl = ppl_data.get("perplexity")
        if ppl is not None:
            delta = f"+{ppl - baseline_ppl:.4f}" if baseline_ppl is not None else "N/A"
            if cfg == "f16/f16":
                delta = "(baseline)"
            print(f"  {cfg:<18} {ppl:>12.4f} {delta:>14}")
        else:
            err = ppl_data.get("error", "skipped")
            print(f"  {cfg:<18} {'N/A':>12} {err:>14}")

    # --- NIAH Table ---
    print("\n  [5] NEEDLE-IN-A-HAYSTACK RECALL")
    print("  " + "-" * 80)
    header = f"  {'Config':<18}"
    for cl in CONTEXT_LENGTHS_NIAH:
        header += f" {'ctx=' + str(cl):>12}"
    print(header)
    print("  " + "-" * 80)
    for cfg in configs:
        niah = all_results[cfg].get("niah", {})
        if "error" in niah:
            print(f"  {cfg:<18} {'ERROR':>12}")
            continue
        row = f"  {cfg:<18}"
        for cl in CONTEXT_LENGTHS_NIAH:
            entry = niah.get(str(cl), {})
            recall = entry.get("recall")
            if recall is not None:
                hits = entry.get("hits", 0)
                total = entry.get("total", 0)
                row += f" {recall * 100:>7.0f}% ({hits}/{total})"
            else:
                row += f" {'N/A':>12}"
        print(row)

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Comprehensive RotorQuant KV cache benchmark for llama.cpp server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Path to GGUF model file")
    parser.add_argument("--server-binary", default=DEFAULT_SERVER,
                        help="Path to llama-server binary")
    parser.add_argument("--perplexity-binary", default=DEFAULT_PERPLEXITY,
                        help="Path to llama-perplexity binary")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help="Port for llama-server")
    parser.add_argument("--gpu-layers", type=int, default=DEFAULT_GPU_LAYERS,
                        help="Number of layers to offload to GPU")
    parser.add_argument("--ctx-size", type=int, default=DEFAULT_CTX_SIZE,
                        help="Context size for the server")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Specific configs to test (e.g., 'iso3/iso3' 'f16/f16')")
    parser.add_argument("--skip-vram", action="store_true",
                        help="Skip VRAM measurement")
    parser.add_argument("--skip-throughput", action="store_true",
                        help="Skip throughput measurement")
    parser.add_argument("--skip-perplexity", action="store_true",
                        help="Skip perplexity measurement")
    parser.add_argument("--skip-niah", action="store_true",
                        help="Skip NIAH measurement")
    parser.add_argument("--output", default=None,
                        help="Output JSON file path (default: auto-generated)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate paths
    if not os.path.exists(args.model):
        log.error("Model not found: %s", args.model)
        sys.exit(1)
    if not os.path.exists(args.server_binary):
        log.warning("Server binary not found: %s (build it first)", args.server_binary)
    if not os.path.exists(args.perplexity_binary):
        log.warning("Perplexity binary not found: %s (perplexity tests will fail)", args.perplexity_binary)

    # Filter configs if specified
    if args.configs:
        configs = []
        for c in args.configs:
            parts = c.split("/")
            if len(parts) == 2:
                configs.append((c, parts[0], parts[1]))
            else:
                log.error("Invalid config format: %s (expected 'ctk/ctv')", c)
                sys.exit(1)
    else:
        configs = ALL_CONFIGS

    log.info("Benchmarking %d configs: %s", len(configs), [c[0] for c in configs])
    log.info("Model: %s", args.model)
    log.info("Context size: %d", args.ctx_size)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    all_results = {}

    for name, ctk, ctv in configs:
        log.info("\n%s Running config: %s %s", "=" * 30, name, "=" * 30)
        result = {
            "model": args.model,
            "timestamp": timestamp,
            "cache_type_k": ctk,
            "cache_type_v": ctv,
        }

        # Compression ratio (always computed, no server needed)
        result["compression_ratio"] = compute_compression_ratio(ctk, ctv)

        # VRAM
        if not args.skip_vram:
            log.info("[VRAM] %s", name)
            result["vram"] = benchmark_vram(args, ctk, ctv)
        else:
            result["vram"] = {"skipped": True}

        # Throughput
        if not args.skip_throughput:
            log.info("[THROUGHPUT] %s", name)
            result["throughput"] = benchmark_throughput(args, ctk, ctv)
        else:
            result["throughput"] = {"skipped": True}

        # Perplexity
        if not args.skip_perplexity:
            log.info("[PERPLEXITY] %s", name)
            kill_existing_servers()
            result["perplexity"] = benchmark_perplexity(args, ctk, ctv)
        else:
            result["perplexity"] = {"skipped": True}

        # NIAH
        if not args.skip_niah:
            log.info("[NIAH] %s", name)
            result["niah"] = benchmark_niah(args, ctk, ctv)
        else:
            result["niah"] = {"skipped": True}

        all_results[name] = result

    # Ensure server is stopped
    kill_existing_servers()

    # Print report
    print_report(all_results)

    # Save JSON results
    if args.output:
        out_path = args.output
    else:
        out_dir = os.path.join(os.path.dirname(args.model), ".benchmark_data")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"benchmark_{timestamp}.json")

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info("Results written to %s", out_path)

    return all_results


if __name__ == "__main__":
    main()
