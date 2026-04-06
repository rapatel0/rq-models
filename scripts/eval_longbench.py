#!/usr/bin/env python3
"""
LongBench-E evaluation for TurboQuant KV cache.

Evaluates SingleQA and Summarization tasks from LongBench-E dataset.

Usage:
    python scripts/eval_longbench.py --model Qwen/Qwen3.5-27B --bits 3.5
    python scripts/eval_longbench.py --model Qwen/Qwen3.5-27B --bits 3.5 --tasks narrativeqa qasper
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from turboquant import PRESET_2_5BIT, PRESET_3_5BIT, TurboKVCache

PRESETS = {"2.5": PRESET_2_5BIT, "3.5": PRESET_3_5BIT, "none": None}

# LongBench-E tasks: (dataset_name, subset, max_gen_tokens)
TASKS = {
    # SingleQA
    "narrativeqa":   ("THUDM/LongBench", "narrativeqa_e",  32),
    "qasper":        ("THUDM/LongBench", "qasper_e",       32),
    "multifieldqa":  ("THUDM/LongBench", "multifieldqa_en_e", 64),
    # Summarization
    "gov_report":    ("THUDM/LongBench", "gov_report_e",   512),
    "qmsum":         ("THUDM/LongBench", "qmsum_e",        512),
}

DEFAULT_TASKS = ["narrativeqa", "qasper", "gov_report"]


def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 score (standard for QA tasks)."""
    pred_tokens = set(prediction.lower().split())
    truth_tokens = set(ground_truth.lower().split())
    if not pred_tokens or not truth_tokens:
        return 0.0
    common = pred_tokens & truth_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, ground_truth: str) -> float:
    """Simplified ROUGE-L (longest common subsequence / len(truth))."""
    pred = prediction.lower().split()
    truth = ground_truth.lower().split()
    if not truth:
        return 0.0
    m, n = len(pred), len(truth)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i - 1] == truth[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    return lcs / n


def evaluate_task(model, tokenizer, task_name: str, config, max_samples: int = 100) -> float:
    """Evaluate a single LongBench-E task. Returns score in [0, 1]."""
    ds_name, subset, max_gen = TASKS[task_name]

    try:
        dataset = load_dataset(ds_name, subset, split="test")
    except Exception as e:
        print(f"  Could not load {task_name}: {e}")
        return float("nan")

    scores = []
    is_summarization = "gov_report" in task_name or "qmsum" in task_name

    for i, sample in enumerate(dataset):
        if i >= max_samples:
            break
        prompt = sample.get("input", sample.get("context", ""))
        answers = sample.get("answers", [sample.get("output", "")])
        if isinstance(answers, str):
            answers = [answers]

        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=32768
        ).to(model.device)

        past_kv = TurboKVCache(config) if config is not None else None

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                past_key_values=past_kv,
                max_new_tokens=max_gen,
                do_sample=False,
            )

        new_ids = output_ids[0, inputs["input_ids"].shape[-1]:]
        prediction = tokenizer.decode(new_ids, skip_special_tokens=True)

        if is_summarization:
            score = max(rouge_l(prediction, a) for a in answers)
        else:
            score = max(f1_score(prediction, a) for a in answers)
        scores.append(score)

    return sum(scores) / len(scores) if scores else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--bits", choices=["2.5", "3.5", "none"], default="3.5")
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS.keys()), default=DEFAULT_TASKS)
    parser.add_argument("--max-samples", type=int, default=100)
    args = parser.parse_args()

    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    config = PRESETS[args.bits]
    print(f"\nLongBench-E evaluation ({args.bits}-bit TurboQuant):")

    results = {}
    for task in args.tasks:
        print(f"  {task}...", end=" ", flush=True)
        score = evaluate_task(model, tokenizer, task, config, args.max_samples)
        results[task] = score
        print(f"{score:.3f}")

    valid = [v for v in results.values() if v == v]  # filter NaN
    avg = sum(valid) / len(valid) if valid else float("nan")
    print(f"\nAverage score: {avg:.3f}")
    print(f"Results: {json.dumps({k: round(v, 3) for k, v in results.items()}, indent=2)}")


if __name__ == "__main__":
    main()
