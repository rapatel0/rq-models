"""
Shared calibration and evaluation corpus loader.

Single source of truth for dataset loading across all quantization methods:
  - SpectralQuant calibration (calibrate_spectral.py)
  - RotorQuant calibration    (rotorquant/turboquant/calibrate.py)
  - Perplexity evaluation     (eval_perplexity.py, benchmark_spectral.py)

Calibration standard (matches GPTQ / AWQ):
  - Dataset:  allenai/c4, English, train split
  - Samples:  128 sequences × 2048 tokens each
  - Seed:     42 (reproducible across methods)

Evaluation standard:
  - Dataset:  wikitext-2-raw-v1, test split (industry-standard PPL benchmark)
  - Sliding-window PPL with stride 512, max_length 2048

Usage:
    from turboquant.corpus import load_calibration_tokens, load_eval_text

    # Calibration — returns list[Tensor[seq_len]] (int64 token ids)
    seqs = load_calibration_tokens(tokenizer)

    # Evaluation — returns a single long string for sliding-window PPL
    text = load_eval_text()
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    pass  # avoid importing transformers at module level

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constants — override via function args if needed
# ---------------------------------------------------------------------------

CALIBRATION_DATASET   = "allenai/c4"
CALIBRATION_CONFIG    = "en"
CALIBRATION_SPLIT     = "train"
CALIBRATION_N_SAMPLES = 128
CALIBRATION_SEQ_LEN   = 2048
CALIBRATION_SEED      = 42

EVAL_DATASET  = "wikitext"
EVAL_CONFIG   = "wikitext-2-raw-v1"
EVAL_SPLIT    = "test"


# ---------------------------------------------------------------------------
# Calibration corpus
# ---------------------------------------------------------------------------

def load_calibration_tokens(
    tokenizer,
    n_samples: int = CALIBRATION_N_SAMPLES,
    seq_len: int = CALIBRATION_SEQ_LEN,
    dataset: str = CALIBRATION_DATASET,
    config: str = CALIBRATION_CONFIG,
    split: str = CALIBRATION_SPLIT,
    seed: int = CALIBRATION_SEED,
) -> list[Tensor]:
    """
    Load calibration corpus and return tokenized sequences.

    Streams from C4 (default) to avoid downloading the full dataset.
    Each returned tensor is exactly [seq_len] int64 token ids.

    Args:
        tokenizer: HuggingFace tokenizer
        n_samples:  number of sequences to return (default 128)
        seq_len:    tokens per sequence (default 2048)
        dataset:    HuggingFace dataset name (default "allenai/c4")
        config:     dataset config/subset (default "en")
        split:      dataset split (default "train")
        seed:       random seed for reproducibility (default 42)

    Returns:
        list of Tensors, each shape [seq_len], dtype=int64
    """
    from datasets import load_dataset

    logger.info(
        f"Loading calibration corpus: {dataset}/{config} {split} "
        f"({n_samples} samples × {seq_len} tokens, seed={seed})"
    )

    rng = random.Random(seed)

    if dataset == "allenai/c4":
        return _load_c4_calibration(tokenizer, n_samples, seq_len, config, split, seed)
    elif dataset in ("wikitext", "wikitext-2-raw-v1"):
        return _load_wikitext_calibration(tokenizer, n_samples, seq_len, split, seed)
    else:
        # Generic: stream and tokenize, concatenate until we have enough tokens
        return _load_generic_calibration(tokenizer, n_samples, seq_len, dataset, config, split, seed)


def load_calibration_texts(
    n_samples: int = CALIBRATION_N_SAMPLES,
    dataset: str = CALIBRATION_DATASET,
    config: str = CALIBRATION_CONFIG,
    split: str = CALIBRATION_SPLIT,
    seed: int = CALIBRATION_SEED,
) -> list[str]:
    """
    Load calibration corpus as raw text strings (no tokenizer required).

    Returns list of text samples. Useful for SpectralCalibrator which
    tokenizes internally.

    Args:
        n_samples: number of text samples (default 128; each is a C4 article)
        dataset:   HuggingFace dataset name
        config:    dataset config/subset
        split:     dataset split
        seed:      random seed

    Returns:
        list of str
    """
    from datasets import load_dataset

    logger.info(
        f"Loading calibration texts: {dataset}/{config} {split} "
        f"({n_samples} samples, seed={seed})"
    )

    if dataset == "allenai/c4":
        ds = load_dataset(dataset, config, split=split, streaming=True)
        # Shuffle with buffer to get diverse samples
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
        texts = []
        for sample in ds:
            text = sample["text"].strip()
            if len(text) > 200:  # skip very short samples
                texts.append(text)
            if len(texts) >= n_samples:
                break
        logger.info(f"Collected {len(texts)} calibration texts from C4")
        return texts

    elif dataset in ("wikitext", "wikitext-2-raw-v1"):
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        texts = [t for t in ds["text"] if len(t.strip()) > 100]
        rng = random.Random(seed)
        rng.shuffle(texts)
        return texts[:n_samples]

    else:
        ds = load_dataset(dataset, config, split=split, streaming=True)
        texts = []
        for sample in ds:
            # Try common text field names
            text = sample.get("text") or sample.get("content") or sample.get("passage") or ""
            text = text.strip()
            if len(text) > 200:
                texts.append(text)
            if len(texts) >= n_samples:
                break
        return texts


# ---------------------------------------------------------------------------
# Evaluation corpus
# ---------------------------------------------------------------------------

def load_eval_text(
    dataset: str = EVAL_DATASET,
    config: str = EVAL_CONFIG,
    split: str = EVAL_SPLIT,
) -> str:
    """
    Load evaluation corpus as a single concatenated string for sliding-window PPL.

    Default: wikitext-2 test (industry standard for LLM perplexity comparison).

    Args:
        dataset: HuggingFace dataset name
        config:  dataset config/subset
        split:   dataset split

    Returns:
        str — concatenated text ready for tokenization
    """
    from datasets import load_dataset

    logger.info(f"Loading eval corpus: {dataset}/{config} {split}")

    if dataset == "wikitext" or dataset == "wikitext-2-raw-v1":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        text = "\n\n".join(ds["text"])
    elif dataset == "allenai/c4":
        # For C4 eval: use first 1000 validation samples concatenated
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        texts = []
        for sample in ds:
            texts.append(sample["text"])
            if len(texts) >= 1000:
                break
        text = "\n\n".join(texts)
    else:
        ds = load_dataset(dataset, config, split=split)
        text = "\n\n".join(ds["text"])

    logger.info(f"Eval corpus: {len(text):,} characters")
    return text


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_c4_calibration(
    tokenizer, n_samples: int, seq_len: int, config: str, split: str, seed: int
) -> list[Tensor]:
    """Stream C4 and return n_samples of exactly seq_len tokens each."""
    from datasets import load_dataset

    ds = load_dataset("allenai/c4", config, split=split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    sequences = []
    token_buffer: list[int] = []

    for sample in ds:
        text = sample["text"]
        ids = tokenizer(text, add_special_tokens=False).input_ids
        token_buffer.extend(ids)

        # Emit complete sequences
        while len(token_buffer) >= seq_len:
            sequences.append(torch.tensor(token_buffer[:seq_len], dtype=torch.int64))
            token_buffer = token_buffer[seq_len:]
            if len(sequences) >= n_samples:
                break

        if len(sequences) >= n_samples:
            break

    if len(sequences) < n_samples:
        logger.warning(
            f"Only collected {len(sequences)}/{n_samples} sequences from C4 "
            f"(stream exhausted early)"
        )

    logger.info(f"Collected {len(sequences)} calibration sequences ({seq_len} tokens each)")
    return sequences


def _load_wikitext_calibration(
    tokenizer, n_samples: int, seq_len: int, split: str, seed: int
) -> list[Tensor]:
    """Load wikitext-2 train split, tokenize, chunk into seq_len blocks."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(ds["text"])
    all_ids = tokenizer(text, add_special_tokens=False).input_ids

    # Chunk into seq_len blocks
    all_ids_t = torch.tensor(all_ids, dtype=torch.int64)
    n_chunks = len(all_ids_t) // seq_len
    chunks = [all_ids_t[i * seq_len:(i + 1) * seq_len] for i in range(n_chunks)]

    rng = random.Random(seed)
    rng.shuffle(chunks)
    selected = chunks[:n_samples]

    logger.info(
        f"Wikitext-2 {split}: {len(all_ids):,} tokens → "
        f"{n_chunks} chunks, using {len(selected)}"
    )
    return selected


def _load_generic_calibration(
    tokenizer, n_samples: int, seq_len: int,
    dataset: str, config: str, split: str, seed: int,
) -> list[Tensor]:
    """Generic streaming loader: concatenate tokens until n_samples × seq_len."""
    from datasets import load_dataset

    try:
        ds = load_dataset(dataset, config, split=split, streaming=True)
    except Exception:
        ds = load_dataset(dataset, split=split, streaming=True)

    sequences = []
    token_buffer: list[int] = []

    for sample in ds:
        text = (
            sample.get("text") or sample.get("content") or
            sample.get("passage") or sample.get("output") or ""
        )
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False).input_ids
        token_buffer.extend(ids)
        while len(token_buffer) >= seq_len:
            sequences.append(torch.tensor(token_buffer[:seq_len], dtype=torch.int64))
            token_buffer = token_buffer[seq_len:]
            if len(sequences) >= n_samples:
                break
        if len(sequences) >= n_samples:
            break

    return sequences
