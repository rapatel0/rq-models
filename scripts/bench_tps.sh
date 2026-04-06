#!/bin/bash
# Token/s benchmark: RotorQuant iso3/iso3 on Qwen3.5-27B at escalating context
# Tests both prefill (prompt eval) and decode (generation) throughput

LLAMA_CLI="/home/ravi/repos/turbo/llama-cpp-rq/build/bin/llama-cli"
MODEL="/home/ravi/repos/turbo/models/Qwen3.5-27B-Q4_K_M.gguf"
RESULTS="/home/ravi/repos/turbo/bench_results.txt"

# Contexts to test: push up to 128K+
CONTEXTS=(512 2048 4096 8192 16384 32768 65536 131072)

# Cache configs to compare
declare -A CONFIGS
CONFIGS[iso3_iso3]="--cache-type-k iso3 --cache-type-v iso3"
CONFIGS[f16_f16]="--cache-type-k f16 --cache-type-v f16"

echo "=================================================================="
echo "  Qwen3.5-27B Token/s Benchmark on RTX 5090 (32GB)"
echo "  Model: Q4_K_M | KV configs: iso3/iso3 vs f16/f16"
echo "=================================================================="
echo ""

# Generate filler text file for large prompts
FILLER="/tmp/bench_filler.txt"
python3 -c "print('The development of large language models has transformed the field of artificial intelligence and natural language processing. ' * 5000)" > "$FILLER"

printf "%-12s %-12s %8s %12s %12s %10s\n" "Config" "Context" "Decode" "Prefill t/s" "Decode t/s" "VRAM"
echo "------------------------------------------------------------------------"

for config_name in iso3_iso3 f16_f16; do
    cache_flags="${CONFIGS[$config_name]}"

    for ctx in "${CONTEXTS[@]}"; do
        # Skip f16 for very large contexts (would OOM)
        if [[ "$config_name" == "f16_f16" && $ctx -gt 32768 ]]; then
            printf "%-12s %-12s %8s %12s %12s %10s\n" "$config_name" "${ctx}" "-" "OOM" "OOM" ">32GB"
            continue
        fi

        # Build prompt from filler, truncated to target context
        PROMPT_FILE="/tmp/bench_prompt_${ctx}.txt"
        head -c $((ctx * 4)) "$FILLER" > "$PROMPT_FILE"  # ~4 chars per token

        # Run llama-cli with timing, generate 32 tokens
        # Use --no-warmup for consistent timing, --temp 0 for determinism
        OUTPUT=$($LLAMA_CLI \
            -m "$MODEL" -ngl 99 \
            $cache_flags \
            -c $((ctx + 256)) \
            -n 32 \
            -f "$PROMPT_FILE" \
            --temp 0 \
            --no-display-prompt \
            --no-warmup \
            2>&1)

        # Parse timing from llama.cpp output
        PREFILL_TPS=$(echo "$OUTPUT" | grep "prompt eval" | grep -oP '[\d.]+(?= tokens per second)')
        DECODE_TPS=$(echo "$OUTPUT" | grep "eval.*tokens per second" | grep -v prompt | grep -oP '[\d.]+(?= tokens per second)')

        # Get VRAM
        VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | tr -d ' ')

        # Fallback parsing if grep missed
        if [ -z "$PREFILL_TPS" ]; then PREFILL_TPS="err"; fi
        if [ -z "$DECODE_TPS" ]; then DECODE_TPS="err"; fi

        printf "%-12s %8d %8d %12s %12s %10s\n" "$config_name" "$ctx" "32" "$PREFILL_TPS" "$DECODE_TPS" "$VRAM"

        # Save raw timing lines
        echo "--- $config_name ctx=$ctx ---" >> "$RESULTS"
        echo "$OUTPUT" | grep -E "eval|token|memory|KV" >> "$RESULTS"
        echo "" >> "$RESULTS"
    done
    echo ""
done

echo ""
echo "Raw timing data saved to: $RESULTS"
echo "Done."
