#!/bin/bash
set -euo pipefail

LLAMA_PPL="/home/ravi/repos/turbo/llama-cpp-rq/build/bin/llama-perplexity"
WIKI="/home/ravi/repos/turbo/models/wikitext2-test.txt"
MODEL_DIR="/home/ravi/repos/turbo/models"
RESULTS="/home/ravi/repos/turbo/docs/ppl_sweep_gemma_results.txt"

VARIANTS=(
  "unsloth/gemma-4-26B-A4B-it-GGUF|gemma-4-26B-A4B-it-UD-IQ2_XXS.gguf"
  "unsloth/gemma-4-26B-A4B-it-GGUF|gemma-4-26B-A4B-it-UD-IQ3_XXS.gguf"
  "unsloth/gemma-4-26B-A4B-it-GGUF|gemma-4-26B-A4B-it-UD-Q3_K_XL.gguf"
  "unsloth/gemma-4-26B-A4B-it-GGUF|gemma-4-26B-A4B-it-UD-IQ4_XS.gguf"
  "unsloth/gemma-4-26B-A4B-it-GGUF|gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
)

echo "=================================================================="
echo "  Gemma 4 26B MoE — Perplexity Sweep (imatrix variants)"
echo "  wikitext-2, ctx=2048"
echo "=================================================================="

> "$RESULTS"
echo "# Gemma 4 26B MoE PPL Sweep — $(date)" >> "$RESULTS"
echo "" >> "$RESULTS"

for entry in "${VARIANTS[@]}"; do
  IFS='|' read -r REPO FILE <<< "$entry"
  MODEL_PATH="$MODEL_DIR/$FILE"
  NAME="${FILE%.gguf}"
  NAME="${NAME#gemma-4-26B-A4B-it-}"

  if [ ! -f "$MODEL_PATH" ]; then
    echo "Downloading $FILE..."
    huggingface-cli download "$REPO" "$FILE" --local-dir "$MODEL_DIR" --local-dir-use-symlinks False
  fi

  SIZE=$(du -h "$MODEL_PATH" | cut -f1)
  echo ""
  echo "--- $NAME ($SIZE) ---"

  OUTPUT=$($LLAMA_PPL -m "$MODEL_PATH" -f "$WIKI" -ngl 99 -c 2048 2>&1)

  PPL=$(echo "$OUTPUT" | grep "Final estimate" | grep -oP 'PPL = [\d.]+')
  ERR=$(echo "$OUTPUT" | grep "Final estimate" | grep -oP '\+/- [\d.]+')

  echo "  $PPL $ERR"
  echo "$NAME|$SIZE|$PPL|$ERR" >> "$RESULTS"
done

echo ""
echo "=================================================================="
echo "  RESULTS"
echo "=================================================================="
echo ""
printf "%-16s %8s %12s %10s\n" "Variant" "Size" "PPL" "Error"
printf "%-16s %8s %12s %10s\n" "────────────────" "────────" "────────────" "──────────"
while IFS='|' read -r name size ppl err; do
  [[ "$name" == "#"* ]] && continue
  [[ -z "$name" ]] && continue
  printf "%-16s %8s %12s %10s\n" "$name" "$size" "$ppl" "$err"
done < "$RESULTS"
