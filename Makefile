.PHONY: build run-qwen run-qwen36-27b run-qwen36-27b-dflash run-qwen36-dflash run-reasoning run-gemma stop logs test bench bench-dflash bench-dflash-leg smoke clean

# ── Build ────────────────────────────────────────────────────────────
build:
	docker build -t rotorquant -f docker/Dockerfile .

# ── Run models ───────────────────────────────────────────────────────
run-qwen:
	docker compose --profile qwen up

run-qwen36-27b:
	docker compose --profile qwen36-27b up

# ── Sprint 004: DFlash speculative decoding ───────────────────────────
# Both DFlash profiles require host-side opt-in. The 27B dense profile is
# preview (community drafts iterating); the 35B MoE profile is experimental
# (no speedup gate, MoE acceptance characteristics differ from dense).
run-qwen36-27b-dflash:
	PREVIEW=1 docker compose --profile qwen36-27b-dflash up

run-qwen36-dflash:
	EXPERIMENTAL=1 docker compose --profile qwen36-dflash up

run-reasoning:
	docker compose --profile reasoning up

run-gemma:
	docker compose --profile gemma up

# ── Background mode ──────────────────────────────────────────────────
run-qwen-bg:
	docker compose --profile qwen up -d
	@echo "Waiting for server..." && \
	for i in $$(seq 1 150); do \
		curl -sf http://localhost:$${PORT:-8080}/health >/dev/null 2>&1 && echo "Ready on port $${PORT:-8080}" && break; \
		sleep 1; \
	done

run-qwen36-27b-bg:
	docker compose --profile qwen36-27b up -d
	@echo "Waiting for server..." && \
	for i in $$(seq 1 150); do \
		curl -sf http://localhost:$${PORT:-8080}/health >/dev/null 2>&1 && echo "Ready on port $${PORT:-8080}" && break; \
		sleep 1; \
	done

run-qwen36-27b-dflash-bg:
	PREVIEW=1 docker compose --profile qwen36-27b-dflash up -d
	@echo "Waiting for server..." && \
	for i in $$(seq 1 240); do \
		curl -sf http://localhost:$${PORT:-8080}/health >/dev/null 2>&1 && echo "Ready on port $${PORT:-8080}" && break; \
		sleep 1; \
	done

run-qwen36-dflash-bg:
	EXPERIMENTAL=1 docker compose --profile qwen36-dflash up -d
	@echo "Waiting for server..." && \
	for i in $$(seq 1 240); do \
		curl -sf http://localhost:$${PORT:-8080}/health >/dev/null 2>&1 && echo "Ready on port $${PORT:-8080}" && break; \
		sleep 1; \
	done

run-reasoning-bg:
	docker compose --profile reasoning up -d

run-gemma-bg:
	docker compose --profile gemma up -d

# ── Throughput mode (parallel slots) ─────────────────────────────────
run-throughput:
	docker compose --profile qwen-throughput up

run-throughput-bg:
	docker compose --profile qwen-throughput up -d
	@echo "Waiting for server..." && \
	for i in $$(seq 1 150); do \
		curl -sf http://localhost:$${PORT:-8080}/health >/dev/null 2>&1 && echo "Ready on port $${PORT:-8080}" && break; \
		sleep 1; \
	done

# ── Logs ─────────────────────────────────────────────────────────────
logs:
	docker compose \
	  --profile qwen --profile qwen36-q3 --profile qwen36-iq3 \
	  --profile qwen36-27b --profile qwen36-27b-q3 --profile qwen36-27b-iq3 \
	  --profile qwen36-27b-dflash --profile qwen36-dflash \
	  --profile qwen36-throughput \
	  --profile reasoning --profile gemma \
	  --profile qwen-q3 --profile qwen-q3-xxs --profile qwen-iq4 --profile gemma-q3 \
	  --profile qwen-throughput --profile qwen-throughput-q3 \
	  logs -f --tail 50

# ── Stop ─────────────────────────────────────────────────────────────
stop:
	docker compose \
	  --profile qwen --profile qwen36-q3 --profile qwen36-iq3 \
	  --profile qwen36-27b --profile qwen36-27b-q3 --profile qwen36-27b-iq3 \
	  --profile qwen36-27b-dflash --profile qwen36-dflash \
	  --profile qwen36-throughput \
	  --profile reasoning --profile gemma \
	  --profile qwen-q3 --profile qwen-q3-xxs --profile qwen-iq4 --profile gemma-q3 \
	  --profile qwen-throughput --profile qwen-throughput-q3 \
	  down

# ── Test ─────────────────────────────────────────────────────────────
test:
	python -m pytest tests/ -v

bench:
	python scripts/battery_test.py

# ── Sprint 004 L4: 3-way decode tok/s benchmark ──────────────────────
# Reproducibility entrypoint per Phase 6 spec. Operator brings up each
# leg's profile in sequence; the script appends to a single results JSON.
# Usage:
#   make run-qwen36-27b-bg                        ; make bench-dflash-leg LEG=target-only       ; make stop
#   SPECULATIVE_MODE=autoregressive DRAFT_MODEL_NAME=qwen3.6-27b-dflash \
#     docker compose --profile qwen36-27b up -d   ; make bench-dflash-leg LEG=autoregressive    ; make stop
#   make run-qwen36-27b-dflash-bg                 ; make bench-dflash-leg LEG=dflash            ; make stop
#   make bench-dflash
bench-dflash:
	python3 scripts/bench_speculative.py --finalize

bench-dflash-leg:
	@test -n "$(LEG)" || (echo "ERROR: LEG=target-only|autoregressive|dflash required" && exit 1)
	python3 scripts/bench_speculative.py --leg $(LEG)

smoke:
	bash docker/test.sh

# ── Sprint 004 F-001: source-convert DFlash draft GGUFs ──────────────
# Downloads z-lab safetensors + target tokenizer, runs the cherry-picked
# convert_hf_to_gguf.py, and publishes the result into the llm-models
# named volume. Idempotent. The 27B-DFlash repo is gated — request access
# at https://huggingface.co/z-lab/Qwen3.6-27B-DFlash before running.
convert-drafts:
	bash scripts/convert_dflash_drafts.sh

# ── Clean ────────────────────────────────────────────────────────────
clean:
	docker compose \
	  --profile qwen --profile qwen36-q3 --profile qwen36-iq3 \
	  --profile qwen36-27b --profile qwen36-27b-q3 --profile qwen36-27b-iq3 \
	  --profile qwen36-27b-dflash --profile qwen36-dflash \
	  --profile reasoning --profile gemma \
	  down --volumes
	docker rmi rotorquant 2>/dev/null || true
