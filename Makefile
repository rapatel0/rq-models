.PHONY: build run-qwen run-qwen36-27b run-reasoning run-gemma stop logs test bench clean

# ── Build ────────────────────────────────────────────────────────────
build:
	docker build -t rotorquant -f docker/Dockerfile .

# ── Run models ───────────────────────────────────────────────────────
run-qwen:
	docker compose --profile qwen up

run-qwen36-27b:
	docker compose --profile qwen36-27b up

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

smoke:
	bash docker/test.sh

# ── Clean ────────────────────────────────────────────────────────────
clean:
	docker compose \
	  --profile qwen --profile qwen36-q3 --profile qwen36-iq3 \
	  --profile qwen36-27b --profile qwen36-27b-q3 --profile qwen36-27b-iq3 \
	  --profile reasoning --profile gemma \
	  down --volumes
	docker rmi rotorquant 2>/dev/null || true
