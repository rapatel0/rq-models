.PHONY: build run-qwen run-reasoning run-gemma stop logs test bench clean

# ── Build ────────────────────────────────────────────────────────────
build:
	docker build -t rotorquant -f docker/Dockerfile .

# ── Run models ───────────────────────────────────────────────────────
run-qwen:
	docker compose --profile qwen up

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

run-reasoning-bg:
	docker compose --profile reasoning up -d

run-gemma-bg:
	docker compose --profile gemma up -d

# ── Throughput mode (parallel slots, 16K context) ────────────────────
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
	docker compose --profile qwen --profile reasoning --profile gemma --profile qwen-q3 --profile qwen-q3-xxs --profile qwen-iq4 --profile gemma-q3 logs -f --tail 50

# ── Stop ─────────────────────────────────────────────────────────────
stop:
	docker compose --profile qwen --profile reasoning --profile gemma down

# ── Test ─────────────────────────────────────────────────────────────
test:
	python -m pytest tests/ -v

bench:
	python scripts/battery_test.py

smoke:
	bash docker/test.sh

# ── Clean ────────────────────────────────────────────────────────────
clean:
	docker compose --profile qwen --profile reasoning --profile gemma down --volumes
	docker rmi rotorquant 2>/dev/null || true
