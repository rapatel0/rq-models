# Sprint 002 Deferred Items

## D-010: Multi-GPU / Tensor Parallelism Support
**What**: Support `--split-mode row` and multi-GPU for models that don't fit on a single card.
**Why deferred**: All 3 models fit on a single 32GB GPU. Multi-GPU adds compose complexity.
**Target sprint**: Sprint 003
**Files**: `docker-compose.yml`, `docker/entrypoint.sh`

## D-011: Web UI (Open WebUI integration)
**What**: Add Open WebUI as a compose service for browser-based chat.
**Why deferred**: API-first approach. Web UI is a separate concern.
**Target sprint**: Sprint 003
**Files**: `docker-compose.yml` (add service)

## D-012: Model TOML Config File
**What**: Replace bash associative arrays with external `models.toml` for easier model additions.
**Why deferred**: 3 models don't justify the parsing complexity. Bash arrays work fine.
**Target sprint**: When model count exceeds 5

## D-013: Automated Benchmark on Docker Build
**What**: Run `benchmark_rotorquant.py` as part of CI/CD to validate each build.
**Why deferred**: Requires GPU in CI. Manual validation sufficient for Sprint 002.
**Target sprint**: Sprint 003

## D-014: Metal Backend for macOS
**What**: Build variant with Metal support for Apple Silicon deployment.
**Why deferred**: RotorQuant has Metal shaders but Docker GPU passthrough is NVIDIA-only.
**Target sprint**: Future (native macOS install script instead)

| Item | Target Sprint | Blocker |
|------|---------------|---------|
| D-010 | Sprint 003 | Need multi-GPU hardware |
| D-011 | Sprint 003 | Sprint 002 API serving |
| D-012 | When >5 models | Sprint 002 complete |
| D-013 | Sprint 003 | GPU CI pipeline |
| D-014 | Future | macOS Docker GPU limitation |
