# ─── verifiable-multi-agent Makefile ───
# All Python commands run inside the vllm conda environment.
#
# Usage:
#   make run TASK="..."     用 Ollama + Qwen3.5:4b 运行任务（默认）
#   make test                测试 LLM 是否正常响应
#   make test-unit           运行单元测试
#   make benchmark            运行 profiler 对比基准测试 (100 tasks)
#   make benchmark-quick      快速基准测试 (10 tasks)
#   make install-dev         安装项目 + dev 依赖

# ─── configurable vars ──────────────────────────────────────
CONDA_ENV    ?= vllm
OLLAMA_URL   ?= http://localhost:11434
OLLAMA_MODEL ?= qwen3.5:4b

# wrap any command inside the conda env
_conda = conda run -n $(CONDA_ENV) --no-capture-output $(1)

# ─── helpers ────────────────────────────────────────────────
_ensure_task = $(if $(TASK),,$(error TASK= is required, e.g. make run TASK="Plan and execute a research task."))

# ─── install ────────────────────────────────────────────────
.PHONY: install
install:
	$(call _conda, pip install -e .)

.PHONY: install-dev
install-dev:
	$(call _conda, pip install -e ".[dev]")

# ─── run ─────────────────────────────────────────────────────
.PHONY: run
run:
	$(call _ensure_task)
	$(call _conda, vma "$(TASK)" --backend ollama --model "$(OLLAMA_MODEL)")

.PHONY: run-vllm
run-vllm:
	$(call _ensure_task)
	$(call _conda, vma "$(TASK)" --backend vllm --base-url http://127.0.0.1:8000/v1)

# ─── tests ───────────────────────────────────────────────────
.PHONY: test
test:
	@echo "[test] probing Ollama at $(OLLAMA_URL) with model $(OLLAMA_MODEL)..."
	$(call _conda, python -c "\
import httpx, json, sys; \
payload = { \
    'model': '$(OLLAMA_MODEL)', \
    'messages': [{'role': 'user', 'content': 'Say hello in one sentence.'}], \
    'think': False, \
    'stream': False, \
    'options': {'temperature': 0, 'num_predict': 64}, \
}; \
try: \
    r = httpx.post('$(OLLAMA_URL)/api/chat', json=payload, timeout=30.0); \
    r.raise_for_status(); \
    data = r.json(); \
    content = data['message']['content'].strip(); \
    eval_count = data.get('eval_count', '?'); \
    print(f'[test] response ({eval_count} tokens): {content}'); \
    if not content: \
        print('[test] FAIL: empty response'); \
        sys.exit(2); \
    print('[test] PASS.'); \
except httpx.ConnectError: \
    print('[test] FAIL: cannot connect to Ollama — is it running?'); \
    sys.exit(1); \
except Exception as exc: \
    print(f'[test] FAIL: {exc}'); \
    sys.exit(3); \
")

.PHONY: test-unit
test-unit:
	$(call _conda, python -m pytest tests/ -v)

.PHONY: test-unit-cov
test-unit-cov:
	$(call _conda, python -m pytest tests/ -v --tb=short)

# ─── benchmark ───────────────────────────────────────────────
.PHONY: benchmark
benchmark:
	$(call _conda, python scripts/benchmark.py)

.PHONY: benchmark-quick
benchmark-quick:
	$(call _conda, python scripts/benchmark.py --sample 10)

.PHONY: benchmark-disagree
benchmark-disagree:
	$(call _conda, python scripts/benchmark.py --disagreements-only)

.PHONY: benchmark-e2e
benchmark-e2e:
	$(call _conda, python scripts/benchmark_e2e.py)

.PHONY: benchmark-e2e-quick
benchmark-e2e-quick:
	$(call _conda, python scripts/benchmark_e2e.py --sample 10)

.PHONY: benchmark-e2e-verbose
benchmark-e2e-verbose:
	$(call _conda, python scripts/benchmark_e2e.py --verbose)

# ─── clean ───────────────────────────────────────────────────
.PHONY: clean
clean:
	rm -rf src/*.egg-info build dist .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
