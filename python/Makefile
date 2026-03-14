PYTHON = /opt/homebrew/bin/python3.11
PYTEST = $(PYTHON) -m pytest
OPTS   = -x -q --timeout=20 --tb=short

# All test files, ordered fast → slow
TEST_FILES = \
	tests/test_version.py \
	tests/test_config.py \
	tests/test_repr.py \
	tests/test_split.py \
	tests/test_evaluate.py \
	tests/test_predict.py \
	tests/test_assess.py \
	tests/test_validate.py \
	tests/test_profile.py \
	tests/test_encode.py \
	tests/test_encode_datetime.py \
	tests/test_scale.py \
	tests/test_impute.py \
	tests/test_tokenize.py \
	tests/test_null_flags.py \
	tests/test_bin.py \
	tests/test_pipeline.py \
	tests/test_normalize.py \
	tests/test_utils.py \
	tests/test_io.py \
	tests/test_fit.py \
	tests/test_explain.py \
	tests/test_compare.py \
	tests/test_screen.py \
	tests/test_drift.py \
	tests/test_shelf.py \
	tests/test_enough.py \
	tests/test_calibrate.py \
	tests/test_report.py \
	tests/test_plot.py \
	tests/test_help.py \
	tests/test_blend.py \
	tests/test_cluster.py \
	tests/test_select.py \
	tests/test_preprocessing.py \
	tests/test_check_data.py \
	tests/test_scoring.py \
	tests/test_roundtrip.py \
	tests/test_edge_cases.py \
	tests/test_balance.py \
	tests/test_leak.py \
	tests/test_predict_intervals.py \
	tests/test_interact.py \
	tests/test_nested.py \
	tests/test_tune.py \
	tests/test_tune_time_budget.py \
	tests/test_stack.py \
	tests/test_harvest.py \
	tests/test_encode.py \
	tests/test_impute.py \
	tests/test_persona_hardening.py \
	tests/test_hardening.py \
	tests/test_benchmark_correctness.py \
	tests/test_chain1_engine_mastery.py \
	tests/test_optimize.py \
	tests/test_performance.py \
	tests/test_integration.py

# ── Safe: one file at a time, full RAM release between each ──────────────────
.PHONY: test-safe
test-safe:
	@echo "Running tests sequentially (safe mode)..."
	@for f in $(TEST_FILES); do \
		[ -f $$f ] || continue; \
		printf "▶ %-45s" $$f; \
		$(PYTEST) $$f $(OPTS) -q --no-header 2>&1 | tail -1; \
	done
	@echo "✓ done"

# ── Batch: 5 files per group, balance speed vs RAM ────────────────────────────
.PHONY: test-batch
test-batch:
	@echo "Running tests in batches of 5..."
	@$(PYTHON) -c "\
import subprocess, sys, os; \
files = [f for f in '''$(TEST_FILES)'''.split() if os.path.exists(f)]; \
batches = [files[i:i+5] for i in range(0, len(files), 5)]; \
failed = []; \
[subprocess.run(['$(PYTEST)'] + b + ['-x', '-q', '--timeout=20', '--tb=short', '--no-header'], \
    check=False).returncode == 0 or failed.append(b) for b in batches]; \
print(f'\n✓ done — {len(failed)} batch(es) failed' if not failed else f'\n✗ {len(failed)} batch(es) failed: {failed}'); \
sys.exit(1 if failed else 0)"

# ── Default: subprocess isolation (safe on any hardware) ──────────────────────
.PHONY: test
test:
	$(PYTHON) safe_test_runner.py

# ── Fast: all at once (server only, 16GB+ with no other load) ─────────────────
.PHONY: test-fast
test-fast:
	$(PYTEST) tests/ $(OPTS)

# ── Lint ─────────────────────────────────────────────────────────────────────
.PHONY: lint
lint:
	$(PYTHON) -m ruff check ml/

# ── Hero bug: Docker 1GB cage on server (proves macbook-safe) ─────────────────
# Not in `check` — run manually: make cage
.PHONY: cage
cage:
	@echo "Syncing to server..."
	rsync -az --delete --exclude='__pycache__' --exclude='.pytest_cache' ml/ server:~/ml_test/ml/
	rsync -az --delete --exclude='__pycache__' --exclude='.pytest_cache' tests/ server:~/ml_test/tests/
	@echo "Building Docker image..."
	ssh server "cd ~/ml_test && sg docker -c 'docker build -t ml-ml-container .'"
	@echo "Running in 1GB cage..."
	ssh server "sg docker -c 'docker run --rm --memory=1g --cpus=1 ml-ml-container'"
	@echo "✓ cage passed — macbook-safe"

# ── Beast: full suite unconstrained (Tier 1) ────────────────────────────────
RSYNC_EXCLUDE = --exclude='__pycache__' --exclude='.pytest_cache' --exclude='*.pyc'
SERVER_ACTIVATE = source ~/ml_test_venv/bin/activate && cd ~/ml_test

.PHONY: server-sync
server-sync:
	rsync -az --delete $(RSYNC_EXCLUDE) ml/ server:~/ml_test/ml/
	rsync -az --delete $(RSYNC_EXCLUDE) tests/ server:~/ml_test/tests/

.PHONY: server
server: server-sync
	ssh server "$(SERVER_ACTIVATE) && python3 -m pytest tests/ --timeout=120 -q --ignore=tests/test_harvest.py --ignore=tests/test_properties.py 2>&1"

# Hypothesis PBT tests run in isolation — OOM if combined with full suite (shap/numba pressure)
.PHONY: test-pbt
test-pbt: server-sync
	ssh server "$(SERVER_ACTIVATE) && python3 -m pytest tests/test_properties.py tests/test_parallelism.py --timeout=300 -v 2>&1"

# ── Fulltest: public + private tests on server ────────────────────────────────
.PHONY: fulltest-sync
fulltest-sync: server-sync
	rsync -az --delete $(RSYNC_EXCLUDE) \
		--exclude='benchmark/' --exclude='parity/' --exclude='dataset_sweep/' \
		../../../private/tests/ server:~/ml_test/private_tests/
	scp tests/conftest.py server:~/ml_test/conftest.py

.PHONY: fulltest-quick
fulltest-quick: fulltest-sync
	ssh server "$(SERVER_ACTIVATE) && python3 -m pytest tests/ private_tests/test_*_edge.py --timeout=120 -q --ignore=tests/test_harvest.py --ignore=tests/test_properties.py 2>&1"

.PHONY: fulltest
fulltest: fulltest-sync
	ssh server "$(SERVER_ACTIVATE) && python3 -m pytest tests/ private_tests/ --timeout=120 -q --ignore=tests/test_harvest.py --ignore=tests/test_properties.py --ignore=private_tests/benchmark --ignore=private_tests/parity --ignore=private_tests/dataset_sweep 2>&1"

# ── Fortress: production-grade test suite (Sessions 4-13) ────────────────────
.PHONY: fortress
fortress: fulltest-sync
	ssh server "$(SERVER_ACTIVATE) && python3 -m pytest \
		tests/test_benchmark_correctness.py \
		tests/test_math_invariants.py \
		tests/test_numerical_stability.py \
		tests/test_adversarial.py \
		tests/test_metamorphic.py \
		tests/test_speed_regression.py \
		tests/test_real_world.py \
		tests/test_properties.py \
		tests/test_parallelism.py \
		tests/test_roundtrip.py \
		tests/test_rust_backend.py \
		-v --timeout=300 --ignore=tests/test_harvest.py 2>&1"

# ── Golden refresh: re-capture pinned values after algorithm changes ──────────
.PHONY: golden-refresh
golden-refresh:
	@echo "Capture golden values on server after rebuilding Rust binary:"
	@echo "  rsync then: ssh server '$(SERVER_ACTIVATE) && python3 -c \\"
	@echo "    import ml, json, numpy as np, pandas as pd; ..."
	@echo "Then update pinned values in tests/test_benchmark_correctness.py golden sections"
	@echo "Commit: test: update golden values for <algorithm> — <reason>"

.PHONY: check
check: lint test-safe

# ── Benchmarks: Grammar of Benchmarks ─────────────────────────────────────
.PHONY: bench-engines bench-parity-bridge bench-dev bench-all

bench-engines: server-sync
	rsync -az $(RSYNC_EXCLUDE) benchmarks/ server:~/ml_test/benchmarks/
	ssh server "$(SERVER_ACTIVATE) && RAYON_NUM_THREADS=1 python3 benchmarks/bench_engines.py --server --seed-instability --json --output benchmarks/engines.json 2>&1"
	rsync server:~/ml_test/benchmarks/engines.json benchmarks/engines.json

bench-parity-bridge:
	RAYON_NUM_THREADS=1 $(PYTHON) benchmarks/bench_parity_bridge.py --json --output benchmarks/parity_bridge.json

bench-dev: server-sync
	rsync -az $(RSYNC_EXCLUDE) benchmarks/ server:~/ml_test/benchmarks/
	ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=20 server "$(SERVER_ACTIVATE) && RAYON_NUM_THREADS=1 python3 benchmarks/bench_dev.py 2>&1"
