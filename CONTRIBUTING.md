# Contributing to ml

Thanks for your interest. ml is a tabular ML library — clean API, reproducible by default.

## Quick start

```bash
git clone https://github.com/mlwork-org/ml
cd ml/ml/python
pip install -e ".[dev,xgboost]"
pytest tests/ -x -v
```

**Python and R contributions are independent.** If you're working on Python, you only need to run Python tests. Same for R. No cross-language setup required.

## What we accept

- **Bug fixes** — always welcome. Include a test that reproduces the bug.
- **Documentation improvements** — typos, clarifications, better examples.
- **New tests** — especially for edge cases.
- **Performance improvements** — with before/after benchmarks.

## What requires discussion first

**API changes** (new verbs, changed signatures, new algorithms) require an issue before a PR. Open an issue, describe the problem you're solving, and wait for maintainer feedback. This keeps the API intentionally narrow.

## Scope

ml does tabular supervised ML. It does not do:
- Deep learning
- Time series forecasting
- Image or audio classification
- Distributed training
- Experiment tracking

If you want to add these, open an issue to discuss. Possible answer: out of scope, but we'll point you to alternatives.

## Pull request checklist

- [ ] `ruff check ml/ tests/` passes (0 errors)
- [ ] `pytest tests/ -x` passes (all green)
- [ ] Tests added for new behavior
- [ ] Docstring updated if signature changed
- [ ] CHANGELOG.md entry added

## Running tests

```bash
# Fast (default — no slow/network tests)
pytest tests/ -x -v

# Include slow tests
pytest tests/ -x -v -m slow

# Single file
pytest tests/test_fit.py -v
```

## Code style

- `ruff` for linting (config in pyproject.toml)
- 100-char line limit
- Google-style docstrings
- Named keyword arguments in tests (never positional for ml.fit(), ml.split(), etc.)

## Questions

Open a [GitHub Discussion](https://github.com/mlwork-org/ml/discussions) — not an issue. Issues are for bugs and confirmed feature requests.
