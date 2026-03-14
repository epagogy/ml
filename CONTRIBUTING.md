# Contributing

## Quick start

```bash
git clone https://github.com/epagogy/ml
cd python
pip install -e ".[dev,xgboost]"
pytest tests/ -x -v
```

Python and R contributions are independent. No cross-language setup required.

## What we accept

- **Bug fixes.** Include a test that reproduces the bug.
- **Documentation.** Typos, clarifications, better examples.
- **New tests.** Especially edge cases.
- **Performance.** With before/after benchmarks.

## What requires discussion first

API changes (new verbs, changed signatures, new algorithms) require an [issue](https://github.com/epagogy/ml/issues) before a PR. The API is intentionally narrow.

## Scope

ml does tabular supervised ML. It does not do deep learning, time series, image/audio, distributed training, or experiment tracking.

## PR checklist

- [ ] `ruff check ml/ tests/` passes
- [ ] `pytest tests/ -x` passes
- [ ] Tests added for new behavior
- [ ] Docstring updated if signature changed
- [ ] CHANGELOG.md entry added

## Tests

```bash
pytest tests/ -x -v               # fast
pytest tests/ -x -v -m slow       # include slow tests
pytest tests/test_fit.py -v       # single file
```

## Style

- `ruff` (config in pyproject.toml)
- 100-char line limit
- Google-style docstrings
- Named keyword arguments in tests

## Questions

[GitHub Discussions](https://github.com/epagogy/ml/discussions), not issues.
