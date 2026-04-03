# Governance

## Decision authority

ml uses the BDFL model. Simon Roth makes final decisions on scope, API design, and releases. This is not a committee — decisions are fast and accountable.

## Contributing

PRs are welcome. Bug fixes and tests can be merged without discussion. API changes (new verbs, changed signatures, algorithm additions) require an issue and maintainer approval first.

## Scope

ml is a tabular supervised ML library. Its scope is intentionally narrow:

**In scope:** split, fit, predict, evaluate, assess, explain, screen, compare, tune, stack, validate, drift, shelf, calibrate, preprocessing, visualization for tabular data.

**Out of scope:** deep learning, time series forecasting, NLP beyond TF-IDF, image classification, distributed training, experiment tracking, feature stores.

Scope changes require: a public issue + 14-day comment period + maintainer decision. "No" is a complete answer.

## Naming conventions

Verbs follow a single-word naming standard (e.g., `fit`, `split`, `screen`). Multi-word verbs are acceptable when a single word would be ambiguous (e.g., `cluster_features` vs `cluster`, `check_data` vs `check`, `nested_cv` as a fixed technical term).

## Release process

- Patch releases (1.0.x): bug fixes, dependency updates — no discussion needed
- Minor releases (1.x.0): new features — requires issue discussion
- Major releases (x.0.0): API changes — requires extended community discussion

## AI assistance

ml was built with AI assistance. Every design decision, test, and API choice was made by the maintainer and validated against real problems. AI assistance is used in development and code review. All responsibility for the codebase rests with the maintainer.

PRs from contributors are reviewed by the maintainer (with or without AI assistance).

## Contact

Open an issue or Discussion on GitHub. For security issues, see SECURITY.md.
