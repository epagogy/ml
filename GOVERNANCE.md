# Governance

## Decision authority

BDFL model. [Simon Roth](https://epagogy.ai) makes final decisions on scope, API, and releases.

## Contributing

Bug fixes and tests merge without discussion. API changes require an issue and maintainer approval first.

## Scope

**In scope:** split, fit, predict, evaluate, assess, explain, screen, compare, tune, stack, validate, drift, shelf, calibrate, preprocessing, visualization. Tabular data.

**Out of scope:** deep learning, time series, NLP beyond TF-IDF, image classification, distributed training, experiment tracking, feature stores.

Scope changes require a public issue + 14-day comment period. "No" is a complete answer.

## Naming

Single-word verbs (`fit`, `split`, `screen`). Multi-word when needed (`check_data`, `nested_cv`).

## Releases

- **Patch** (1.0.x): bug fixes
- **Minor** (1.x.0): new features, requires issue discussion
- **Major** (x.0.0): API changes, requires extended discussion

## AI assistance

ml was built with AI assistance. Every design decision was made by the maintainer and validated against real problems. All responsibility rests with the maintainer.

## Contact

[Issues](https://github.com/epagogy/ml/issues) or [Discussions](https://github.com/epagogy/ml/discussions). Security: [SECURITY.md](SECURITY.md).
