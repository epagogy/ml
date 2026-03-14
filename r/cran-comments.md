# CRAN Submission Comments -- ml 1.1.0

## R CMD check results

0 errors | 0 warnings | 2 notes (macOS aarch64 R 4.5.2, `_R_CHECK_FORCE_SUGGESTS_=FALSE`)

**NOTE 1**: New submission.

**NOTE 2**: "unable to verify current time" — network unavailable in the
check environment; the package does not depend on network time.

**WARNING on macOS only**: 'checkbashisms' is not available on macOS.
The `configure` script was verified clean via
`checkbashisms.pl` from Debian devscripts — 0 bashisms found.

## Test environments

- macOS Sequoia 15.6.1 (aarch64), R 4.5.2 — **0 errors | 0 warnings | 2 notes**
- Ubuntu 24.04 (x86_64), R 4.3.3 — 0 errors | 0 warnings | 2 notes (LaTeX/pandoc/tidy are infrastructure, not package)
- GitHub Actions: ubuntu-latest (R 4.3, 4.4, 4.5), macos-latest (R 4.3, 4.4, 4.5)

## Changes since 0.1.0 (addressing reviewer feedback from Konstanze Lauseker)

- **Software name quoting**: 'Rust', 'Python', 'mlw', 'PyPI' are now in
  single quotes in Description; the book title is in double quotes.
- **set.seed()**: All calls replaced with `withr::local_seed()`, which
  restores RNG state on exit. `withr` moved from Suggests to Imports.
- **.GlobalEnv**: Confirmed not written anywhere in the package.

## Possibly misspelled words (from pre-test NOTE)

'Hastie' and 'Tibshirani' are author surnames from the cited reference.
'backends' is standard English for software fallback implementations.

## Package notes

- **Optional Rust backend**: The `configure` script detects a Rust toolchain
  at install time. Without Rust, all algorithms fall back to established
  CRAN packages (ranger, glmnet, xgboost, etc.). The package installs and
  passes all tests in both configurations.

- **Suggested packages**: All algorithm backends are in Suggests. The package
  reports available algorithms via `ml_algorithms()`. Examples using these
  backends are wrapped in `\donttest{}`.

## Downstream dependencies

None (new package).
