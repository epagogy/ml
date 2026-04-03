# Papers

Two companion papers on data leakage in machine learning.

## Reproduce

### Option 1: Docker (no local dependencies)

```bash
cd papers
docker build -t ml-papers-verify .
docker run ml-papers-verify
```

### Option 2: Local (Python 3.10+ and [Quarto](https://quarto.org/))

```bash
# 1. Landscape paper (run first — grammar paper depends on its claims.json)
cd landscape
pip install -r requirements.txt
python compile_claims.py    # data/*.jsonl → claims.json
python figures.py           # data/*.jsonl → figures/*.pdf
quarto render paper.qmd     # → paper.pdf

# 2. Grammar paper
cd ../grammar
pip install -r requirements.txt
python compile_claims.py    # reads ../landscape/claims.json
quarto render paper.qmd     # → paper.pdf
```

## Structure

```
landscape/              Leakage landscape study (29 experiments, 2,047 datasets)
  data/                 Raw experiment results (JSONL, ~31MB)
  figures/              Generated publication figures (6 PDFs)
  experiments/          Experiment runners (to rerun from scratch)
  compile_claims.py     data/*.jsonl → claims.json
  figures.py            data/*.jsonl → figures/*.pdf
  claims.py             Quarto inline lookup
  paper.qmd             Paper source

grammar/                ML workflow grammar specification
  compile_claims.py     Extracts subset from landscape claims
  claims.py             Quarto inline lookup
  paper.qmd             Paper source

Dockerfile              Verify both papers end-to-end
```

## Data provenance

Every number in the papers traces to raw experiment data:

```
experiments/*.py → data/*.jsonl → compile_claims.py → claims.json → paper.qmd → paper.pdf
```

| JSONL file | Rows | Source script | Content |
|---|---|---|---|
| leakage_landscape_v1_final.jsonl | 2,288 | run_leakage_landscape.py | Core experiments (A–L) |
| leakage_landscape_v1_extended.jsonl | 2,288 | run_leakage_landscape.py | Extended experiments |
| leakage_landscape_v2.jsonl | 2,047 | run_leakage_landscape.py | Experiments AQ, AC, AI, BB |
| v3/v3_an.jsonl | 866 | run_v3_experiments.py --only an | N-scaling (50–10K) |
| v3/v3_ap.jsonl | 1,965 | run_v3_experiments.py --only ap | Seed dose-response (K=5–100) |
| v3/v3_ao_merged.jsonl | 2,047 | run_v3_experiments.py --only ao | CV coverage gap |
| v3/phase2_ao_v2.jsonl | 3,522 | (phase 2 CI methods) | 6-method CI comparison |

## Rerunning experiments

Requires additional dependencies beyond paper verification:

```bash
cd landscape/experiments
pip install -r requirements.txt    # includes scikit-learn, openml, xgboost
```

Experiments download datasets from OpenML (1,855 datasets), PMLB (79), and ml (113).
Total compute: ~48–72 hours on a server with 64GB+ RAM.

Seeds: `SEED=42`, `CV_FOLDS=5`, `N_REPS=5`. All models use `random_state=SEED`.

## Verification

```bash
# Independent claim verification (no trust in compile_claims.py needed):
python3 -c "
import json, numpy as np
v1 = [json.loads(l) for l in open('landscape/data/leakage_landscape_v1_final.jsonl')]
ok = [r for r in v1 if r.get('status') == 'ok']
diffs = [r['b_infl_k10'] for r in ok if r.get('b_infl_k10') is not None]
print(f'peek.dz = {np.mean(diffs)/np.std(diffs, ddof=1):.3f}')  # should be 0.929
print(f'n_datasets = {len(ok)}')                                  # should be 2047
"
```
